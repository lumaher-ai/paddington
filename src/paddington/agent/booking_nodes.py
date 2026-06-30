"""Phase nodes and routers for the booking graph.

A phase node is the bridge between the outer graph's structured state and the inner ReAct
agent. It reads ``BookingState``, builds the phase's system prompt, runs the inner agent,
parses the ``STATUS:`` token from the answer, and writes the outcome back to state. A
matching router then turns that outcome into the name of the next node.

Only Phase 1 lives here today; the ``StateGraph`` that wires these nodes (plus interrupts
and phases 2-6) is a later slice. See docs/phase-based-booking-architecture.md.
"""

from typing import Protocol

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from paddington.agent.agent_loop import AgentLoop
from paddington.agent.booking_state import BookingState
from paddington.agent.phase_prompts import (
    PHASE1_FOUND_SHOWTIMES,
    PHASE1_MOVIE_NOT_FOUND,
    PHASE1_NEEDS_RETRY,
    PHASE1_STATUSES,
    PHASE1_THEATER_UNAVAILABLE,
    parse_phase_status,
    phase1_find_showtimes_prompt,
)


class PhaseNode(Protocol):
    """A phase node: reads state + the run config, returns a partial-state update dict.

    Declared as a Protocol with *named* ``state``/``config`` params (not a bare
    ``Callable``) so it matches LangGraph's ``StateNode`` (``_NodeWithConfig``), whose
    ``__call__`` takes named params — a positional-only ``Callable`` alias fails that
    structural check when passed to ``add_node``.
    """

    async def __call__(self, state: BookingState, config: RunnableConfig) -> dict: ...


# Route names returned by route_after_phase_1. Defined as constants so the future
# StateGraph wires add_node / add_conditional_edges against these exact strings.
ROUTE_PRESENT_SHOWTIMES = "present_showtimes"
ROUTE_INFORM_MOVIE_UNAVAILABLE = "inform_movie_unavailable"
ROUTE_INFORM_THEATER_UNAVAILABLE = "inform_theater_unavailable"
ROUTE_INFORM_NEEDS_RETRY = "inform_needs_retry"


def build_phase_1_node(agent_loop: AgentLoop) -> PhaseNode:
    """Build the Phase 1 node, closing over the request-scoped ``AgentLoop``.

    ``AgentLoop`` is request-scoped — its tools close over the per-thread
    ``BrowserSession`` — so there is no global to import. We follow the codebase's
    closure-factory idiom (``build_browser_tools(session)``) and capture it here.
    """

    async def phase_1_find_showtimes(state: BookingState, config: RunnableConfig) -> dict:
        # 1. READ structured state.
        movie = state["movie"]
        date = state["date"]
        theaters = state["preferred_multiplexes"]

        # 2. BUILD the phase prompt. The theaters are passed as an ordered list; the prompt
        #    resolves each to its /cinemas/<slug>/ URL and renders a fallback chain (try the
        #    first, fall through to the next). Resolving *which* one was available
        #    (selected_theater) is deferred to a later slice.
        prompt = phase1_find_showtimes_prompt(
            movie=movie,
            theaters=theaters,
            date=date,
        )

        # 3. RUN the inner agent on a per-phase isolated thread, so its checkpoint never
        #    collides with the outer booking graph's under the shared thread id. The live
        #    BrowserSession persists across phases regardless of this id.
        outer_thread_id = config.get("configurable", {}).get("thread_id", "default_thread_id")
        result = await agent_loop.run(
            user_message=f"Find showtimes for {movie} on {date} for {theaters}",
            thread_id=f"{outer_thread_id}:p1",
            system_prompt=prompt,
        )

        # 4. PARSE the outcome the LLM declared in its STATUS line.
        outcome = parse_phase_status(result.answer, PHASE1_STATUSES, PHASE1_NEEDS_RETRY)

        # 5. WRITE structured state back. The AIMessage is the outer-graph summary of this
        #    phase, appended via the add_messages reducer for a later interrupt to surface.
        return {
            "phase_outcome": outcome,
            "messages": [AIMessage(content=result.answer)],
        }

    return phase_1_find_showtimes


def route_after_phase_1(state: BookingState) -> str:
    """Map Phase 1's outcome to the next node name (the conditional edge)."""
    outcome = state["phase_outcome"]
    if outcome == PHASE1_FOUND_SHOWTIMES:
        return ROUTE_PRESENT_SHOWTIMES
    if outcome == PHASE1_MOVIE_NOT_FOUND:
        return ROUTE_INFORM_MOVIE_UNAVAILABLE
    if outcome == PHASE1_THEATER_UNAVAILABLE:
        return ROUTE_INFORM_THEATER_UNAVAILABLE
    # NEEDS_RETRY or anything unexpected: terminate safely rather than loop.
    return ROUTE_INFORM_NEEDS_RETRY


# --- Terminal inform nodes -------------------------------------------------------
# The unhappy-path exits from the mermaid diagram. Each is a plain node: it reads
# state and returns a single user-facing AIMessage, then the graph routes to END.
# No browser, no LLM — they just tell the user what happened.


async def inform_movie_unavailable(state: BookingState) -> dict:
    """Phase 1 could not find the movie in the listing."""
    return {
        "messages": [
            AIMessage(
                content=(
                    f"Sorry — '{state['movie']}' isn't currently listed. "
                    f"Try a different title or date."
                )
            )
        ]
    }


async def inform_theater_unavailable(state: BookingState) -> dict:
    """Phase 1 found the movie, but not at any preferred theater for the date."""
    theaters = ", ".join(state["preferred_multiplexes"])
    return {
        "messages": [
            AIMessage(
                content=(
                    f"'{state['movie']}' isn't showing at {theaters} on {state['date']}. "
                    f"Try another theater or date."
                )
            )
        ]
    }


async def inform_needs_retry(state: BookingState) -> dict:
    """Phase 1 ended without a usable outcome (no/invalid STATUS)."""
    return {
        "messages": [AIMessage(content="Something went wrong finding showtimes. Please try again.")]
    }


def build_phase_2_node() -> PhaseNode:

    async def phase_2_present_showtimes(state: BookingState, config: RunnableConfig) -> dict:
        # The showtimes are in Phase 1's answer (last AIMessage in messages)
        last_answer = state["messages"][-1].content

        # Pause the graph and present options to the user
        user_choice = interrupt(
            {
                "question": "Which showtime would you like?",
                "showtimes": last_answer,
            }
        )

        # Graph resumes here when the user responds
        return {
            "chosen_showtime": user_choice,
            "phase": "get_to_seats",
        }

    return phase_2_present_showtimes
