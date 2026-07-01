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
    PHASE2_NO_SHOWTIME,
    PHASE2_SHOWTIME_CHOSEN,
    PHASE3_NEEDS_RETRY,
    PHASE3_SEAT_MAP_VISIBLE,
    PHASE3_STATUSES,
    parse_phase_status,
    phase1_find_showtimes_prompt,
    phase3_get_to_seats_prompt,
)
from paddington.agent.showtime_extraction import (
    ShowtimeExtractor,
    default_showtime_extractor,
    offered_options,
)
from paddington.browser.browser_session import BrowserSession
from paddington.logging_config import get_logger

logger = get_logger(__name__)


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

# Route names returned by route_after_phase_2.
ROUTE_GET_TO_SEATS = "get_to_seats"  # Phase 3 node
ROUTE_INFORM_NO_SHOWTIME = "inform_no_showtime"

# Route names returned by route_after_phase_3. ROUTE_PRESENT_SEATS is Phase 4 (seat
# selection); it maps to END as a placeholder until that slice lands.
ROUTE_PRESENT_SEATS = "present_seats"


def build_phase_1_node(
    agent_loop: AgentLoop,
    session: BrowserSession,
    extract_showtimes: ShowtimeExtractor | None = None,
) -> PhaseNode:
    """Build the Phase 1 node, closing over the request-scoped ``AgentLoop``.

    ``AgentLoop`` is request-scoped — its tools close over the per-thread
    ``BrowserSession`` — so there is no global to import. We follow the codebase's
    closure-factory idiom (``build_browser_tools(session)``) and capture it here.

    ``session`` is that same ``BrowserSession`` (the inner agent's tools close over it).
    After the inner agent reveals the showtimes, its links live on ``session.last_snapshot``;
    we read them here to stamp each option with a code-owned seat-selection ``url`` — the
    LLM never transcribes URLs (see ``offered_options`` / ``_match_url``).

    ``extract_showtimes`` is the Option-B structuring step (prose -> typed showtimes);
    it is injected so tests can supply a deterministic double instead of a live LLM call.
    When omitted, the production extractor is built lazily from the agent loop's model the
    first time a happy-path answer actually needs structuring (so a node that never hits
    FOUND_SHOWTIMES never constructs an LLM).
    """
    extractor = extract_showtimes

    def _extractor() -> ShowtimeExtractor:
        nonlocal extractor
        if extractor is None:
            extractor = default_showtime_extractor(agent_loop.model)
        return extractor

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
        update: dict = {
            "phase_outcome": outcome,
            "messages": [AIMessage(content=result.answer)],
        }

        # 6. STRUCTURE (Option B). Only on the happy path: turn the prose answer into a
        #    typed showtime list and persist it, so Phase 2 can present clean options and
        #    validate the user's choice without re-running an LLM on resume. The extractor
        #    fails loud (raises) — we catch and downgrade to NEEDS_RETRY rather than carry
        #    a FOUND outcome with nothing to present.
        if outcome == PHASE1_FOUND_SHOWTIMES:
            try:
                extraction = await _extractor()(result.answer)
                # The links the agent last saw carry the real seat-selection hrefs; match
                # each screening to its URL here (code-owned, never LLM-transcribed).
                links = session.last_snapshot.interactive_elements if session.last_snapshot else []
                options = offered_options(extraction, links)
                if not options:
                    raise ValueError("extraction returned no showtimes")
            except Exception as exc:  # noqa: BLE001 — any failure is a retry signal
                logger.warning("phase1_showtime_extraction_failed", error=str(exc))
                update["phase_outcome"] = PHASE1_NEEDS_RETRY
            else:
                update["selected_theater"] = extraction.selected_theater
                update["offered_showtimes"] = options

        return update

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
    """Build the Phase 2 node — present Phase 1's showtimes and pause for the user's choice.

    The node is a *pure read* over ``offered_showtimes`` (persisted by Phase 1): on
    resume it re-runs from the top, so it must not re-derive options with an LLM or the
    ids it validates against would drift. It surfaces a structured, discriminated
    interrupt payload (no ``STATUS:`` leak), then maps the resume value to a Phase 2
    outcome token for ``route_after_phase_2``.
    """

    async def phase_2_present_showtimes(state: BookingState, config: RunnableConfig) -> dict:
        offered = state.get("offered_showtimes") or []
        valid_ids = {opt["id"] for opt in offered}
        # Only the selection boundary crosses the interrupt: id + human label. The
        # raw answer (with its STATUS line and agent chatter) stays out of the payload.
        payload = {
            "kind": "present_showtimes",  # discriminator: which interrupt is this
            "prompt": "Which showtime would you like?",
            "options": [{"id": opt["id"], "label": opt["label"]} for opt in offered],
            "allow_reject": True,
        }

        # Validation-retry loop: the resume value is untrusted input crossing the API
        # boundary. A malformed/unknown choice re-interrupts (re-prompts) rather than
        # being trusted or silently dropped. On each resume the node replays from the
        # top; earlier interrupts return their prior resume values and the loop reaches
        # the still-pending one with the new value.
        attempt = 0
        while True:
            attempt += 1
            response = interrupt(payload)
            action = response.get("action") if isinstance(response, dict) else None

            if action == "reject":
                return {"phase_outcome": PHASE2_NO_SHOWTIME}

            if action == "select":
                chosen_id = response.get("showtime_id")
                if chosen_id in valid_ids:
                    chosen = next(opt for opt in offered if opt["id"] == chosen_id)
                    return {
                        "phase_outcome": PHASE2_SHOWTIME_CHOSEN,
                        # The label is Phase 3's re-grounding fallback (a durable human
                        # descriptor, not the round-trip id); the code-owned url is its
                        # fast path straight to the seat map (None if no link matched).
                        "chosen_showtime": chosen["label"],
                        "chosen_showtime_url": chosen.get("url"),
                    }

            # An invalid choice is handled gracefully (we re-prompt), so this is debug,
            # not warning. Note it re-fires on every later resume: the loop replays from
            # the top, so an already-rejected value is re-seen before the loop advances to
            # the new pending interrupt. ``attempt`` makes those replays distinguishable.
            logger.debug("phase2_invalid_resume_value", attempt=attempt, response=str(response))
            payload = {**payload, "error": "Please choose one of the offered options."}

    return phase_2_present_showtimes


def route_after_phase_2(state: BookingState) -> str:
    """Map Phase 2's outcome to the next node name (the conditional edge)."""
    if state["phase_outcome"] == PHASE2_NO_SHOWTIME:
        return ROUTE_INFORM_NO_SHOWTIME
    # SHOWTIME_CHOSEN -> Phase 3 (get-to-seats).
    return ROUTE_GET_TO_SEATS


def build_phase_3_node(agent_loop: AgentLoop) -> PhaseNode:
    """Build the Phase 3 node — drive the inner agent from the chosen showtime to the seat map.

    A thin navigation phase: it hands the inner agent the chosen showtime's code-owned
    seat URL (fast path) and human label (fallback), plus the theater list for the fallback
    route, then lets the agent navigate, clear the guest-checkout interstitial, and confirm
    the seat map. Mirrors the Phase 1 node's shape (build prompt -> run inner agent on an
    isolated ``:p3`` thread -> parse the STATUS token -> write the outcome back).
    """

    async def phase_3_get_to_seats(state: BookingState, config: RunnableConfig) -> dict:
        chosen_label = state["chosen_showtime"] or ""
        seat_url = state.get("chosen_showtime_url")
        theaters = state["preferred_multiplexes"]

        prompt = phase3_get_to_seats_prompt(
            chosen_label=chosen_label,
            seat_url=seat_url,
            theaters=theaters,
        )

        outer_thread_id = config.get("configurable", {}).get("thread_id", "default_thread_id")
        result = await agent_loop.run(
            user_message=f"Get to the seat map for the chosen showtime: {chosen_label}",
            thread_id=f"{outer_thread_id}:p3",
            system_prompt=prompt,
        )

        outcome = parse_phase_status(result.answer, PHASE3_STATUSES, PHASE3_NEEDS_RETRY)
        return {
            "phase_outcome": outcome,
            "messages": [AIMessage(content=result.answer)],
        }

    return phase_3_get_to_seats


def route_after_phase_3(state: BookingState) -> str:
    """Map Phase 3's outcome to the next node name (the conditional edge)."""
    if state["phase_outcome"] == PHASE3_SEAT_MAP_VISIBLE:
        # -> Phase 4 (seat selection); placeholder END until that slice lands.
        return ROUTE_PRESENT_SEATS
    return ROUTE_INFORM_NEEDS_RETRY


async def inform_no_showtime(state: BookingState) -> dict:
    """The user rejected every offered showtime — acknowledge and end (mermaid edge E3)."""
    return {
        "messages": [
            AIMessage(
                content=(
                    "No problem — I won't book any of those showtimes. "
                    "Let me know if you'd like to try a different movie, date, or theater."
                )
            )
        ]
    }
