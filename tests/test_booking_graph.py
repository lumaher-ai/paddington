from dataclasses import dataclass
from typing import cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from paddington.agent.agent_loop import AgentLoop, AgentResult
from paddington.agent.booking_graph import build_booking_graph, initial_booking_state
from paddington.agent.booking_state import BookingState
from paddington.agent.showtime_extraction import ExtractedShowtime, ShowtimeList


@dataclass
class _FakeAgentLoop:
    """Stand-in for AgentLoop: returns a canned answer so the graph can run end-to-end."""

    answer: str

    async def run(self, **kwargs) -> AgentResult:
        return AgentResult(
            answer=self.answer,
            iterations=1,
            tools_used=[],
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=0.0,
        )


# Deterministic Option-B extractor double: skips the live LLM and returns two showtimes
# at the Andino theater, so Phase 1 persists ``offered_showtimes`` (st_1, st_2).
async def _fake_extractor(answer: str) -> ShowtimeList:
    return ShowtimeList(
        selected_theater="Andino",
        showtimes=[
            ExtractedShowtime(time="7:20 P.M.", hall="SALA 4", format="2D", language="Subtitled"),
            ExtractedShowtime(time="9:50 P.M.", hall="SALA 1"),
        ],
    )


_CONFIG: RunnableConfig = {"configurable": {"thread_id": "t1"}}


def _graph(answer: str, extractor=_fake_extractor):
    """Build the booking graph over a fake agent loop and an in-memory checkpointer.

    ``_FakeAgentLoop`` is a structural stand-in (it implements ``run``), so cast it to
    ``AgentLoop`` to satisfy the static signature without subclassing the real loop. The
    Option-B extractor is injected so Phase 1 never makes a live LLM call in tests.
    """
    return build_booking_graph(
        cast(AgentLoop, _FakeAgentLoop(answer)),
        checkpointer=MemorySaver(),
        extract_showtimes=extractor,
    )


def _state() -> BookingState:
    return initial_booking_state(
        movie="Dune 3",
        date="Saturday",
        preferred_multiplexes=["Titan Plaza", "Multiplaza"],
    )


def _contents(final: dict) -> str:
    # AIMessage.content is typed ``str | list``; these fakes only ever set str, so
    # coerce for join's str-iterable overload.
    return "\n".join(
        str(m.content) for m in final["messages"] if isinstance(m, AIMessage)
    )


_FOUND_ANSWER = "Showtimes for the movie at Andino: 7:20 PM, 9:50 PM.\nSTATUS: FOUND_SHOWTIMES"


async def test_found_showtimes_interrupts_with_structured_options() -> None:
    graph = _graph(_FOUND_ANSWER)

    final = await graph.ainvoke(_state(), config=_CONFIG)

    assert final["phase_outcome"] == "FOUND_SHOWTIMES"
    # Phase 1 structured + persisted the options; selected_theater came from extraction.
    assert final["selected_theater"] == "Andino"
    assert [o["id"] for o in final["offered_showtimes"]] == ["st_1", "st_2"]
    # Happy path flows into Phase 2, which interrupts rather than ending — one phase
    # summary message, no inform message yet.
    assert len(final["messages"]) == 1
    assert "__interrupt__" in final

    # The interrupt payload is structured (discriminated, with labelled options) and
    # does NOT leak the internal STATUS token.
    payload = final["__interrupt__"][0].value
    assert payload["kind"] == "present_showtimes"
    assert payload["allow_reject"] is True
    assert [o["id"] for o in payload["options"]] == ["st_1", "st_2"]
    assert payload["options"][0]["label"] == "7:20 P.M. · SALA 4 · 2D · Subtitled"
    assert "STATUS" not in str(payload)


async def test_selecting_a_showtime_sets_chosen_and_routes_to_phase_3() -> None:
    graph = _graph(_FOUND_ANSWER)

    await graph.ainvoke(_state(), config=_CONFIG)
    final = await graph.ainvoke(
        Command(resume={"action": "select", "showtime_id": "st_2"}), config=_CONFIG
    )

    assert "__interrupt__" not in final
    assert final["phase_outcome"] == "SHOWTIME_CHOSEN"
    # chosen_showtime is the human label (Phase 3 re-grounds on it), not the id.
    assert final["chosen_showtime"] == "9:50 P.M. · SALA 1"
    # Routed to the Phase 3 placeholder (END) — no inform message was appended.
    assert len(final["messages"]) == 1


async def test_rejecting_all_showtimes_routes_to_inform_no_showtime() -> None:
    graph = _graph(_FOUND_ANSWER)

    await graph.ainvoke(_state(), config=_CONFIG)
    final = await graph.ainvoke(Command(resume={"action": "reject"}), config=_CONFIG)

    assert "__interrupt__" not in final
    assert final["phase_outcome"] == "NO_SHOWTIME"
    assert final["chosen_showtime"] is None
    assert "won't book any of those" in _contents(final)


async def test_invalid_resume_id_reprompts_then_accepts_valid_choice() -> None:
    graph = _graph(_FOUND_ANSWER)

    await graph.ainvoke(_state(), config=_CONFIG)

    # An unknown id is untrusted input: the node re-interrupts rather than trusting it.
    reprompt = await graph.ainvoke(
        Command(resume={"action": "select", "showtime_id": "st_99"}), config=_CONFIG
    )
    assert "__interrupt__" in reprompt
    assert reprompt["__interrupt__"][0].value["error"]

    # A subsequent valid choice resolves the loop.
    final = await graph.ainvoke(
        Command(resume={"action": "select", "showtime_id": "st_1"}), config=_CONFIG
    )
    assert "__interrupt__" not in final
    assert final["phase_outcome"] == "SHOWTIME_CHOSEN"
    assert final["chosen_showtime"] == "7:20 P.M. · SALA 4 · 2D · Subtitled"


async def test_extraction_failure_downgrades_to_needs_retry() -> None:
    async def _boom(answer: str) -> ShowtimeList:
        raise ValueError("could not parse")

    graph = _graph(_FOUND_ANSWER, extractor=_boom)

    final = await graph.ainvoke(_state(), config=_CONFIG)

    # A FOUND outcome with nothing to present is downgraded so the user is told to retry,
    # rather than reaching an interrupt with zero options.
    assert "__interrupt__" not in final
    assert final["phase_outcome"] == "NEEDS_RETRY"
    assert "Please try again" in _contents(final)


async def test_movie_not_found_routes_to_inform_movie() -> None:
    graph = _graph("Not listed.\nSTATUS: MOVIE_NOT_FOUND")

    final = await graph.ainvoke(_state(), config=_CONFIG)

    assert final["phase_outcome"] == "MOVIE_NOT_FOUND"
    assert "isn't currently listed" in _contents(final)


async def test_theater_unavailable_routes_to_inform_theater() -> None:
    graph = _graph("No theater.\nSTATUS: THEATER_UNAVAILABLE")

    final = await graph.ainvoke(_state(), config=_CONFIG)

    assert final["phase_outcome"] == "THEATER_UNAVAILABLE"
    contents = _contents(final)
    assert "Titan Plaza, Multiplaza" in contents
    assert "Saturday" in contents


async def test_missing_status_routes_to_inform_needs_retry() -> None:
    graph = _graph("I wandered off without a status.")

    final = await graph.ainvoke(_state(), config=_CONFIG)

    assert final["phase_outcome"] == "NEEDS_RETRY"
    assert "Please try again" in _contents(final)


def test_initial_booking_state_seeds_defaults_and_none_fields() -> None:
    state = initial_booking_state(
        movie="Dune 3", date="Saturday", preferred_multiplexes=["Titan Plaza"]
    )

    assert state["seat_quantity"] == 2
    assert state["phase"] == "find_showtimes"
    assert state["phase_outcome"] is None
    assert state["selected_theater"] is None
    assert state["chosen_showtime"] is None
    assert state["seat_section"] is None
    assert state["chosen_seats"] is None
    assert state["payment_link"] is None
