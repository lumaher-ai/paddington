from dataclasses import dataclass

from langchain_core.messages import AIMessage

from paddington.agent.agent_loop import AgentResult
from paddington.agent.booking_graph import build_booking_graph, initial_booking_state


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


_CONFIG = {"configurable": {"thread_id": "t1"}}


def _state() -> dict:
    return initial_booking_state(
        movie="Dune 3",
        date="Saturday",
        preferred_multiplexes=["Titan Plaza", "Multiplaza"],
    )


def _contents(final: dict) -> str:
    return "\n".join(
        m.content for m in final["messages"] if isinstance(m, AIMessage)
    )


async def test_found_showtimes_ends_without_inform_message() -> None:
    graph = build_booking_graph(_FakeAgentLoop("Here they are.\nSTATUS: FOUND_SHOWTIMES"))

    final = await graph.ainvoke(_state(), config=_CONFIG)

    assert final["phase_outcome"] == "FOUND_SHOWTIMES"
    # Only the phase's own summary message — no inform node ran on the happy path.
    assert len(final["messages"]) == 1
    assert final["messages"][0].content == "Here they are.\nSTATUS: FOUND_SHOWTIMES"


async def test_movie_not_found_routes_to_inform_movie() -> None:
    graph = build_booking_graph(_FakeAgentLoop("Not listed.\nSTATUS: MOVIE_NOT_FOUND"))

    final = await graph.ainvoke(_state(), config=_CONFIG)

    assert final["phase_outcome"] == "MOVIE_NOT_FOUND"
    assert "isn't currently listed" in _contents(final)


async def test_theater_unavailable_routes_to_inform_theater() -> None:
    graph = build_booking_graph(
        _FakeAgentLoop("No theater.\nSTATUS: THEATER_UNAVAILABLE")
    )

    final = await graph.ainvoke(_state(), config=_CONFIG)

    assert final["phase_outcome"] == "THEATER_UNAVAILABLE"
    contents = _contents(final)
    assert "Titan Plaza, Multiplaza" in contents
    assert "Saturday" in contents


async def test_missing_status_routes_to_inform_needs_retry() -> None:
    graph = build_booking_graph(_FakeAgentLoop("I wandered off without a status."))

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
