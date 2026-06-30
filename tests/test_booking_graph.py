from dataclasses import dataclass
from typing import cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from paddington.agent.agent_loop import AgentLoop, AgentResult
from paddington.agent.booking_graph import build_booking_graph, initial_booking_state
from paddington.agent.booking_state import BookingState


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


_CONFIG: RunnableConfig = {"configurable": {"thread_id": "t1"}}


def _graph(answer: str):
    """Build the booking graph over a fake agent loop and an in-memory checkpointer.

    ``_FakeAgentLoop`` is a structural stand-in (it implements ``run``), so cast it to
    ``AgentLoop`` to satisfy the static signature without subclassing the real loop.
    """
    return build_booking_graph(
        cast(AgentLoop, _FakeAgentLoop(answer)),
        checkpointer=MemorySaver(),
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


async def test_found_showtimes_ends_without_inform_message() -> None:
    graph = _graph("Here they are.\nSTATUS: FOUND_SHOWTIMES")

    final = await graph.ainvoke(_state(), config=_CONFIG)

    assert final["phase_outcome"] == "FOUND_SHOWTIMES"
    # Happy path now flows into Phase 2, which interrupts for the user's showtime
    # choice rather than ending. State carries an __interrupt__ and no inform message.
    assert "__interrupt__" in final
    assert len(final["messages"]) == 1
    assert final["messages"][0].content == "Here they are.\nSTATUS: FOUND_SHOWTIMES"


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
