from dataclasses import dataclass

import pytest
from langchain_core.messages import AIMessage

from paddington.agent.agent_loop import AgentResult
from paddington.agent.booking_nodes import (
    ROUTE_INFORM_MOVIE_UNAVAILABLE,
    ROUTE_INFORM_NEEDS_RETRY,
    ROUTE_INFORM_THEATER_UNAVAILABLE,
    ROUTE_PRESENT_SHOWTIMES,
    build_phase_1_node,
    route_after_phase_1,
)


@dataclass
class _FakeAgentLoop:
    """Stand-in for AgentLoop: records run() kwargs and returns a canned answer."""

    answer: str
    calls: list[dict]

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = []

    async def run(self, **kwargs) -> AgentResult:
        self.calls.append(kwargs)
        return AgentResult(
            answer=self.answer,
            iterations=1,
            tools_used=[],
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=0.0,
        )


def _state() -> dict:
    return {
        "movie": "Dune 3",
        "date": "Saturday",
        "preferred_multiplexes": ["Titan Plaza", "Multiplaza"],
        "phase_outcome": None,
    }


def _config(thread_id: str = "user-1:thread-9") -> dict:
    return {"configurable": {"thread_id": thread_id}}


async def test_node_parses_found_showtimes_and_returns_summary_message() -> None:
    fake = _FakeAgentLoop("Here are the showtimes.\nSTATUS: FOUND_SHOWTIMES")
    node = build_phase_1_node(fake)

    update = await node(_state(), _config())

    assert update["phase_outcome"] == "FOUND_SHOWTIMES"
    assert len(update["messages"]) == 1
    msg = update["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.content == "Here are the showtimes.\nSTATUS: FOUND_SHOWTIMES"


async def test_node_runs_inner_agent_on_isolated_thread_with_phase_prompt() -> None:
    fake = _FakeAgentLoop("STATUS: FOUND_SHOWTIMES")
    node = build_phase_1_node(fake)

    await node(_state(), _config("user-1:thread-9"))

    assert len(fake.calls) == 1
    call = fake.calls[0]
    # Per-phase isolated thread id (decision in the plan).
    assert call["thread_id"] == "user-1:thread-9:p1"
    # The phase prompt carries the booking params, including both preferred theaters.
    assert "Dune 3" in call["system_prompt"]
    assert "Saturday" in call["system_prompt"]
    assert "Titan Plaza" in call["system_prompt"]
    assert "Multiplaza" in call["system_prompt"]


async def test_node_falls_back_to_needs_retry_when_status_missing() -> None:
    fake = _FakeAgentLoop("I navigated around but did not report a status.")
    node = build_phase_1_node(fake)

    update = await node(_state(), _config())

    assert update["phase_outcome"] == "NEEDS_RETRY"


@pytest.mark.parametrize(
    ("outcome", "expected_route"),
    [
        ("FOUND_SHOWTIMES", ROUTE_PRESENT_SHOWTIMES),
        ("MOVIE_NOT_FOUND", ROUTE_INFORM_MOVIE_UNAVAILABLE),
        ("THEATER_UNAVAILABLE", ROUTE_INFORM_THEATER_UNAVAILABLE),
        ("NEEDS_RETRY", ROUTE_INFORM_NEEDS_RETRY),
        (None, ROUTE_INFORM_NEEDS_RETRY),
        ("SOMETHING_UNEXPECTED", ROUTE_INFORM_NEEDS_RETRY),
    ],
)
def test_route_after_phase_1(outcome: str | None, expected_route: str) -> None:
    assert route_after_phase_1({"phase_outcome": outcome}) == expected_route
