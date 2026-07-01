from dataclasses import dataclass
from typing import cast

import pytest
from langchain_core.messages import AIMessage

from paddington.agent.agent_loop import AgentLoop, AgentResult
from paddington.agent.booking_nodes import (
    ROUTE_INFORM_MOVIE_UNAVAILABLE,
    ROUTE_INFORM_NEEDS_RETRY,
    ROUTE_INFORM_THEATER_UNAVAILABLE,
    ROUTE_PRESENT_SEATS,
    ROUTE_PRESENT_SHOWTIMES,
    build_phase_1_node,
    build_phase_3_node,
    route_after_phase_1,
    route_after_phase_3,
)
from paddington.agent.showtime_extraction import ExtractedShowtime, ShowtimeList
from paddington.browser.browser_session import BrowserSession
from paddington.schemas.browser import InteractiveElement, PageSnapshot


async def _fake_extractor(answer: str) -> ShowtimeList:
    """Deterministic Option-B double so the node never makes a live LLM call."""
    return ShowtimeList(
        selected_theater="Andino",
        showtimes=[ExtractedShowtime(time="7:20 P.M.", hall="SALA 4")],
    )


class _FakeSession:
    """Minimal BrowserSession stand-in: the Phase 1 node only reads ``last_snapshot``."""

    def __init__(self, last_snapshot: PageSnapshot | None = None) -> None:
        self.last_snapshot = last_snapshot


def _session(links: list[InteractiveElement] | None = None) -> BrowserSession:
    snapshot = (
        PageSnapshot(
            url="https://www.cinecolombia.com/cinemas/andino/",
            title="",
            markdown="",
            interactive_elements=links,
            truncated=False,
            total_chars=0,
        )
        if links is not None
        else None
    )
    return cast(BrowserSession, _FakeSession(snapshot))


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
    node = build_phase_1_node(fake, _session(), extract_showtimes=_fake_extractor)

    update = await node(_state(), _config())

    assert update["phase_outcome"] == "FOUND_SHOWTIMES"
    assert len(update["messages"]) == 1
    msg = update["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.content == "Here are the showtimes.\nSTATUS: FOUND_SHOWTIMES"
    # On the happy path the node structures + persists the options for Phase 2.
    assert update["selected_theater"] == "Andino"
    assert [o["id"] for o in update["offered_showtimes"]] == ["st_1"]


async def test_node_runs_inner_agent_on_isolated_thread_with_phase_prompt() -> None:
    fake = _FakeAgentLoop("STATUS: FOUND_SHOWTIMES")
    node = build_phase_1_node(fake, _session(), extract_showtimes=_fake_extractor)

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
    node = build_phase_1_node(fake, _session())

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
