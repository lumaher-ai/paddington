from dataclasses import dataclass
from typing import cast

import pytest
from langchain_core.messages import AIMessage

from paddington.agent.agent_loop import AgentResult
from paddington.agent.booking_graph import initial_booking_state
from paddington.agent.booking_nodes import (
    ROUTE_CHECKOUT,
    ROUTE_INFORM_MOVIE_UNAVAILABLE,
    ROUTE_INFORM_NEEDS_RETRY,
    ROUTE_INFORM_NO_SEATS,
    ROUTE_INFORM_THEATER_UNAVAILABLE,
    ROUTE_PRESENT_SEATS,
    ROUTE_PRESENT_SHOWTIMES,
    ROUTE_SELECT_SEATS,
    build_phase_1_node,
    build_phase_3_node,
    build_phase_4_node,
    build_phase_5_node,
    route_after_phase_1,
    route_after_phase_3,
    route_after_phase_4,
    route_after_phase_5,
)
from paddington.agent.booking_state import BookingState
from paddington.agent.showtime_extraction import ExtractedShowtime, ShowtimeList
from paddington.browser.browser_session import BrowserSession
from paddington.schemas.browser import InteractiveElement, PageSnapshot


async def _fake_extractor(answer: str) -> ShowtimeList:
    """Deterministic Option-B double so the node never makes a live LLM call."""
    return ShowtimeList(
        selected_theater="Andino",
        showtimes=[ExtractedShowtime(time="7:20 P.M.", hall="SALA 4")],
    )


def _snapshot(elements: list[InteractiveElement] | None) -> PageSnapshot | None:
    if elements is None:
        return None
    return PageSnapshot(
        url="https://www.cinecolombia.com/cinemas/andino/",
        title="",
        markdown="",
        interactive_elements=elements,
        truncated=False,
        total_chars=0,
    )


class _FakeSession:
    """BrowserSession stand-in. Phase 1 reads ``last_snapshot`` (showtime links); Phase 3
    calls ``get_snapshot()`` for a fresh seat-page read."""

    def __init__(
        self,
        last_snapshot: PageSnapshot | None = None,
        seat_snapshot: PageSnapshot | None = None,
    ) -> None:
        self.last_snapshot = last_snapshot
        self._seat_snapshot = seat_snapshot

    async def get_snapshot(self, *args, **kwargs) -> PageSnapshot:
        if self._seat_snapshot is not None:
            return self._seat_snapshot
        return PageSnapshot(
            url="",
            title="",
            markdown="",
            interactive_elements=[],
            truncated=False,
            total_chars=0,
        )


def _session(
    links: list[InteractiveElement] | None = None,
    seats: list[InteractiveElement] | None = None,
) -> BrowserSession:
    return cast(BrowserSession, _FakeSession(_snapshot(links), _snapshot(seats)))


@dataclass
class _FakeAgentLoop:
    """Stand-in for AgentLoop: records run() kwargs and returns a canned answer.

    Pass ``raises`` to simulate the inner agent failing (e.g. hitting the recursion limit)
    so the phase node's error handling can be exercised without a live loop.
    """

    answer: str
    calls: list[dict]

    def __init__(self, answer: str, raises: Exception | None = None) -> None:
        self.answer = answer
        self.calls = []
        self._raises = raises

    async def run(self, **kwargs) -> AgentResult:
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
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


# --- Phase 1: code-owned URL capture -----------------------------------------

_SEAT_URL = "https://multiplex.cinecolombia.com/order/showtimes/6493-6874/seats"


async def test_node_stamps_showtime_url_from_matching_snapshot_link() -> None:
    """The node pulls each screening's real href from the last snapshot — never the LLM."""
    fake = _FakeAgentLoop("Found them.\nSTATUS: FOUND_SHOWTIMES")
    # The showtimes page the agent last read: a link whose name matches the screening.
    links = [
        InteractiveElement(ref="el_1", role="link", name="7:20 P.M. SALA 4", href=_SEAT_URL),
        InteractiveElement(ref="el_2", role="button", name="Menu", href=None),
    ]
    node = build_phase_1_node(fake, _session(links), extract_showtimes=_fake_extractor)

    update = await node(_state(), _config())

    assert update["offered_showtimes"][0]["url"] == _SEAT_URL


async def test_node_leaves_url_none_when_no_link_matches() -> None:
    """No matching link (or no snapshot) → url is None; Phase 3 falls back to the label."""
    fake = _FakeAgentLoop("Found them.\nSTATUS: FOUND_SHOWTIMES")
    links = [InteractiveElement(ref="el_1", role="link", name="Some other film", href=_SEAT_URL)]
    node = build_phase_1_node(fake, _session(links), extract_showtimes=_fake_extractor)

    update = await node(_state(), _config())

    assert update["offered_showtimes"][0]["url"] is None


# --- Phase 3: get to the seat map + capture seats ----------------------------


def _phase3_state(url: str | None = _SEAT_URL) -> dict:
    return {
        "chosen_showtime": "7:20 P.M. · SALA 4 · 2D · Subtitled",
        "chosen_showtime_url": url,
        "preferred_multiplexes": ["Andino", "Avenida Chile"],
        "phase_outcome": None,
    }


def _seat_elements(available: bool = True) -> list[InteractiveElement]:
    """A tiny seat-map snapshot: two seats (available or all taken) + one non-seat button."""
    a1 = "Silla A1" if available else "Silla no disponible A1"
    a2 = "Silla A2" if available else "Silla no disponible A2"
    return [
        InteractiveElement(ref="el_1", role="button", name="Iniciar sesión"),
        InteractiveElement(ref="el_2", role="button", name=a1),
        InteractiveElement(ref="el_3", role="button", name=a2),
    ]


async def test_phase_3_reaches_seat_map_and_captures_seats() -> None:
    fake = _FakeAgentLoop("The seat map is on screen.\nSTATUS: SEAT_MAP_VISIBLE")
    node = build_phase_3_node(fake, _session(seats=_seat_elements()))

    update = await node(_phase3_state(), _config("user-1:thread-9"))

    assert update["phase_outcome"] == "SEAT_MAP_VISIBLE"
    assert len(update["messages"]) == 1
    # It parsed + persisted the seats (non-seat button dropped) for Phase 4.
    assert [s["label"] for s in update["offered_seats"]] == ["A1", "A2"]
    # Runs on its own per-phase thread and is handed the direct seat URL (fast path).
    call = fake.calls[0]
    assert call["thread_id"] == "user-1:thread-9:p3"
    assert _SEAT_URL in call["system_prompt"]
    assert "Comprar sin registrarse" in call["system_prompt"]


async def test_phase_3_no_available_seats_downgrades_to_needs_retry() -> None:
    # Agent reported the map visible, but every parsed seat is taken → NEEDS_RETRY, no offer.
    fake = _FakeAgentLoop("STATUS: SEAT_MAP_VISIBLE")
    node = build_phase_3_node(fake, _session(seats=_seat_elements(available=False)))

    update = await node(_phase3_state(), _config())

    assert update["phase_outcome"] == "NEEDS_RETRY"
    assert "offered_seats" not in update


async def test_phase_3_without_url_falls_back_to_label_route() -> None:
    fake = _FakeAgentLoop("STATUS: SEAT_MAP_VISIBLE")
    node = build_phase_3_node(fake, _session(seats=_seat_elements()))

    await node(_phase3_state(url=None), _config())

    prompt = fake.calls[0]["system_prompt"]
    # No URL to navigate to → the prompt re-grounds by label on the theater page.
    assert _SEAT_URL not in prompt
    assert "7:20 P.M. · SALA 4 · 2D · Subtitled" in prompt
    assert "Andino" in prompt


async def test_phase_3_missing_status_downgrades_to_needs_retry() -> None:
    fake = _FakeAgentLoop("I got lost on the way to the seats.")
    node = build_phase_3_node(fake, _session())

    update = await node(_phase3_state(), _config())

    assert update["phase_outcome"] == "NEEDS_RETRY"


@pytest.mark.parametrize(
    ("outcome", "expected_route"),
    [
        ("SEAT_MAP_VISIBLE", ROUTE_PRESENT_SEATS),
        ("NEEDS_RETRY", ROUTE_INFORM_NEEDS_RETRY),
        (None, ROUTE_INFORM_NEEDS_RETRY),
    ],
)
def test_route_after_phase_3(outcome: str | None, expected_route: str) -> None:
    assert route_after_phase_3({"phase_outcome": outcome}) == expected_route


# --- Phase 4: seat selection interrupt ---------------------------------------


def _offered_seats() -> list[dict]:
    return [
        {"label": "A1", "row": "A", "number": 1, "available": True},
        {"label": "A2", "row": "A", "number": 2, "available": False},  # taken
        {"label": "B1", "row": "B", "number": 1, "available": True},
        {"label": "B2", "row": "B", "number": 2, "available": True},
    ]


def _phase4_state(seat_quantity: int = 2) -> BookingState:
    state = initial_booking_state(
        movie="Dune 3",
        date="Saturday",
        preferred_multiplexes=["Andino"],
        seat_quantity=seat_quantity,
    )
    state["offered_seats"] = _offered_seats()
    return state


async def _run_phase_4(state: BookingState, resume):
    """Drive the Phase 4 interrupt node through a MemorySaver graph to a single resume.

    A bare interrupt node can't be called directly, so wrap it in a one-node ``BookingState``
    graph (the same way the real booking graph hosts it) and resume once with ``resume``.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command

    builder = StateGraph(BookingState)
    builder.add_node("p4", build_phase_4_node())
    builder.add_edge(START, "p4")
    builder.add_edge("p4", END)
    graph = builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "p4-thread"}}
    first = await graph.ainvoke(state, config)
    if resume is None:
        return first
    return await graph.ainvoke(Command(resume=resume), config)


async def test_phase_4_presents_available_seats_grouped_by_row() -> None:
    first = await _run_phase_4(_phase4_state(), resume=None)

    payload = first["__interrupt__"][0].value
    assert payload["kind"] == "present_seats"
    assert payload["seat_quantity"] == 2
    assert payload["allow_reject"] is True
    # Only available seats, grouped by row; the taken A2 is omitted.
    assert payload["rows"] == {"A": ["A1"], "B": ["B1", "B2"]}


async def test_phase_4_valid_selection_writes_chosen_seats() -> None:
    final = await _run_phase_4(
        _phase4_state(), resume={"action": "select", "seats": ["B1", "B2"]}
    )
    assert "__interrupt__" not in final
    assert final["phase_outcome"] == "SEATS_CHOSEN"
    assert final["chosen_seats"] == ["B1", "B2"]


async def test_phase_4_reject_yields_no_seats() -> None:
    final = await _run_phase_4(_phase4_state(), resume={"action": "reject"})
    assert final["phase_outcome"] == "NO_SEATS"
    assert final.get("chosen_seats") is None


@pytest.mark.parametrize(
    "bad_seats",
    [
        ["B1"],            # too few (seat_quantity is 2)
        ["B1", "B2", "A1"],  # too many
        ["A2", "B1"],      # A2 is taken (not in the available set)
        ["Z9", "B1"],      # unknown label
        ["B1", "B1"],      # duplicate
    ],
)
async def test_phase_4_invalid_selection_reprompts(bad_seats: list[str]) -> None:
    reprompt = await _run_phase_4(
        _phase4_state(), resume={"action": "select", "seats": bad_seats}
    )
    assert "__interrupt__" in reprompt
    assert reprompt["__interrupt__"][0].value["error"]


@pytest.mark.parametrize(
    ("outcome", "expected_route"),
    [
        ("SEATS_CHOSEN", ROUTE_SELECT_SEATS),
        ("NO_SEATS", ROUTE_INFORM_NO_SEATS),
    ],
)
def test_route_after_phase_4(outcome: str, expected_route: str) -> None:
    assert route_after_phase_4({"phase_outcome": outcome}) == expected_route


# --- Phase 5: click the chosen seats -----------------------------------------


def _phase5_state(chosen: list[str] | None = None) -> dict:
    return {"chosen_seats": chosen if chosen is not None else ["K10", "K11"]}


async def test_phase_5_clicks_seats_and_reports_selected() -> None:
    fake = _FakeAgentLoop("Both seats are now selected.\nSTATUS: SEATS_SELECTED")
    node = build_phase_5_node(fake)

    update = await node(_phase5_state(), _config("user-1:thread-9"))

    assert update["phase_outcome"] == "SEATS_SELECTED"
    assert len(update["messages"]) == 1
    assert isinstance(update["messages"][0], AIMessage)
    # Runs on its own per-phase thread; the prompt lists the chosen seats by name.
    call = fake.calls[0]
    assert call["thread_id"] == "user-1:thread-9:p5"
    assert '"Silla K10"' in call["system_prompt"]
    assert '"Silla K11"' in call["system_prompt"]


async def test_phase_5_missing_status_downgrades_to_needs_retry() -> None:
    fake = _FakeAgentLoop("I couldn't find one of the seat buttons.")
    node = build_phase_5_node(fake)

    update = await node(_phase5_state(), _config())

    assert update["phase_outcome"] == "NEEDS_RETRY"


async def test_phase_5_recursion_limit_downgrades_to_needs_retry() -> None:
    # A toggling agent hits the recursion limit; the node must not crash the graph — it
    # downgrades to NEEDS_RETRY (like Phase 1/3) so the outer flow ends cleanly.
    from paddington.agent.agent_loop import AgentRecursionLimitError

    fake = _FakeAgentLoop("", raises=AgentRecursionLimitError("step limit"))
    node = build_phase_5_node(fake)

    update = await node(_phase5_state(), _config())

    assert update["phase_outcome"] == "NEEDS_RETRY"
    assert isinstance(update["messages"][0], AIMessage)


@pytest.mark.parametrize(
    ("outcome", "expected_route"),
    [
        ("SEATS_SELECTED", ROUTE_CHECKOUT),
        ("NEEDS_RETRY", ROUTE_INFORM_NEEDS_RETRY),
        (None, ROUTE_INFORM_NEEDS_RETRY),
    ],
)
def test_route_after_phase_5(outcome: str | None, expected_route: str) -> None:
    assert route_after_phase_5({"phase_outcome": outcome}) == expected_route
