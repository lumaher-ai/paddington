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
from paddington.browser.browser_session import BrowserSession
from paddington.schemas.browser import InteractiveElement, PageSnapshot

# A Phase 3 answer that reaches the seat map, so a selection flows all the way through.
_SEAT_MAP_ANSWER = "The seat map is on screen.\nSTATUS: SEAT_MAP_VISIBLE"
# A Phase 6 answer that prepares the order, so the flow continues into Phase 7.
_ORDER_PREPARED_ANSWER = "Added the tickets and continued to payment.\nSTATUS: ORDER_PREPARED"
# A Phase 7 answer that fills the form and picks the card method, so the flow runs to END.
_PAYMENT_READY_ANSWER = "Filled the form and selected the card method.\nSTATUS: PAYMENT_READY"

_SEATS_URL = "https://multiplex.cinecolombia.com/order/showtimes/6493-7284/seats"
_TICKETS_URL = "https://multiplex.cinecolombia.com/order/showtimes/6493-7284/tickets"
_PLACETOPAY_URL = "https://checkout.placetopay.com/spa/session/262085824/26a5a9c0"


@dataclass
class _FakeAgentLoop:
    """Stand-in for AgentLoop: returns a per-phase canned answer so the graph runs through.

    Phases 1, 3, 6, and 7 run the inner agent; they are told apart by the isolated thread-id
    suffix (``:p1`` / ``:p3`` / ``:p6`` / ``:p7``). Phase 5 is a deterministic, code-owned node
    (no inner agent) — its behavior is driven by the fake session, not this loop.
    """

    answer: str
    phase3_answer: str = _SEAT_MAP_ANSWER
    phase6_answer: str = _ORDER_PREPARED_ANSWER
    phase7_answer: str = _PAYMENT_READY_ANSWER

    async def run(self, **kwargs) -> AgentResult:
        thread_id = kwargs.get("thread_id", "")
        if thread_id.endswith(":p3"):
            answer = self.phase3_answer
        elif thread_id.endswith(":p6"):
            answer = self.phase6_answer
        elif thread_id.endswith(":p7"):
            answer = self.phase7_answer
        else:
            answer = self.answer
        return AgentResult(
            answer=answer,
            iterations=1,
            tools_used=[],
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=0.0,
        )


# Default seat map for Phase 3's parse + Phase 5's deterministic selection: two available seats,
# a non-seat button, and the confirm button Phase 5 clicks to advance.
_DEFAULT_SEATS = [
    InteractiveElement(ref="el_1", role="button", name="Iniciar sesión"),
    InteractiveElement(ref="el_2", role="button", name="Silla A1"),
    InteractiveElement(ref="el_3", role="button", name="Silla A2"),
    InteractiveElement(ref="el_4", role="button", name="Seleccionar boletas"),
]


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


class _FakePage:
    """Playwright Page stand-in for the URL checks in Phase 5 (leaves /seats) and Phase 7
    (reaches the placetopay gateway). Each ``wait_for_url`` advances to the next queued URL."""

    def __init__(self, url: str = "", next_urls: list[str] | None = None) -> None:
        self.url = url
        self._next = list(next_urls or [])

    async def wait_for_url(self, predicate, timeout: int = 0) -> None:  # noqa: ANN001
        # Simulate a redirect settling: advance to the next queued URL (or stay put if none).
        if self._next:
            self.url = self._next.pop(0)


class _ClickRes:
    def __init__(self, success: bool = True) -> None:
        self.success = success


class _FakeSession:
    """BrowserSession stand-in. Phase 1 reads ``last_snapshot`` (showtime links); Phase 3
    calls ``get_snapshot()`` for the seat map; Phase 5 clicks seats + confirm and reads
    ``page.url`` to verify it left the map; Phase 7 reads ``recorder`` + ``page.url``."""

    def __init__(
        self,
        last_snapshot: PageSnapshot | None = None,
        seat_snapshot: PageSnapshot | None = None,
        page: _FakePage | None = None,
    ) -> None:
        self.last_snapshot = last_snapshot
        self._seat_snapshot = seat_snapshot
        self.page = page or _FakePage()
        self.clicked: list[str] = []
        # Phase 7 disables debug recording for PII; None means "already off" (no-op).
        self.recorder = None

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

    async def click(self, ref: str, *args, **kwargs) -> _ClickRes:
        self.clicked.append(ref)
        return _ClickRes(True)


def _session(
    links: list[InteractiveElement] | None = None,
    seats: list[InteractiveElement] | None = _DEFAULT_SEATS,
    page: _FakePage | None = None,
) -> BrowserSession:
    # Default page walks the happy path: on the seat map, then Phase 5's confirm redirects to
    # /tickets, then Phase 7's card-method click redirects to the placetopay gateway.
    return cast(
        BrowserSession,
        _FakeSession(
            _snapshot(links),
            _snapshot(seats),
            page or _FakePage(_SEATS_URL, next_urls=[_TICKETS_URL, _PLACETOPAY_URL]),
        ),
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


def _graph(answer: str, extractor=_fake_extractor, session=None, loop=None):
    """Build the booking graph over a fake agent loop and an in-memory checkpointer.

    ``_FakeAgentLoop`` is a structural stand-in (it implements ``run``), so cast it to
    ``AgentLoop`` to satisfy the static signature without subclassing the real loop. The
    Option-B extractor is injected so Phase 1 never makes a live LLM call in tests. A fake
    ``session`` (no snapshot by default → no captured URLs) supplies Phase 1's link source.
    """
    return build_booking_graph(
        cast(AgentLoop, loop or _FakeAgentLoop(answer)),
        session=session if session is not None else _session(),
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
    return "\n".join(str(m.content) for m in final["messages"] if isinstance(m, AIMessage))


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


async def test_selecting_a_showtime_runs_phase_3_then_interrupts_for_seats() -> None:
    graph = _graph(_FOUND_ANSWER)

    await graph.ainvoke(_state(), config=_CONFIG)
    final = await graph.ainvoke(
        Command(resume={"action": "select", "showtime_id": "st_2"}), config=_CONFIG
    )

    # chosen_showtime is the human label (Phase 3's fallback), set by Phase 2.
    assert final["chosen_showtime"] == "9:50 P.M. · SALA 1"
    # Phase 3 reached the seat map and parsed the seats; Phase 4 now interrupts for the pick.
    assert [s["label"] for s in final["offered_seats"]] == ["A1", "A2"]
    payload = final["__interrupt__"][0].value
    assert payload["kind"] == "present_seats"
    assert payload["rows"] == {"A": ["A1", "A2"]}
    assert payload["allow_reject"] is True


async def test_full_flow_selecting_seats_reaches_end() -> None:
    graph = _graph(_FOUND_ANSWER)

    await graph.ainvoke(_state(), config=_CONFIG)
    await graph.ainvoke(Command(resume={"action": "select", "showtime_id": "st_2"}), config=_CONFIG)
    # seat_quantity defaults to 2; pick both available seats.
    final = await graph.ainvoke(
        Command(resume={"action": "select", "seats": ["A1", "A2"]}), config=_CONFIG
    )

    assert "__interrupt__" not in final
    # The pick flows through Phase 5 (clicks the seats), Phase 6 (prepares the order), and
    # Phase 7 (fills the form + picks the card method), which reaches the payment gateway and
    # reports PAYMENT_READY as the terminal outcome.
    assert final["phase_outcome"] == "PAYMENT_READY"
    assert final["chosen_seats"] == ["A1", "A2"]
    # Phase 7 captured the real payment link (the external gateway), not the checkout URL.
    assert final["payment_link"] == _PLACETOPAY_URL
    assert "checkout.placetopay.com" in _contents(final)


async def test_phase_5_failure_routes_to_inform_needs_retry() -> None:
    # Deterministic Phase 5 fails when the page never leaves the seat map after confirm
    # (url_after=None) → NEEDS_RETRY → inform exit with the retry message.
    stuck_session = _session(page=_FakePage(_SEATS_URL, next_urls=[]))
    graph = _graph(_FOUND_ANSWER, session=stuck_session)

    await graph.ainvoke(_state(), config=_CONFIG)
    await graph.ainvoke(Command(resume={"action": "select", "showtime_id": "st_2"}), config=_CONFIG)
    final = await graph.ainvoke(
        Command(resume={"action": "select", "seats": ["A1", "A2"]}), config=_CONFIG
    )

    assert "__interrupt__" not in final
    assert final["phase_outcome"] == "NEEDS_RETRY"
    assert "Please try again" in _contents(final)


async def test_phase_6_failure_routes_to_inform_needs_retry() -> None:
    loop = _FakeAgentLoop(_FOUND_ANSWER, phase6_answer="Couldn't reach the payment step.")
    graph = _graph(_FOUND_ANSWER, loop=loop)

    await graph.ainvoke(_state(), config=_CONFIG)
    await graph.ainvoke(Command(resume={"action": "select", "showtime_id": "st_2"}), config=_CONFIG)
    final = await graph.ainvoke(
        Command(resume={"action": "select", "seats": ["A1", "A2"]}), config=_CONFIG
    )

    # Phase 5 succeeds, but Phase 6 emits no valid STATUS → NEEDS_RETRY → inform exit.
    assert "__interrupt__" not in final
    assert final["phase_outcome"] == "NEEDS_RETRY"
    assert "Please try again" in _contents(final)


async def test_rejecting_seats_routes_to_inform_no_seats() -> None:
    graph = _graph(_FOUND_ANSWER)

    await graph.ainvoke(_state(), config=_CONFIG)
    await graph.ainvoke(Command(resume={"action": "select", "showtime_id": "st_2"}), config=_CONFIG)
    final = await graph.ainvoke(Command(resume={"action": "reject"}), config=_CONFIG)

    assert "__interrupt__" not in final
    assert final["phase_outcome"] == "NO_SEATS"
    assert final.get("chosen_seats") is None
    assert "won't reserve any seats" in _contents(final)


async def test_chosen_showtime_url_is_captured_and_threaded_to_phase_3() -> None:
    # The showtimes page Phase 1 last read carries the real seat-selection href.
    seat_url = "https://multiplex.cinecolombia.com/order/showtimes/6493-6874/seats"
    links = [
        InteractiveElement(ref="el_1", role="link", name="9:50 P.M. SALA 1", href=seat_url),
    ]
    graph = _graph(_FOUND_ANSWER, session=_session(links))

    interrupted = await graph.ainvoke(_state(), config=_CONFIG)
    # Phase 1 stamped the code-owned URL onto the option (st_2 == 9:50 P.M. · SALA 1).
    st2 = next(o for o in interrupted["offered_showtimes"] if o["id"] == "st_2")
    assert st2["url"] == seat_url

    final = await graph.ainvoke(
        Command(resume={"action": "select", "showtime_id": "st_2"}), config=_CONFIG
    )
    # The chosen option's URL was threaded into state for Phase 3's fast path; the flow
    # then reached the seat map and paused at Phase 4's seat interrupt.
    assert final["chosen_showtime_url"] == seat_url
    assert final["phase_outcome"] == "SEAT_MAP_VISIBLE"
    assert final["__interrupt__"][0].value["kind"] == "present_seats"


async def test_phase_3_failure_routes_to_inform_needs_retry() -> None:
    loop = _FakeAgentLoop(_FOUND_ANSWER, phase3_answer="Could not reach the seats.")
    graph = _graph(_FOUND_ANSWER, loop=loop)

    await graph.ainvoke(_state(), config=_CONFIG)
    final = await graph.ainvoke(
        Command(resume={"action": "select", "showtime_id": "st_1"}), config=_CONFIG
    )

    # Phase 3 emitted no valid STATUS → NEEDS_RETRY → inform exit with the retry message.
    assert final["phase_outcome"] == "NEEDS_RETRY"
    assert "Please try again" in _contents(final)


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

    # A subsequent valid choice resolves the loop; the graph runs through Phase 3 and
    # pauses at Phase 4's seat interrupt.
    final = await graph.ainvoke(
        Command(resume={"action": "select", "showtime_id": "st_1"}), config=_CONFIG
    )
    assert final["chosen_showtime"] == "7:20 P.M. · SALA 4 · 2D · Subtitled"
    assert final["__interrupt__"][0].value["kind"] == "present_seats"


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
    assert state["chosen_seats"] is None
    assert state["payment_link"] is None
