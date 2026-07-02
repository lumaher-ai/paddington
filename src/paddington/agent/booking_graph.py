"""Assembly of the outer booking graph.

This is the deterministic outer layer of the phase architecture
(docs/phase-based-booking-architecture.md): a ``StateGraph(BookingState)`` whose nodes are
phases and whose edges encode the happy path plus the unhappy-path early exits. The
phases' actual browser work is delegated to the inner ReAct agent via the phase nodes in
``booking_nodes``; this module only wires them together.

Phases 1-5 are wired: Phase 1 finds showtimes, Phase 2 interrupts for the user's choice,
Phase 3 navigates to the seat map and parses the seats, Phase 4 interrupts for the user's
seat picks, Phase 5 clicks those seats on the seat map. Phase 5's ``SEATS_SELECTED`` branch
stops at ``END`` as a placeholder for Phase 6 (checkout/payment, a later slice).
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from paddington.agent.agent_loop import AgentLoop
from paddington.agent.booking_nodes import (
    ROUTE_CHECKOUT,
    ROUTE_GET_TO_SEATS,
    ROUTE_INFORM_MOVIE_UNAVAILABLE,
    ROUTE_INFORM_NEEDS_RETRY,
    ROUTE_INFORM_NO_SEATS,
    ROUTE_INFORM_NO_SHOWTIME,
    ROUTE_INFORM_THEATER_UNAVAILABLE,
    ROUTE_PRESENT_SEATS,
    ROUTE_PRESENT_SHOWTIMES,
    ROUTE_SELECT_SEATS,
    build_phase_1_node,
    build_phase_2_node,
    build_phase_3_node,
    build_phase_4_node,
    build_phase_5_node,
    inform_movie_unavailable,
    inform_needs_retry,
    inform_no_seats,
    inform_no_showtime,
    inform_theater_unavailable,
    route_after_phase_1,
    route_after_phase_2,
    route_after_phase_3,
    route_after_phase_4,
    route_after_phase_5,
)
from paddington.agent.booking_state import BookingState
from paddington.agent.showtime_extraction import ShowtimeExtractor
from paddington.browser.browser_session import BrowserSession


def build_booking_graph(
    agent_loop: AgentLoop,
    session: BrowserSession,
    checkpointer: BaseCheckpointSaver,
    extract_showtimes: ShowtimeExtractor | None = None,
) -> CompiledStateGraph:
    """Wire and compile the booking graph, injecting the request-scoped ``AgentLoop``.

    Mirrors the ``AgentLoop`` dependency pattern: the caller (a booking route) builds the
    ``AgentLoop`` once from per-request tools and hands it in, so the Phase 1 node can run
    the inner agent. ``checkpointer`` is required, not optional: Phase 2's ``interrupt()``
    has nowhere to persist state without it and would raise at interrupt time, so a missing
    checkpointer is always a bug — fail at construction rather than mid-booking. The caller
    passes ``app.state.checkpointer`` (the same instance the inner ``AgentLoop`` uses; the
    per-phase ``thread_id`` isolation — ``T:p1``, ``T:p3`` — keeps their data separate).

    ``session`` is the same ``BrowserSession`` the ``AgentLoop``'s tools close over. Phase 1
    reads its ``last_snapshot`` to stamp each showtime option with a code-owned seat URL
    (never LLM-transcribed); the caller already holds this session to build the tools.

    ``extract_showtimes`` is the Option-B structuring step injected into Phase 1; left
    ``None`` in production (the node builds the real LLM extractor), it lets tests supply
    a deterministic double.
    """
    builder = StateGraph(BookingState)

    builder.add_node("phase_1", build_phase_1_node(agent_loop, session, extract_showtimes))
    # Phase 2 presents Phase 1's showtimes and interrupts for the user's choice.
    # Registered under ROUTE_PRESENT_SHOWTIMES so the conditional edge's path_map
    # routes the FOUND_SHOWTIMES branch straight to it.
    builder.add_node(ROUTE_PRESENT_SHOWTIMES, build_phase_2_node())
    # Phase 3 drives the chosen showtime to the seat map and parses the seats. Registered
    # under ROUTE_GET_TO_SEATS so Phase 2's path_map routes SHOWTIME_CHOSEN straight to it.
    builder.add_node(ROUTE_GET_TO_SEATS, build_phase_3_node(agent_loop, session))
    # Phase 4 presents Phase 3's seats and interrupts for the user's choice. Registered
    # under ROUTE_PRESENT_SEATS so Phase 3's path_map routes SEAT_MAP_VISIBLE straight to it.
    builder.add_node(ROUTE_PRESENT_SEATS, build_phase_4_node())
    # Phase 5 clicks the chosen seats on the seat map. Registered under ROUTE_SELECT_SEATS so
    # Phase 4's path_map routes SEATS_CHOSEN straight to it.
    builder.add_node(ROUTE_SELECT_SEATS, build_phase_5_node(agent_loop))
    # Inform nodes registered under their route-constant names so the conditional
    # edge's path_map maps each route string straight to its node.
    builder.add_node(ROUTE_INFORM_MOVIE_UNAVAILABLE, inform_movie_unavailable)
    builder.add_node(ROUTE_INFORM_THEATER_UNAVAILABLE, inform_theater_unavailable)
    builder.add_node(ROUTE_INFORM_NEEDS_RETRY, inform_needs_retry)
    # Phase 2's rejection exit (mermaid edge E3): the user wanted none of the showtimes.
    builder.add_node(ROUTE_INFORM_NO_SHOWTIME, inform_no_showtime)
    # Phase 4's rejection exit: the user wanted none of the offered seats.
    builder.add_node(ROUTE_INFORM_NO_SEATS, inform_no_seats)

    builder.add_edge(START, "phase_1")
    builder.add_conditional_edges(
        "phase_1",
        route_after_phase_1,
        {
            ROUTE_PRESENT_SHOWTIMES: ROUTE_PRESENT_SHOWTIMES,
            ROUTE_INFORM_MOVIE_UNAVAILABLE: ROUTE_INFORM_MOVIE_UNAVAILABLE,
            ROUTE_INFORM_THEATER_UNAVAILABLE: ROUTE_INFORM_THEATER_UNAVAILABLE,
            ROUTE_INFORM_NEEDS_RETRY: ROUTE_INFORM_NEEDS_RETRY,
        },
    )
    # After Phase 2 resumes with the user's choice, route on the outcome: a chosen
    # showtime flows to Phase 3 (get-to-seats), a rejection to the inform exit.
    builder.add_conditional_edges(
        ROUTE_PRESENT_SHOWTIMES,
        route_after_phase_2,
        {
            ROUTE_GET_TO_SEATS: ROUTE_GET_TO_SEATS,
            ROUTE_INFORM_NO_SHOWTIME: ROUTE_INFORM_NO_SHOWTIME,
        },
    )
    # After Phase 3: a visible seat map (with parsed seats) flows to Phase 4 (seat
    # selection); a failure to reach it routes to the needs-retry inform exit.
    builder.add_conditional_edges(
        ROUTE_GET_TO_SEATS,
        route_after_phase_3,
        {
            ROUTE_PRESENT_SEATS: ROUTE_PRESENT_SEATS,
            ROUTE_INFORM_NEEDS_RETRY: ROUTE_INFORM_NEEDS_RETRY,
        },
    )
    # After Phase 4 resumes with the user's seat choice: chosen seats flow to Phase 5
    # (click the seats); a rejection routes to the no-seats inform exit.
    builder.add_conditional_edges(
        ROUTE_PRESENT_SEATS,
        route_after_phase_4,
        {
            ROUTE_SELECT_SEATS: ROUTE_SELECT_SEATS,
            ROUTE_INFORM_NO_SEATS: ROUTE_INFORM_NO_SEATS,
        },
    )
    # After Phase 5: the seats were clicked, so flow to Phase 6 (checkout/payment), which
    # doesn't exist yet — ROUTE_CHECKOUT maps to END as a placeholder. A failure to select
    # them (e.g. a seat taken since Phase 4) routes to the needs-retry inform exit.
    builder.add_conditional_edges(
        ROUTE_SELECT_SEATS,
        route_after_phase_5,
        {
            ROUTE_CHECKOUT: END,
            ROUTE_INFORM_NEEDS_RETRY: ROUTE_INFORM_NEEDS_RETRY,
        },
    )
    builder.add_edge(ROUTE_INFORM_MOVIE_UNAVAILABLE, END)
    builder.add_edge(ROUTE_INFORM_THEATER_UNAVAILABLE, END)
    builder.add_edge(ROUTE_INFORM_NEEDS_RETRY, END)
    builder.add_edge(ROUTE_INFORM_NO_SHOWTIME, END)
    builder.add_edge(ROUTE_INFORM_NO_SEATS, END)

    return builder.compile(checkpointer=checkpointer)


def initial_booking_state(
    movie: str,
    date: str,
    preferred_multiplexes: list[str],
    seat_quantity: int = 2,
) -> BookingState:
    """Seed a fresh ``BookingState`` for a booking run.

    ``TypedDict`` carries no runtime defaults, so every field is set explicitly here: the
    user inputs, ``seat_quantity`` (default 2), and the not-yet-resolved fields to ``None``.
    """
    return BookingState(
        messages=[],
        phase="find_showtimes",
        phase_outcome=None,
        movie=movie,
        date=date,
        preferred_multiplexes=preferred_multiplexes,
        selected_theater=None,
        offered_showtimes=None,
        chosen_showtime=None,
        chosen_showtime_url=None,
        offered_seats=None,
        chosen_seats=None,
        seat_quantity=seat_quantity,
        payment_link=None,
    )
