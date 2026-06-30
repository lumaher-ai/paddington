"""Assembly of the outer booking graph.

This is the deterministic outer layer of the phase architecture
(docs/phase-based-booking-architecture.md): a ``StateGraph(BookingState)`` whose nodes are
phases and whose edges encode the happy path plus the unhappy-path early exits. The
phases' actual browser work is delegated to the inner ReAct agent via the phase nodes in
``booking_nodes``; this module only wires them together.

Milestone 1 builds Phase 1 and its exits only. The ``FOUND_SHOWTIMES`` branch stops at
``END`` as a placeholder for the Phase 2 interrupt (later slice).
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from paddington.agent.agent_loop import AgentLoop
from paddington.agent.booking_nodes import (
    ROUTE_INFORM_MOVIE_UNAVAILABLE,
    ROUTE_INFORM_NEEDS_RETRY,
    ROUTE_INFORM_THEATER_UNAVAILABLE,
    ROUTE_PRESENT_SHOWTIMES,
    build_phase_1_node,
    build_phase_2_node,
    inform_movie_unavailable,
    inform_needs_retry,
    inform_theater_unavailable,
    route_after_phase_1,
)
from paddington.agent.booking_state import BookingState


def build_booking_graph(
    agent_loop: AgentLoop,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Wire and compile the booking graph, injecting the request-scoped ``AgentLoop``.

    Mirrors the ``AgentLoop`` dependency pattern: the caller (a booking route) builds the
    ``AgentLoop`` once from per-request tools and hands it in, so the Phase 1 node can run
    the inner agent. ``checkpointer`` is required, not optional: Phase 2's ``interrupt()``
    has nowhere to persist state without it and would raise at interrupt time, so a missing
    checkpointer is always a bug — fail at construction rather than mid-booking. The caller
    passes ``app.state.checkpointer`` (the same instance the inner ``AgentLoop`` uses; the
    per-phase ``thread_id`` isolation — ``T:p1``, ``T:p3`` — keeps their data separate).
    """
    builder = StateGraph(BookingState)

    builder.add_node("phase_1", build_phase_1_node(agent_loop))
    # Phase 2 presents Phase 1's showtimes and interrupts for the user's choice.
    # Registered under ROUTE_PRESENT_SHOWTIMES so the conditional edge's path_map
    # routes the FOUND_SHOWTIMES branch straight to it.
    builder.add_node(ROUTE_PRESENT_SHOWTIMES, build_phase_2_node())
    # Inform nodes registered under their route-constant names so the conditional
    # edge's path_map maps each route string straight to its node.
    builder.add_node(ROUTE_INFORM_MOVIE_UNAVAILABLE, inform_movie_unavailable)
    builder.add_node(ROUTE_INFORM_THEATER_UNAVAILABLE, inform_theater_unavailable)
    builder.add_node(ROUTE_INFORM_NEEDS_RETRY, inform_needs_retry)

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
    # After Phase 2 resumes with the user's choice, the next slice is Phase 3
    # (get-to-seats). Stops at END as a placeholder until that node exists.
    builder.add_edge(ROUTE_PRESENT_SHOWTIMES, END)
    builder.add_edge(ROUTE_INFORM_MOVIE_UNAVAILABLE, END)
    builder.add_edge(ROUTE_INFORM_THEATER_UNAVAILABLE, END)
    builder.add_edge(ROUTE_INFORM_NEEDS_RETRY, END)

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
        chosen_showtime=None,
        seat_section=None,
        chosen_seats=None,
        seat_quantity=seat_quantity,
        payment_link=None,
    )
