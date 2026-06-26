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
    inform_movie_unavailable,
    inform_needs_retry,
    inform_theater_unavailable,
    route_after_phase_1,
)
from paddington.agent.booking_state import BookingState


def build_booking_graph(
    agent_loop: AgentLoop,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Wire and compile the booking graph, injecting the request-scoped ``AgentLoop``.

    Mirrors the ``AgentLoop`` dependency pattern: the caller (a booking route) builds the
    ``AgentLoop`` once from per-request tools and hands it in, so the Phase 1 node can run
    the inner agent. ``checkpointer`` is optional today — no interrupts yet — but is
    threaded into ``compile`` so the Phase 2 interrupt slice needs no signature change.
    """
    builder = StateGraph(BookingState)

    builder.add_node("phase_1", build_phase_1_node(agent_loop))
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
            ROUTE_PRESENT_SHOWTIMES: END,  # placeholder for the Phase 2 interrupt
            ROUTE_INFORM_MOVIE_UNAVAILABLE: ROUTE_INFORM_MOVIE_UNAVAILABLE,
            ROUTE_INFORM_THEATER_UNAVAILABLE: ROUTE_INFORM_THEATER_UNAVAILABLE,
            ROUTE_INFORM_NEEDS_RETRY: ROUTE_INFORM_NEEDS_RETRY,
        },
    )
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
