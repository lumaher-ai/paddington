"""Structured state for the phase-based booking graph.

The phase architecture (docs/phase-based-booking-architecture.md) has two layers: the
outer phase graph (deterministic, ours) and the inner ReAct loop (autonomous, the LLM's).
``BookingState`` is the outer graph's memory — the structured data threaded *between*
phases. A phase node reads what it needs from here, runs the inner agent, and writes its
result back (e.g. ``phase_outcome``) so the next conditional edge can route on it.

This is distinct from the inner agent's message history, which lives in its own checkpoint
under a per-phase thread id (see ``booking_nodes``). ``messages`` here is the outer-graph
transcript — short per-phase summaries, not the inner click-by-click trace.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class BookingState(TypedDict):
    """Data threaded between booking phases by the outer graph.

    Fields are filled in as phases complete: Phase 1 resolves ``selected_theater``,
    the Phase 2 interrupt sets ``chosen_showtime``, the Phase 4 interrupt sets
    ``seat_section``/``chosen_seats``, and Phase 5 sets ``payment_link``.

    ``TypedDict`` carries no runtime defaults — ``seat_quantity`` is seeded (to 2) when
    the initial state is constructed at graph invocation, not here.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    phase: str
    phase_outcome: str | None

    # Supplied by the user up front.
    movie: str
    date: str
    preferred_multiplexes: list[str]
    seat_quantity: int

    # Resolved as phases complete.
    selected_theater: str | None
    chosen_showtime: str | None
    seat_section: str | None
    chosen_seats: list[str] | None
    payment_link: str | None
