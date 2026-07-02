"""Parse the seat map into typed, structured seats — pure code, no LLM.

Unlike Phase 1's showtimes (free-text prose that needs an LLM to structure — see
``showtime_extraction``), the seat page is already structured DOM: every seat is a
``<button>`` whose accessible name is a regular grammar. So Phase 3 parses
``PageSnapshot.interactive_elements`` deterministically here and persists the result to
``BookingState.offered_seats``; Phase 4 then pure-reads that at its interrupt boundary,
the same producer/consumer split as Phase 1 -> Phase 2.

Seat-name grammar observed on cinecolombia.com (accessible name of each button):
  "Silla F5"                        -> available regular seat F5
  "Silla no disponible D18"         -> seat D18, already taken
  "Silla acompañante A14"           -> companion seat A14 (bookable)
  "Espacio para sillas de ruedas A17" -> wheelchair SPACE (not a "Silla" — excluded)

We do NOT capture the section (General vs Preferencial): the user knows the last rows are
Preferencial, so encoding it would add a snapshot-JS/schema change for no benefit. The
``label`` ("F5") is the durable, unique selection token — no synthetic id is assigned
(contrast the ``st_1`` round-trip ids in ``showtime_extraction``): the user picks by label
and Phase 4 validates against the available labels.
"""

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from paddington.schemas.browser import InteractiveElement

# Matches a seat button's accessible name. Optional "acompañante " (companion seats are
# bookable) and optional "no disponible " (taken) precede the row (letters) + number.
# Wheelchair spaces start with "Espacio" and simply don't match, so they're dropped.
_SEAT_RE = re.compile(r"^Silla (?:acompañante )?(no disponible )?([A-Za-z]+)(\d+)$")


class ExtractedSeat(BaseModel):
    """One seat as parsed from its button's accessible name."""

    label: str = Field(description="Row+number as shown, e.g. 'F5'. Unique, durable.")
    row: str = Field(description="Row letter(s), e.g. 'F'.")
    number: int = Field(description="Seat number within the row, e.g. 5.")
    available: bool = Field(description="True when the seat is selectable (not taken).")


def parse_seat_name(name: str) -> ExtractedSeat | None:
    """Parse one button's accessible name into an ``ExtractedSeat``.

    Returns ``None`` for names that aren't seats (wheelchair spaces, header/legend buttons,
    etc.), so callers can filter with a single comprehension.
    """
    match = _SEAT_RE.match(name.strip())
    if match is None:
        return None
    taken, row, number = match.groups()
    return ExtractedSeat(
        label=f"{row}{number}",
        row=row,
        number=int(number),
        available=taken is None,
    )


def parse_seat_map(elements: Sequence[InteractiveElement]) -> list[ExtractedSeat]:
    """Parse every seat button in a snapshot's interactive elements, in page order.

    Non-seat elements (links, the login button, wheelchair spaces) are skipped. Order is
    preserved so downstream grouping/presentation follows the visual layout.
    """
    seats = []
    for el in elements:
        seat = parse_seat_name(el.name)
        if seat is not None:
            seats.append(seat)
    return seats


def offered_seats(seats: Sequence[ExtractedSeat]) -> list[dict]:
    """Shape parsed seats into the dicts persisted in ``BookingState.offered_seats``.

    Kept as plain dicts (like ``offered_options``) so the Phase 4 interrupt node is a pure
    read on resume — no re-parsing, no drift. The ``label`` is the selection key; ``row``
    is retained so Phase 4 can group the available seats by row for presentation.
    """
    return [
        {
            "label": s.label,
            "row": s.row,
            "number": s.number,
            "available": s.available,
        }
        for s in seats
    ]
