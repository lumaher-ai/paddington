from paddington.agent.seat_extraction import (
    ExtractedSeat,
    offered_seats,
    parse_seat_map,
    parse_seat_name,
)
from paddington.schemas.browser import InteractiveElement


def test_parse_available_regular_seat() -> None:
    seat = parse_seat_name("Silla F5")
    assert seat == ExtractedSeat(label="F5", row="F", number=5, available=True)


def test_parse_taken_seat() -> None:
    seat = parse_seat_name("Silla no disponible D18")
    assert seat == ExtractedSeat(label="D18", row="D", number=18, available=False)


def test_parse_companion_seat_is_bookable() -> None:
    # Companion seats ("Silla acompañante") are selectable, so they parse as available.
    seat = parse_seat_name("Silla acompañante A14")
    assert seat == ExtractedSeat(label="A14", row="A", number=14, available=True)


def test_parse_taken_companion_seat() -> None:
    seat = parse_seat_name("Silla acompañante no disponible A11")
    assert seat == ExtractedSeat(label="A11", row="A", number=11, available=False)


def test_wheelchair_space_is_not_a_seat() -> None:
    assert parse_seat_name("Espacio para sillas de ruedas A17") is None


def test_non_seat_names_return_none() -> None:
    assert parse_seat_name("Iniciar sesión") is None
    assert parse_seat_name("Comprar sin registrarse") is None
    assert parse_seat_name("") is None


def test_parse_seat_map_filters_non_seats_and_preserves_order() -> None:
    elements = [
        InteractiveElement(ref="el_1", role="link", name="Cine Colombia", href="x"),
        InteractiveElement(ref="el_2", role="button", name="Espacio para sillas de ruedas A17"),
        InteractiveElement(ref="el_3", role="button", name="Silla A13"),
        InteractiveElement(ref="el_4", role="button", name="Silla no disponible A12"),
        InteractiveElement(ref="el_5", role="button", name="Iniciar sesión"),
    ]
    seats = parse_seat_map(elements)

    assert [s.label for s in seats] == ["A13", "A12"]
    assert [s.available for s in seats] == [True, False]


def test_offered_seats_dict_shape() -> None:
    seats = [
        ExtractedSeat(label="A13", row="A", number=13, available=True),
        ExtractedSeat(label="A12", row="A", number=12, available=False),
    ]
    assert offered_seats(seats) == [
        {"label": "A13", "row": "A", "number": 13, "available": True},
        {"label": "A12", "row": "A", "number": 12, "available": False},
    ]
