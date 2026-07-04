import pytest

from paddington.agent.agent_loop import (
    _DEFAULT_SYSTEM_PROMPT,
    AgentInvocationContext,
    _resolve_system_prompt,
)
from paddington.agent.phase_prompts import (
    BROWSER_BASE_PROMPT,
    PHASE1_MOVIE_NOT_FOUND,
    PHASE1_NEEDS_RETRY,
    PHASE1_STATUSES,
    build_phase_prompt,
    parse_phase_status,
    phase1_find_showtimes_prompt,
    phase5_select_seats_prompt,
    phase6_prepare_order_prompt,
)


def test_build_phase_prompt_composes_base_goal_and_ends_when() -> None:
    prompt = build_phase_prompt(
        goal="Do the thing", ends_when="the thing is done.", statuses={"DONE"}
    )
    assert BROWSER_BASE_PROMPT in prompt
    assert "Do the thing" in prompt
    # The "ends when" boundary is enforced by an explicit stop-and-report instruction.
    assert "Stop and report your findings as soon as the thing is done." in prompt
    # The prompt asks the LLM for a STATUS line listing the allowed tokens.
    assert "STATUS: <TOKEN>" in prompt
    assert "DONE" in prompt


def test_phase1_prompt_interpolates_booking_params() -> None:
    prompt = phase1_find_showtimes_prompt(
        movie="Dune 3", theaters=["Titan Plaza"], date="Saturday"
    )
    assert BROWSER_BASE_PROMPT in prompt
    assert "Dune 3" in prompt
    assert "Titan Plaza" in prompt
    # The theater name is resolved to its /cinemas/<slug>/ URL in the fallback chain.
    assert "/cinemas/titan-plaza/" in prompt
    assert "Saturday" in prompt
    assert "Stop and report your findings as soon as" in prompt
    # Phase 1's allowed outcome tokens are surfaced in the STATUS instruction.
    assert "FOUND_SHOWTIMES" in prompt
    assert "MOVIE_NOT_FOUND" in prompt
    assert "THEATER_UNAVAILABLE" in prompt


def test_phase5_prompt_lists_chosen_seats_as_accessible_names() -> None:
    prompt = phase5_select_seats_prompt(["K10", "K11"])
    assert BROWSER_BASE_PROMPT in prompt
    # Each label is rendered as the seat button's accessible name the agent matches on.
    assert '"Silla K10"' in prompt
    assert '"Silla K11"' in prompt
    # The companion-seat variant is called out so an exact-string match isn't assumed.
    assert "acompañante" in prompt
    # The anti-toggle rule: never re-click a selected seat (that deselects it → loop).
    assert "EXACTLY ONCE" in prompt
    assert "DESELECTS" in prompt
    # After selecting, the agent must click the confirm button to advance.
    assert "Seleccionar boletas" in prompt
    assert "Stop and report your findings as soon as" in prompt
    # Only the happy outcome token is advertised in the STATUS instruction.
    assert "SEATS_SELECTED" in prompt


def test_phase6_prompt_describes_ticket_quantity_and_checkout_buttons() -> None:
    prompt = phase6_prepare_order_prompt(2)
    assert BROWSER_BASE_PROMPT in prompt
    # The seat_quantity is interpolated as the number of times to click the plus stepper.
    assert "2 tickets" in prompt
    # The plus stepper is matched by its accessible name.
    assert '"Increase quantity"' in prompt
    # Advance past the ticket page, then past food & drinks without adding anything.
    assert "Siguiente" in prompt
    assert "Continuar con el pago" in prompt
    assert "Stop and report your findings as soon as" in prompt
    # Only the happy outcome token is advertised in the STATUS instruction.
    assert "ORDER_PREPARED" in prompt


def test_resolve_system_prompt_returns_per_phase_when_set() -> None:
    phase_prompt = phase1_find_showtimes_prompt(
        movie="Dune 3", theaters=["Titan Plaza"], date="Saturday"
    )
    context = AgentInvocationContext(
        baseline_message_count=0, system_prompt=phase_prompt
    )
    assert _resolve_system_prompt(context) == phase_prompt


def test_resolve_system_prompt_falls_back_to_default() -> None:
    # No prompt supplied on the context.
    assert (
        _resolve_system_prompt(AgentInvocationContext(baseline_message_count=0))
        == _DEFAULT_SYSTEM_PROMPT
    )
    # Empty string is treated as unset.
    assert (
        _resolve_system_prompt(
            AgentInvocationContext(baseline_message_count=0, system_prompt="")
        )
        == _DEFAULT_SYSTEM_PROMPT
    )
    # No context at all (e.g. invoked without a runtime context).
    assert _resolve_system_prompt(None) == _DEFAULT_SYSTEM_PROMPT


@pytest.mark.parametrize(
    "answer",
    [
        "STATUS: found_showtimes",
        "Status: FOUND_SHOWTIMES ",
        "STATUS:FOUND_SHOWTIMES",
    ],
)
def test_parse_phase_status_normalizes_case_and_whitespace(answer: str) -> None:
    assert (
        parse_phase_status(answer, PHASE1_STATUSES, PHASE1_NEEDS_RETRY)
        == "FOUND_SHOWTIMES"
    )


def test_parse_phase_status_returns_valid_non_default_token() -> None:
    answer = "The movie isn't on the listing.\nSTATUS: MOVIE_NOT_FOUND"
    assert (
        parse_phase_status(answer, PHASE1_STATUSES, PHASE1_NEEDS_RETRY)
        == PHASE1_MOVIE_NOT_FOUND
    )


def test_parse_phase_status_falls_back_when_missing() -> None:
    assert (
        parse_phase_status("Here are the showtimes.", PHASE1_STATUSES, PHASE1_NEEDS_RETRY)
        == PHASE1_NEEDS_RETRY
    )


def test_parse_phase_status_falls_back_on_invalid_token() -> None:
    assert (
        parse_phase_status("STATUS: BANANA", PHASE1_STATUSES, PHASE1_NEEDS_RETRY)
        == PHASE1_NEEDS_RETRY
    )


def test_parse_phase_status_falls_back_on_empty_answer() -> None:
    assert parse_phase_status("", PHASE1_STATUSES, PHASE1_NEEDS_RETRY) == PHASE1_NEEDS_RETRY


def test_parse_phase_status_takes_last_occurrence() -> None:
    answer = "STATUS: MOVIE_NOT_FOUND\nActually I found them.\nSTATUS: FOUND_SHOWTIMES"
    assert (
        parse_phase_status(answer, PHASE1_STATUSES, PHASE1_NEEDS_RETRY)
        == "FOUND_SHOWTIMES"
    )


def test_parse_phase_status_tolerates_trailing_prose() -> None:
    answer = "STATUS: FOUND_SHOWTIMES\nLet me know if you want to pick a time."
    assert (
        parse_phase_status(answer, PHASE1_STATUSES, PHASE1_NEEDS_RETRY)
        == "FOUND_SHOWTIMES"
    )
