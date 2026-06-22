from paddington.agent.agent_loop import (
    _DEFAULT_SYSTEM_PROMPT,
    AgentInvocationContext,
    _resolve_system_prompt,
)
from paddington.agent.phase_prompts import (
    BROWSER_BASE_PROMPT,
    build_phase_prompt,
    phase1_find_showtimes_prompt,
)


def test_build_phase_prompt_composes_base_goal_and_ends_when() -> None:
    prompt = build_phase_prompt(goal="Do the thing", ends_when="the thing is done.")
    assert BROWSER_BASE_PROMPT in prompt
    assert "Do the thing" in prompt
    # The "ends when" boundary is enforced by an explicit stop-and-report instruction.
    assert "Stop and report your findings as soon as the thing is done." in prompt


def test_phase1_prompt_interpolates_booking_params() -> None:
    prompt = phase1_find_showtimes_prompt(
        movie="Dune 3", theater="Titan Plaza", date="Saturday"
    )
    assert BROWSER_BASE_PROMPT in prompt
    assert "Dune 3" in prompt
    assert "Titan Plaza" in prompt
    assert "Saturday" in prompt
    assert "Stop and report your findings as soon as" in prompt


def test_resolve_system_prompt_returns_per_phase_when_set() -> None:
    phase_prompt = phase1_find_showtimes_prompt(
        movie="Dune 3", theater="Titan Plaza", date="Saturday"
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
