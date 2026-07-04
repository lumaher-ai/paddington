# Phase 6 hang — fast-fail disabled clicks + expose disabled state

## Incident (real run 2026-07-04 17:00–17:03)

Phase 6 ("prepare the order for payment") hung for ~3 minutes and had to be killed. The logs
prove the cause is **NOT** interactive-element detection:

```
17:01:03 browser_click  name='Increase quantity' ref=el_5  navigated=False   ← click #1 OK (qty 0→1)
17:01:04 browser_click  name='Increase quantity' ref=el_5  navigated=False   ← click #2 OK (qty 1→2 = max)
17:01:05 snapshot  interactive_count 17→16, total_chars 2411→2265            ← page re-renders to maxed state
17:01:36 browser_click_failed 'click timed out after 30000ms' name='Increase quantity' ref=el_5
... same 30s timeout ×6 on el_5 until Ctrl-C
```

Findings:
1. **Detection works.** Both `Increase quantity` (el_5) and `Siguiente` (el_4/el_7, seen during
   Phase 5's overrun) are found by accessible name. No nested-clickable miss occurred.
2. **Quantity caps at 2** (the number of selected seats, N5/N6). After 2 successful "+" clicks
   the button goes **disabled**.
3. **The hang is a disabled-element actionability timeout.** Playwright `locator.click()` waits
   for the element to be *enabled* before clicking; a disabled button never enables, so it blocks
   the full `timeout_ms=30000` and returns "click timed out after 30000ms". The agent, stuck in
   an over-click loop, retried the same disabled `el_5` six times.

A universal nested-clickable detection rewrite was considered and **rejected** for this incident:
the logs show detection already finds the buttons. It remains a possible future hardening for
other providers, but it does not fix this hang.

## Root causes

- **R1 — `click` had no fast-fail for non-actionable elements.** A wrong/disabled click hangs
  30s and repeats.
- **R2 — the snapshot hid disabled state.** `_COLLECT_INTERACTIVE_JS` emitted no
  `disabled`/`aria-disabled`, so neither the agent nor Phase 6 knew the "+" was capped.
- **R3 — behavioral over-click.** The agent kept clicking "+" past the target because it could
  not reliably read the displayed count; a prompt-only mitigation is not robust — a hard page
  signal (disabled) is.

## Fix (implemented)

### 1. Fast-fail non-actionable clicks — `src/paddington/browser/browser_session.py`
`click()` now probes `locator.is_enabled(timeout=2_000)` (which honours native `disabled` and
`aria-disabled`) before clicking. If the element is disabled it returns immediately with
`success=False` and an informative error (`"element 'el_5' ('Increase quantity') is disabled and
cannot be clicked … Do NOT retry it; choose another action."`) and a `browser_click_disabled`
warning log — instead of blocking 30s. The same guard is mirrored in `input_text()`.

### 2. Expose enabled/disabled state — collector JS + schema
- `_COLLECT_INTERACTIVE_JS` computes
  `const disabled = el.disabled === true || el.getAttribute('aria-disabled') === 'true';`
  and returns it per element (strict superset — seats/links and their `name`/`href` unchanged).
- `InteractiveElement` (`src/paddington/schemas/browser.py`) gains `disabled: bool = False`
  (defaulted, so existing consumers/tests/fixtures are unaffected). The LLM now sees which
  controls are inert.

### 3. Phase 6 prompt tie-in — `src/paddington/agent/phase_prompts.py` `_PHASE6_GOAL`
Added a hard, page-state stop signal: when the "Increase quantity" button is shown disabled (or a
click reports it disabled), the quantity is at its maximum — stop clicking it and go to
"Siguiente". This composes with the disabled flag and the instant fast-fail error.

## Safety / consumers

`seat_extraction.parse_seat_map` and `showtime_extraction._match_url` read only `name`/`href`, so
the new defaulted `disabled` field does not disturb them. Existing `InteractiveElement(...)`
fixtures in the test suite keep working.

## Verification

- **Unit:** `tests/test_browser_session.py` — a disabled fake element makes `click()` return
  `success=False` with a "disabled" error in <1s and without attempting the click; an enabled
  element is clicked normally. (No real browser needed.)
- **Regression:** `uv run pytest -q` — 123 passed. (`tests/test_agent.py::test_agent_run_validates_message`
  fails independently of this change — a pre-existing `State.browser_session_manager` wiring issue.)
- **Lint/type:** `ruff` and `mypy` clean on the changed files.
- **End-to-end (manual):** re-run `scripts/run_booking.py` for a 2-seat booking; Phase 6 should
  click "+" twice, see the "+" as `disabled`, fail any stray third "+" instantly, then click
  "Siguiente" → food & drinks → "Continuar con el pago" → `ORDER_PREPARED`, with no 30s timeouts.
