"""Per-phase system-prompt composition for the booking flow.

The phase-based architecture (docs/phase-based-booking-architecture.md) gives each
phase its own goal and an explicit "ends when" boundary. We do NOT evaluate that
boundary with a programmatic page-state validator — the LLM decides when it has
reached the goal, driven entirely by the system prompt. Every phase prompt is
therefore the reusable browser workflow rules plus a phase-specific goal and an
explicit instruction to stop and report once the boundary is satisfied.
"""

# Reusable, phase-agnostic browser workflow rules. Shared by every phase prompt
# (and by the default prompt in agent_loop.py).
BROWSER_BASE_PROMPT = """## Browser tools

You can navigate and interact with real websites.
Follow this workflow:

1. Call navigate_to to go to a URL
2. Call get_snapshot to see the page content and available interactive elements
3. Each element has a ref (like "el_9") — use these refs in click_element and input_text
4. After EVERY action (click, input), call get_snapshot again — the page has changed
5. NEVER guess a ref — only use refs from the most recent snapshot
6. If a ref fails, call get_snapshot to get fresh refs and retry

Use take_screenshot to visually verify a page explicitly when markdown isn't enough,
not something that happens automatically on every step. — layout, CAPTCHA,
rendered charts/canvas, or other visual state get_snapshot can't convey. Keep
using get_snapshot for reading text and for the refs needed to click/type;
screenshots are costly, so use them sparingly."""


def build_phase_prompt(goal: str, ends_when: str) -> str:
    """Compose a phase system prompt from a goal and its "ends when" boundary.

    The returned prompt is the shared browser rules, then the phase goal, then a
    single explicit stop-and-report instruction. That instruction *is* how the
    "ends when" predicate is enforced — by instruction to the LLM, not by code.
    """
    return (
        f"{BROWSER_BASE_PROMPT}\n\n"
        f"## Your goal\n\n{goal.strip()}\n\n"
        f"Stop and report your findings as soon as {ends_when.strip()}"
    )


# --- Worked example: Phase 1 -------------------------------------------------
# Demonstrates the mechanism. The full registry for phases 2-6 lands with the
# outer phase graph (out of scope here).

_PHASE1_GOAL = """\
Navigate to cinecolombia.com and find showtimes for the movie "{movie}" at the
"{theater}" theater on {date}. Dismiss any career or cookie popup, find the
requested movie (scrolling as needed), open it, click "Ver horarios", select the
requested date, and locate the requested theater to reveal its showtimes. If the
movie is not listed, or the theater is not available, say so plainly instead of
guessing."""

_PHASE1_ENDS_WHEN = (
    "the showtimes for the requested date and theater are visible on screen "
    "(or you have determined the movie or theater is unavailable)."
)


def phase1_find_showtimes_prompt(movie: str, theater: str, date: str) -> str:
    """System prompt for Phase 1 — find the movie and reveal its showtimes."""
    return build_phase_prompt(
        goal=_PHASE1_GOAL.format(movie=movie, theater=theater, date=date),
        ends_when=_PHASE1_ENDS_WHEN,
    )
