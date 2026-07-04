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


def build_phase_prompt(goal: str, ends_when: str, statuses: set[str]) -> str:
    """Compose a phase system prompt from a goal, its "ends when" boundary, and outcomes.

    The returned prompt is the shared browser rules, then the phase goal, then a
    single explicit stop-and-report instruction. That instruction *is* how the
    "ends when" predicate is enforced — by instruction to the LLM, not by code.

    ``statuses`` are the allowed outcome tokens for this phase. The prompt instructs
    the LLM to end its answer with a ``STATUS: <TOKEN>`` line; ``parse_phase_status``
    is the inverse that reads that token back so the outer graph can route on it.
    """
    options = ", ".join(sorted(statuses))
    return (
        f"{BROWSER_BASE_PROMPT}\n\n"
        f"## Your goal\n\n{goal.strip()}\n\n"
        f"Stop and report your findings as soon as {ends_when.strip()}\n\n"
        f"## Report status\n\n"
        f"End your final answer with a line in exactly this form:\n"
        f"    STATUS: <TOKEN>\n"
        f"where <TOKEN> is one of: {options}."
    )


def parse_phase_status(answer: str, allowed: set[str], fallback: str) -> str:
    """Extract and validate the STATUS token from the agent's final answer.

    The phase prompt (see ``build_phase_prompt``) instructs the LLM to end its
    answer with a line like ``STATUS: FOUND_SHOWTIMES``. This pure function pulls
    that token out, normalizes it (case/whitespace tolerant), validates it against
    ``allowed``, and returns it. A missing or invalid status is a known failure
    mode — it returns ``fallback`` rather than raising. When several STATUS lines
    are present, the last one wins (most likely the final declaration).
    """
    token = fallback
    for line in answer.splitlines():
        upper = line.upper()
        if "STATUS:" in upper:
            after = upper.split("STATUS:", 1)[1].strip().split()
            candidate = after[0] if after else ""
            token = candidate if candidate in allowed else fallback
    return token


# --- Worked example: Phase 1 -------------------------------------------------
# Demonstrates the mechanism. The full registry for phases 2-6 lands with the
# outer phase graph (out of scope here).

# Phase 1 outcomes. The LLM emits one of these as its STATUS line; the parser
# validates against this set. NEEDS_RETRY is the fallback when no valid status
# is present — a missing status is a known failure mode, not an exception.
PHASE1_FOUND_SHOWTIMES = "FOUND_SHOWTIMES"
PHASE1_MOVIE_NOT_FOUND = "MOVIE_NOT_FOUND"
PHASE1_THEATER_UNAVAILABLE = "THEATER_UNAVAILABLE"
PHASE1_NEEDS_RETRY = "NEEDS_RETRY"  # fallback / no valid status

PHASE1_STATUSES = {
    PHASE1_FOUND_SHOWTIMES,
    PHASE1_MOVIE_NOT_FOUND,
    PHASE1_THEATER_UNAVAILABLE,
}

# Phase 2 outcome tokens. Unlike PHASE1_* these are NOT LLM-declared STATUS lines —
# Phase 2 has no inner agent. They are produced by the node from the user's (validated)
# resume value at the interrupt boundary, and route_after_phase_2 maps them to edges:
# CHOSEN -> Phase 3 (get-to-seats), NO_SHOWTIME -> inform_no_showtime -> END.
PHASE2_SHOWTIME_CHOSEN = "SHOWTIME_CHOSEN"
PHASE2_NO_SHOWTIME = "NO_SHOWTIME"

# Phase 3 outcomes. The inner agent navigates to the seat map and declares one of these
# as its STATUS line; NEEDS_RETRY is the fallback when no valid status is present.
PHASE3_SEAT_MAP_VISIBLE = "SEAT_MAP_VISIBLE"
PHASE3_NEEDS_RETRY = "NEEDS_RETRY"  # fallback / no valid status

PHASE3_STATUSES = {
    PHASE3_SEAT_MAP_VISIBLE,
}

# Phase 4 outcome tokens. Like PHASE2_* (and unlike PHASE1/PHASE3), these are NOT
# LLM-declared STATUS lines — Phase 4 has no inner agent. The node produces them from the
# user's (validated) resume value at the seat-selection interrupt, and route_after_phase_4
# maps them to edges: CHOSEN -> checkout (Phase 5), NO_SEATS -> inform_no_seats -> END.
PHASE4_SEATS_CHOSEN = "SEATS_CHOSEN"
PHASE4_NO_SEATS = "NO_SEATS"

# Theater-first navigation. cinecolombia.com lists each multiplex's movies on its own
# /cinemas/<slug>/ page, which avoids the Cloudflare-blocked movie-detail page. We map
# friendly theater names to those URLs; preferred_multiplexes stays as names so the
# user-facing inform messages read naturally.
CINECOLOMBIA_BASE_URL = "https://www.cinecolombia.com"
MULTIPLEX_URLS = {
    "Andino": "https://www.cinecolombia.com/cinemas/andino/",
    "Avenida Chile": "https://www.cinecolombia.com/cinemas/avenida-chile/",
}


def _theater_url(name: str) -> str:
    """Resolve a theater display name to its cinecolombia listings URL.

    Known theaters use the explicit ``MULTIPLEX_URLS`` map; unknown names fall back to a
    slug (lowercased, spaces -> hyphens) so the agent still has a URL to try.
    """
    if name in MULTIPLEX_URLS:
        return MULTIPLEX_URLS[name]
    slug = name.strip().lower().replace(" ", "-")
    return f"{CINECOLOMBIA_BASE_URL}/cinemas/{slug}/"


_PHASE1_GOAL = """\
Find showtimes for "{movie}" on {date}.

Navigate these theater in priority order - directly to each URL. Move to the
next one only if the movie isn't listed at the first one:
{theaters_block}

For each theater:
1. navigate_to its URL directly
2. find "{movie}" in that theater's listings
3. Below the movie name is listed the showtimes (scroll down to find it)
4. Select the date {date} and reveal its showtimes

Do NOT use take_screenshot — the showtimes/seat data are text/DOM. Use get_snapshot
only. Screenshots waste tokens and add nothing here.

IMPORTANT: Do NOT click on movie titles or poster links. The showtimes are 
listed directly on the theater page below each movie. If you can see the 
movie name but not its showtimes, the data may be below the visible area — 
scroll down, do not navigate away from the theater page.

Stop at the first theater that has the movie. If the movie is not listed at any of
the theaters, say so plainly."""

_PHASE1_ENDS_WHEN = (
    "the showtimes for the requested date and theater are visible on screen "
    "(or movie or theater not available in the list → inform user → **END**"
)


def phase1_find_showtimes_prompt(movie: str, theaters: list[str], date: str) -> str:
    """System prompt for Phase 1 — find the movie and reveal its showtimes.

    ``theaters`` are friendly names in priority order; each is resolved to its
    cinecolombia listings URL and rendered as a numbered fallback chain in the goal.
    """
    theaters_block = "\n".join(
        f"{i}. {name} — {_theater_url(name)}" for i, name in enumerate(theaters, 1)
    )
    return build_phase_prompt(
        goal=_PHASE1_GOAL.format(movie=movie, theaters_block=theaters_block, date=date),
        ends_when=_PHASE1_ENDS_WHEN,
        statuses=PHASE1_STATUSES,
    )


# --- Phase 3: get to the seat map --------------------------------------------
# Thin navigation slice. The chosen showtime's seat-selection URL (captured by code in
# Phase 1) is the fast path; the human label is the fallback if that URL is missing. The
# one non-obvious obstacle is the guest-checkout interstitial: an unauthenticated hit to a
# /seats URL redirects to a sign-in page, and "Comprar sin registrarse" is the way through.

_PHASE3_GOAL_WITH_URL = """\
Get to the seat-selection map for the showtime the user chose: "{chosen_label}".

1. navigate_to this URL directly — it goes straight to seat selection:
   {seat_url}
2. If that page redirects to a sign-in / "Iniciar sesión" screen, get_snapshot and click
   the "Comprar sin registrarse" (buy without registering) button to continue as a guest.
   Do NOT try to log in, create an account, enter any credentials, or solve a CAPTCHA.
3. get_snapshot and confirm the interactive seat map has loaded.

If navigating to the URL fails or does not reach a seat map, fall back to the theater
route below."""

_PHASE3_GOAL_LABEL_ONLY = """\
Get to the seat-selection map for the showtime the user chose: "{chosen_label}".

1. Go to the theater's listings and find the screening labeled "{chosen_label}"
   (the theaters, in priority order):
{theaters_block}
2. Click that showtime to open seat selection.
3. If the page redirects to a sign-in / "Iniciar sesión" screen, get_snapshot and click
   the "Comprar sin registrarse" (buy without registering) button to continue as a guest.
   Do NOT try to log in, create an account, enter any credentials, or solve a CAPTCHA.
4. get_snapshot and confirm the interactive seat map has loaded."""

_PHASE3_ENDS_WHEN = (
    "the interactive seat map is visible (the page is titled 'Seleccione sus sillas' and "
    "shows the seat legend and clickable seat buttons)"
)


def phase3_get_to_seats_prompt(
    chosen_label: str,
    seat_url: str | None,
    theaters: list[str],
) -> str:
    """System prompt for Phase 3 — reach the seat map for the chosen showtime.

    Prefers the code-captured ``seat_url`` (direct navigation); when it is ``None`` the
    prompt falls back to re-finding the screening by ``chosen_label`` on the theater page.
    Either way the agent must handle the guest-checkout ("Comprar sin registrarse")
    interstitial and stop once the seat map is visible.
    """
    if seat_url:
        goal = _PHASE3_GOAL_WITH_URL.format(chosen_label=chosen_label, seat_url=seat_url)
    else:
        theaters_block = "\n".join(
            f"   {i}. {name} — {_theater_url(name)}" for i, name in enumerate(theaters, 1)
        )
        goal = _PHASE3_GOAL_LABEL_ONLY.format(
            chosen_label=chosen_label, theaters_block=theaters_block
        )
    return build_phase_prompt(
        goal=goal,
        ends_when=_PHASE3_ENDS_WHEN,
        statuses=PHASE3_STATUSES,
    )


# --- Phase 5: select (click) the chosen seats --------------------------------
# The seat map is already open (Phase 3 left it there; the Phase 4 interrupt performs no
# browser action and the BrowserSession persists across phases). The agent recovers each
# seat's ref itself: it takes a fresh snapshot and matches the accessible name — no ref is
# threaded through state (Phase 3's offered_seats keeps only the durable label; see
# seat_extraction). Reconstructing the name from a label is a plain f"Silla {label}".

# Phase 5 outcomes. Like Phase 3, only the happy token is advertised in the STATUS
# instruction; NEEDS_RETRY is the parser fallback when the agent can't cleanly select
# every seat (e.g. a seat was taken between Phase 4 and Phase 5, or the map expired).
PHASE5_SEATS_SELECTED = "SEATS_SELECTED"
PHASE5_NEEDS_RETRY = "NEEDS_RETRY"  # fallback / no valid status

PHASE5_STATUSES = {
    PHASE5_SEATS_SELECTED,
}

_PHASE5_GOAL = """\
Select the seats the user chose, then confirm the selection to advance.
The interactive seat map is already open from the previous step.

Seats to select: {named_seats}

Work through the chosen seats ONE AT A TIME:
1. get_snapshot to read the current seat map and its fresh element refs.
2. Find the button for the next chosen seat you have NOT clicked yet. An available
   seat's accessible name ends with its label — "Silla K10", or for a companion
   seat "Silla acompañante K10". Skip any seat whose name contains "no disponible"
   (already taken).
3. click that seat's ref exactly once. Clicking selects the seat.
4. get_snapshot again — the click changed the page and regenerated every ref.

CRITICAL — do not toggle seats. Click each chosen seat EXACTLY ONCE. Keep track of
which seats you have already clicked; a seat you already selected will look
different now (highlighted, or its name changes — e.g. it no longer reads plainly
"Silla <LABEL>"). NEVER click a seat you have already selected: a second click
DESELECTS it and you will loop forever. If all your chosen seats already appear
selected, stop clicking seats and move on.

Once EVERY chosen seat is selected, confirm the selection:
5. Find and click the button that continues to the next step — the bottom button
   labelled "Seleccionar boletas" (accept "Continuar" or "Confirmar" if that exact
   label is not present). It may only appear once the required number of seats is
   selected, so if you do not see it, get_snapshot once more and look again.
6. get_snapshot to verify the page advanced past the seat map.

If a chosen seat is missing or reads "no disponible", do not substitute another
seat — report that you could not select it. Do NOT log in or enter any payment
details; your job ends once you have clicked "Seleccionar boletas"."""

_PHASE5_ENDS_WHEN = (
    'every chosen seat is selected and you have clicked "Seleccionar boletas" '
    '("Continuar"/"Confirmar") to advance past the seat map (or a chosen seat is '
    "unavailable and cannot be selected)"
)


def phase5_select_seats_prompt(chosen_seats: list[str]) -> str:
    """System prompt for Phase 5 — click each chosen seat on the open seat map.

    ``chosen_seats`` are the durable labels the Phase 4 interrupt validated and persisted
    (e.g. ``["K10", "K11"]``). Each is rendered as its expected accessible name
    (``"Silla K10"``) so the agent can find the matching button in a fresh snapshot and
    click its ref — the ref itself is never carried in state (see ``seat_extraction``).
    """
    named_seats = ", ".join(f'"Silla {label}"' for label in chosen_seats)
    return build_phase_prompt(
        goal=_PHASE5_GOAL.format(named_seats=named_seats),
        ends_when=_PHASE5_ENDS_WHEN,
        statuses=PHASE5_STATUSES,
    )


# --- Phase 6: prepare the order for payment ----------------------------------
# After Phase 5 clicks "Seleccionar boletas" the site redirects from the seat map (/seats)
# to the ticket-quantity page (/tickets). Phase 6 adds the tickets, advances past the
# food & drinks step, and lands the agent on the payment/form page (Phase 7's start). Like
# Phase 5 the LLM does all the clicking and there is nothing to parse afterwards.

# Phase 6 outcomes. Like Phase 5, only the happy token is advertised in the STATUS
# instruction; NEEDS_RETRY is the parser fallback when the agent can't prepare the order
# (e.g. the quantity stepper or a continue button never appears).
PHASE6_ORDER_PREPARED = "ORDER_PREPARED"
PHASE6_NEEDS_RETRY = "NEEDS_RETRY"  # fallback / no valid status

PHASE6_STATUSES = {
    PHASE6_ORDER_PREPARED,
}

_PHASE6_GOAL = """\
Prepare the order for payment. The ticket-quantity page is already open from the previous
step (Phase 5 clicked "Seleccionar boletas" and the site advanced to the /tickets page).

You must add {seat_quantity} tickets, advance, and then skip food & drinks.

Add the tickets ONE CLICK AT A TIME:
1. get_snapshot to read the current page and its fresh element refs.
2. Find the "Increase quantity" plus button — its accessible name is "Increase quantity".
   If several ticket categories are listed, each with its own stepper, use the FIRST
   (topmost / general-admission) one.
3. click that button's ref exactly once. This adds one ticket.
4. get_snapshot again — the click changed the page and regenerated every ref. Confirm the
   quantity went up by one.
Repeat steps 2-4 until you have clicked the plus button EXACTLY {seat_quantity} times, so
the ticket quantity reads {seat_quantity}. NEVER guess a ref — only use refs from the most
recent snapshot. Do not over-click past {seat_quantity}.

Once the quantity is {seat_quantity}, advance:
5. Find and click the button labelled "Siguiente" to continue to the next step.
6. get_snapshot to verify the page advanced — a food-and-drinks page is now shown.

On the food-and-drinks page, do NOT add any food or drink:
7. Find and click the button labelled "Continuar con el pago" to advance to the payment step.
8. get_snapshot to verify the page advanced past food & drinks.

Do NOT log in or enter any payment details; your job ends once you have clicked
"Continuar con el pago" and the page has advanced to the payment step."""

_PHASE6_ENDS_WHEN = (
    'you have added {seat_quantity} tickets, clicked "Siguiente", and clicked '
    '"Continuar con el pago" to reach the payment step'
)


def phase6_prepare_order_prompt(seat_quantity: int) -> str:
    """System prompt for Phase 6 — add the tickets, then advance past food & drinks.

    ``seat_quantity`` is the number of tickets to add (the same count the user booked
    seats for; default 2). The agent clicks the first "Increase quantity" stepper that
    many times, clicks "Siguiente", then clicks "Continuar con el pago" without selecting
    any food or drink — leaving it on the payment/form page for Phase 7.
    """
    return build_phase_prompt(
        goal=_PHASE6_GOAL.format(seat_quantity=seat_quantity),
        ends_when=_PHASE6_ENDS_WHEN.format(seat_quantity=seat_quantity),
        statuses=PHASE6_STATUSES,
    )
