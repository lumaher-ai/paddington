import asyncio
import contextlib
import re
import time

from bs4 import BeautifulSoup
from markdownify import markdownify
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from paddington.browser.debug_recorder import DebugRecorder
from paddington.config import get_settings
from paddington.logging_config import get_logger
from paddington.schemas.browser import (
    ClickResult,
    FillCheckoutFormResult,
    InputResult,
    InteractiveElement,
    NavigateResult,
    PageSnapshot,
)

logger = get_logger(__name__)

_COLLECT_INTERACTIVE_JS = """
() => {
  const SELECTOR = [
    'a[href]',
    'button',
    'input:not([type="hidden"])',
    'textarea',
    'select',
    '[role="button"]',
    '[role="link"]',
    '[role="textbox"]',
    '[contenteditable="true"]',
    '[tabindex]:not([tabindex="-1"])',
  ].join(',');

  const TAG_TO_ROLE = {
    A: 'link', BUTTON: 'button',
    INPUT: 'textbox', TEXTAREA: 'textbox',
    SELECT: 'combobox',
  };

  const isVisible = (el) => {
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return false;
    const style = window.getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none') return false;
    return true;
  };

  const elements = Array.from(document.querySelectorAll(SELECTOR)).filter(isVisible);

  return elements.map((el, i) => {
    const ref = `el_${i + 1}`;
    el.setAttribute('data-paddington-ref', ref);
    const role = el.getAttribute('role') || TAG_TO_ROLE[el.tagName] || el.tagName.toLowerCase();
    const name = (
      el.getAttribute('aria-label')
      || el.innerText
      || el.value
      || el.getAttribute('alt')
      || el.getAttribute('placeholder')
      || el.getAttribute('title')
      || ''
    ).trim().replace(/\\s+/g, ' ').slice(0, 200);
    const href = el.tagName === 'A' ? el.href : null;
    // Disabled/inert controls: native `disabled` (button/input/...) or aria-disabled.
    // Surfaced so the agent doesn't keep clicking a maxed/greyed control (e.g. a "+"
    // quantity stepper at its cap) and so click() can fail fast instead of blocking.
    const disabled = el.disabled === true || el.getAttribute('aria-disabled') === 'true';
    return { ref, role, name, href, disabled };
  });
}
"""

_NOISE_TAGS = ("script", "style", "noscript", "svg", "iframe", "link", "meta", "head")


class BrowserSession:
    def __init__(self, context: BrowserContext, page: Page):
        self.context = context
        self.page = page
        self._ref_map: dict[str, str] = {}
        # The most recent snapshot this session read, retained so a phase node can
        # deterministically pull data the LLM saw but must not transcribe — e.g. the
        # href of a showtime link matched to a chosen screening. Set by get_snapshot.
        self.last_snapshot: PageSnapshot | None = None
        # Debug-only per-run screenshot trail; swapped in at the start of each
        # /agent/run. None disables capture entirely (the default / prod path).
        # No lock needed: a conversation is sequential, so runs on one session
        # don't overlap.
        self.recorder: DebugRecorder | None = None

    async def _capture_debug(self, tool_name: str) -> None:
        """Save a viewport screenshot to the debug trail; never break the run.

        Calls page.screenshot directly (not self.screenshot) to avoid the
        per-call info log and any chance of recursion.
        """
        # recorder disabled (None) or turned off mid-run by a PII phase -> capture nothing.
        if self.recorder is None or not self.recorder.enabled:
            return
        try:
            png = await self.page.screenshot(full_page=False, type="png")
            self.recorder.write(tool_name, png)
        except Exception as e:
            logger.warning("debug_capture_failed", tool=tool_name, error=str(e))

    async def navigate(
        self,
        url: str,
        wait_until: str = "load",
        timeout_ms: int = 30_000,
    ) -> NavigateResult:
        start = time.perf_counter()
        status = 0
        error: str | None = None

        try:
            response = await self.page.goto(
                url,
                wait_until=wait_until,  # type: ignore[arg-type]
                timeout=timeout_ms,
            )
            if response is not None:
                status = response.status
        except PlaywrightTimeoutError:
            error = f"navigation timeout after {timeout_ms}ms"
        except PlaywrightError as e:
            error = str(e)

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        final_url = url
        with contextlib.suppress(PlaywrightError):
            final_url = self.page.url

        try:
            title = await self.page.title()
        except PlaywrightError:
            title = ""

        ok = error is None and status < 400 and not self.page.is_closed()

        await self._capture_debug("navigate")
        return NavigateResult(
            final_url=final_url,
            title=title,
            status=status,
            ok=ok,
            elapsed_ms=elapsed_ms,
            error=error,
        )

    async def get_snapshot(self, max_chars: int = 6_500) -> PageSnapshot:
        self._ref_map.clear()

        # Let any in-flight navigation settle before we read the page. A snapshot
        # issued mid-navigation otherwise tears down the JS execution context and
        # raises "Execution context was destroyed" from page.evaluate().
        with contextlib.suppress(PlaywrightError):
            await self.page.wait_for_load_state("domcontentloaded", timeout=5_000)

        try:
            # Retry once on the navigation race: if the context is destroyed while
            # collecting elements, wait for the page to settle and try again.
            for attempt in range(2):
                try:
                    raw_elements: list[dict] = await self.page.evaluate(_COLLECT_INTERACTIVE_JS)
                    break
                except PlaywrightError:
                    if attempt == 1:
                        raise
                    with contextlib.suppress(PlaywrightError):
                        await self.page.wait_for_load_state("domcontentloaded", timeout=5_000)

            for el in raw_elements:
                self._ref_map[el["ref"]] = f'[data-paddington-ref="{el["ref"]}"]'
            interactive_elements = [InteractiveElement(**el) for el in raw_elements]

            html = await self.page.content()
            title = await self.page.title()
        except PlaywrightError as e:
            url = self.page.url
            logger.warning("browser_snapshot_failed", url=url, error=str(e))
            self.last_snapshot = PageSnapshot(
                url=url,
                title="",
                markdown="",
                interactive_elements=[],
                truncated=False,
                total_chars=0,
                error=str(e),
            )
            return self.last_snapshot

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(list(_NOISE_TAGS)):
            tag.decompose()
        for hidden in soup.find_all(attrs={"aria-hidden": "true"}):
            hidden.decompose()
        for hidden in soup.find_all(attrs={"hidden": True}):
            hidden.decompose()

        body = soup.body or soup
        markdown_raw = markdownify(str(body), heading_style="ATX").strip()
        markdown_clean = re.sub(r"\n{3,}", "\n\n", markdown_raw)

        total_chars = len(markdown_clean)
        truncated = total_chars > max_chars
        markdown = markdown_clean[:max_chars] if truncated else markdown_clean

        logger.info(
            "browser_snapshot_captured",
            url=self.page.url,
            interactive_count=len(interactive_elements),
            total_chars=total_chars,
            truncated=truncated,
            returned_chars=len(markdown),
        )

        await self._capture_debug("get_snapshot")
        self.last_snapshot = PageSnapshot(
            url=self.page.url,
            title=title,
            markdown=markdown,
            interactive_elements=interactive_elements,
            truncated=truncated,
            total_chars=total_chars,
        )
        return self.last_snapshot

    async def screenshot(self, *, full_page: bool = False) -> bytes:
        """Capture the current page as PNG bytes.

        Mirrors get_snapshot's settle-before-read: let any in-flight navigation
        reach a stable state so we don't screenshot a blank/transitional frame.
        Raises PlaywrightError if the page is closed or the capture fails; the
        caller (the tool) turns that into a text-only result.
        """
        with contextlib.suppress(PlaywrightError):
            await self.page.wait_for_load_state("domcontentloaded", timeout=5_000)

        png = await self.page.screenshot(full_page=full_page, type="png")

        logger.info(
            "browser_screenshot_captured",
            url=self.page.url,
            full_page=full_page,
            size_bytes=len(png),
        )
        return png

    def _describe_ref(self, ref: str) -> tuple[str | None, str | None]:
        """Resolve a ref's accessible name (aria-label) and role from the last snapshot.

        The ``_ref_map`` only stores CSS selectors, so the human-meaningful element
        metadata — the accessible name the LLM matched on, e.g. ``"Increase quantity"`` —
        comes from ``last_snapshot.interactive_elements``. Used purely for logging so a
        click trace says *what* was clicked, not just an opaque ref. Returns ``(None, None)``
        when the ref isn't in the most recent snapshot (e.g. a stale ref).
        """
        snapshot = self.last_snapshot
        if snapshot is None:
            return None, None
        for el in snapshot.interactive_elements:
            if el.ref == ref:
                return el.name, el.role
        return None, None

    async def click(self, ref: str, timeout_ms: int = 30_000) -> ClickResult:
        start = time.perf_counter()
        previous_url = self.page.url
        error: str | None = None

        # Resolve the element's accessible name/role from the last snapshot up front so
        # every log line below names *what* was clicked (its aria-label), not just its ref.
        name, role = self._describe_ref(ref)

        selector = self._ref_map.get(ref)
        if selector is None:
            # The LLM asked for a ref that isn't in the current snapshot — the classic
            # "can't find/click the reference" failure. Log it (with how many refs *are*
            # available) so the cause is visible instead of silently returned to the model.
            logger.warning(
                "browser_click_unknown_ref",
                ref=ref,
                name=name,
                url=previous_url,
                known_refs=len(self._ref_map),
            )
            return ClickResult(
                success=False,
                previous_url=previous_url,
                current_url=previous_url,
                navigated=False,
                elapsed_ms=int((time.perf_counter() - start) * 1000),
                error=(f"unknown ref {ref!r}. Call get_snapshot() to refresh available refs."),
            )

        locator = self.page.locator(selector)

        # Fast-fail on inert elements. A disabled control never becomes actionable, so a plain
        # click() would block for the FULL timeout (30s) and the agent would retry it in a loop
        # (the Phase 6 "+" stepper at its max quantity did exactly this). Probe the enabled
        # state cheaply first — is_enabled() honours native `disabled` and `aria-disabled` — and
        # return immediately with an informative error instead of hanging.
        try:
            enabled = await locator.is_enabled(timeout=2_000)
        except PlaywrightError:
            enabled = True  # couldn't determine (detached/late render) — fall through to click
        if not enabled:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "browser_click_disabled",
                ref=ref,
                name=name,
                role=role,
                url=previous_url,
                elapsed_ms=elapsed_ms,
            )
            return ClickResult(
                success=False,
                previous_url=previous_url,
                current_url=previous_url,
                navigated=False,
                elapsed_ms=elapsed_ms,
                error=(
                    f"element {ref!r} ({name!r}) is disabled and cannot be clicked — it may have "
                    "reached its limit or be inactive. Do NOT retry it; choose another action."
                ),
            )

        try:
            await locator.click(timeout=timeout_ms)
        except PlaywrightTimeoutError:
            error = f"click timed out after {timeout_ms}ms"
        except PlaywrightError as e:
            error = str(e)
        else:
            with contextlib.suppress(PlaywrightError):
                await self.page.wait_for_load_state("load", timeout=5_000)

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        current_url = previous_url
        with contextlib.suppress(PlaywrightError):
            current_url = self.page.url

        success = error is None and not self.page.is_closed()

        # Log the outcome with the element's accessible name so a failed booking step is
        # debuggable: which element (aria-label), whether the page moved, and why it failed.
        if success:
            logger.info(
                "browser_click",
                ref=ref,
                name=name,
                role=role,
                navigated=previous_url != current_url,
                current_url=current_url,
                elapsed_ms=elapsed_ms,
            )
        else:
            logger.warning(
                "browser_click_failed",
                ref=ref,
                name=name,
                role=role,
                error=error,
                url=current_url,
                elapsed_ms=elapsed_ms,
            )

        await self._capture_debug("click")
        return ClickResult(
            success=success,
            previous_url=previous_url,
            current_url=current_url,
            navigated=previous_url != current_url,
            elapsed_ms=elapsed_ms,
            error=error,
        )

    async def input_text(
        self,
        ref: str,
        text: str,
        clear_first: bool = True,
        press_enter: bool = True,
        timeout_ms: int = 30_000,
    ) -> InputResult:
        start = time.perf_counter()
        error: str | None = None
        value_set = ""

        selector = self._ref_map.get(ref)
        if selector is None:
            return InputResult(
                success=False,
                ref=ref,
                value_set="",
                elapsed_ms=int((time.perf_counter() - start) * 1000),
                error=f"unknown ref {ref!r}. Call get_snapshot() to refresh available refs.",
            )

        locator = self.page.locator(selector)

        # Same fast-fail as click(): a disabled field never becomes fillable, so bail out with
        # a clear error rather than blocking for the full timeout.
        try:
            enabled = await locator.is_enabled(timeout=2_000)
        except PlaywrightError:
            enabled = True
        if not enabled:
            return InputResult(
                success=False,
                ref=ref,
                value_set="",
                elapsed_ms=int((time.perf_counter() - start) * 1000),
                error=f"element {ref!r} is disabled and cannot be typed into.",
            )

        try:
            if clear_first:
                await locator.fill(text, timeout=timeout_ms)
            else:
                await locator.press_sequentially(text, timeout=timeout_ms)
            if press_enter:
                await locator.press("Enter", timeout=timeout_ms)
                with contextlib.suppress(PlaywrightError):
                    await self.page.wait_for_load_state("load", timeout=5_000)
        except PlaywrightTimeoutError:
            error = f"input timed out after {timeout_ms}ms"
        except PlaywrightError as e:
            error = str(e)

        if error is None:
            try:
                value_set = await locator.input_value(timeout=2_000)
            except PlaywrightError:
                value_set = text

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        success = error is None and not self.page.is_closed()

        await self._capture_debug("input_text")
        return InputResult(
            success=success,
            ref=ref,
            value_set=value_set,
            elapsed_ms=elapsed_ms,
            error=error,
        )

    async def fill_checkout_form(
        self,
        first_name_ref: str,
        last_name_ref: str,
        email_ref: str,
        document_ref: str,
        timeout_ms: int = 30_000,
    ) -> FillCheckoutFormResult:
        """Fill the checkout PII fields from settings, keyed by ref.

        The caller (LLM) only supplies which ref is which field; the actual values
        come from settings and are resolved here so PII never crosses the tool
        boundary into the conversation / checkpoint / model API.
        """
        start = time.perf_counter()
        error: str | None = None
        filled_refs: list[str] = []
        failed_refs: dict[str, str] = {}

        settings = get_settings()
        # Ordered so the fill happens top-to-bottom as a human would.
        field_map = {
            first_name_ref: settings.booking_full_name,
            last_name_ref: settings.booking_last_name,
            email_ref: settings.booking_email,
            document_ref: settings.booking_dni,
        }

        for ref, value in field_map.items():
            selector = self._ref_map.get(ref)
            if selector is None:
                failed_refs[ref] = (
                    f"unknown ref {ref!r}. Call get_snapshot() to refresh available refs."
                )
                continue

            locator = self.page.locator(selector)

            # Same fast-fail as input_text: a disabled field never becomes fillable.
            try:
                enabled = await locator.is_enabled(timeout=2_000)
            except PlaywrightError:
                enabled = True
            if not enabled:
                failed_refs[ref] = f"element {ref!r} is disabled and cannot be typed into."
                continue

            try:
                # No press_enter: filling a checkout form must not submit it.
                await locator.fill(value, timeout=timeout_ms)
                filled_refs.append(ref)
            except PlaywrightTimeoutError:
                failed_refs[ref] = f"input timed out after {timeout_ms}ms"
            except PlaywrightError as e:
                failed_refs[ref] = str(e)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        success = not failed_refs and error is None and not self.page.is_closed()

        # Intentionally do NOT call self._capture_debug(): the filled form shows the
        # user's name/email/document, and a debug screenshot would persist that PII to
        # disk. Keep this omission — do not "restore" the capture for consistency.
        logger.info(
            "browser_fill_checkout_form",
            filled=len(filled_refs),
            failed=len(failed_refs),
            elapsed_ms=elapsed_ms,
        )
        return FillCheckoutFormResult(
            success=success,
            filled_refs=filled_refs,
            failed_refs=failed_refs,
            elapsed_ms=elapsed_ms,
            error=error,
        )


class BrowserSessionManager:
    """Owns the Playwright instance and the single browser process, and hands
    out one BrowserSession per thread.

    Self-contained async-context lifecycle (launch in __aenter__, teardown in
    __aexit__) so the application entrypoint never has to know how a browser
    starts or stops — same shape as the AsyncPostgresSaver checkpointer.
    """

    def __init__(self, headless: bool = False) -> None:
        self._headless = headless
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._sessions: dict[str, BrowserSession] = {}
        # Guards the check-and-create in get_or_create so two concurrent
        # requests can't both build a session (and leak a browser context).
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "BrowserSessionManager":
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        for session in self._sessions.values():
            with contextlib.suppress(PlaywrightError):
                await session.context.close()  # closes the page too
        self._sessions.clear()
        if self._browser is not None:
            with contextlib.suppress(PlaywrightError):
                await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()

    async def get_or_create(self, thread_id: str) -> BrowserSession:
        # Fast path: an existing session needs no lock (dict reads are safe
        # between await points in a single event loop).
        session = self._sessions.get(thread_id)
        if session is not None:
            return session

        # Slow path: serialize creation. Re-check inside the lock because another
        # coroutine may have created the session while we waited for it.
        async with self._lock:
            session = self._sessions.get(thread_id)
            if session is None:
                if self._browser is None:
                    raise RuntimeError(
                        "BrowserSessionManager used outside its async context; "
                        "enter it with `async with` (or AsyncExitStack) first."
                    )
                # Stealth context: cinecolombia.com sits behind Cloudflare, which
                # fingerprints Playwright's default Chromium. A real desktop Chrome
                # user-agent + viewport + es-CO locale, plus hiding the
                # navigator.webdriver automation flag before any page script runs,
                # gets us past the bot check.
                context = await self._browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1280, "height": 720},
                    locale="es-CO",
                )
                page = await context.new_page()
                await page.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => false})"
                )
                # Pre-seed CineColombia's cinema-group selection in localStorage so the
                # site treats Bogotá / our two preferred theaters as already chosen and
                # never raises the city/cinema popup. add_init_script runs before every
                # navigation, so this lands before the site's JS reads the store. Trimmed
                # to Andino + Avenida Chile and the fields the selection check needs.
                await page.add_init_script(
                    """
    localStorage.setItem(
        'VistaOmnichannelComponents::cinema-group-domain-store',
        JSON.stringify({
            "json": {
                "selected": {
                    "cinemas": [
                        {
                            "name": "Andino", "urlSegment": "andino",
                            "url": "/cinemas/andino/", "nodeId": 2420, "alias": "cinema",
                            "vistaCinema": {"key": "6493", "value": "ANDINO"}
                        },
                        {
                            "name": "Avenida Chile", "urlSegment": "avenida-chile",
                            "url": "/cinemas/avenida-chile/", "nodeId": 2553,
                            "alias": "cinema",
                            "vistaCinema": {"key": "6669", "value": "AVENIDA CHILE"}
                        }
                    ],
                    "name": "Bogotá",
                    "id": 2140
                }
            }
        })
    );
"""
                )
                session = BrowserSession(context, page)
                self._sessions[thread_id] = session
        return session
