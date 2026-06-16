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

from paddington.schemas.browser import (
    ClickResult,
    InputResult,
    InteractiveElement,
    NavigateResult,
    PageSnapshot,
)

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
    return { ref, role, name };
  });
}
"""

_NOISE_TAGS = ("script", "style", "noscript", "svg", "iframe", "link", "meta", "head")


class BrowserSession:
    def __init__(self, context: BrowserContext, page: Page):
        self.context = context
        self.page = page
        self._ref_map: dict[str, str] = {}

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

        return NavigateResult(
            final_url=final_url,
            title=title,
            status=status,
            ok=ok,
            elapsed_ms=elapsed_ms,
            error=error,
        )

    async def get_snapshot(self, max_chars: int = 8_000) -> PageSnapshot:
        self._ref_map.clear()

        raw_elements: list[dict] = await self.page.evaluate(_COLLECT_INTERACTIVE_JS)
        for el in raw_elements:
            self._ref_map[el["ref"]] = f'[data-paddington-ref="{el["ref"]}"]'
        interactive_elements = [InteractiveElement(**el) for el in raw_elements]

        html = await self.page.content()
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

        return PageSnapshot(
            url=self.page.url,
            title=await self.page.title(),
            markdown=markdown,
            interactive_elements=interactive_elements,
            truncated=truncated,
            total_chars=total_chars,
        )

    async def click(self, ref: str, timeout_ms: int = 30_000) -> ClickResult:
        start = time.perf_counter()
        previous_url = self.page.url
        error: str | None = None

        selector = self._ref_map.get(ref)
        if selector is None:
            return ClickResult(
                success=False,
                previous_url=previous_url,
                current_url=previous_url,
                navigated=False,
                elapsed_ms=int((time.perf_counter() - start) * 1000),
                error=(f"unknown ref {ref!r}. Call get_snapshot() to refresh available refs."),
            )

        try:
            await self.page.locator(selector).click(timeout=timeout_ms)
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

        return InputResult(
            success=success,
            ref=ref,
            value_set=value_set,
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
        session = self._sessions.get(thread_id)
        if session is None:
            if self._browser is None:
                raise RuntimeError(
                    "BrowserSessionManager used outside its async context; "
                    "enter it with `async with` (or AsyncExitStack) first."
                )
            context = await self._browser.new_context()
            page = await context.new_page()
            session = BrowserSession(context, page)
            self._sessions[thread_id] = session
        return session
