"""Unit tests for BrowserSession click/input actionability handling.

These use lightweight fakes (no real browser) to prove the fast-fail path: a disabled
element must return immediately with an informative error instead of blocking for the full
Playwright actionability timeout (the Phase 6 "+"-stepper hang).
"""

from typing import cast

from playwright.async_api import Page

from paddington.browser.browser_session import BrowserSession


class _FakeLocator:
    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self.click_called = False

    async def is_enabled(self, timeout: float | None = None) -> bool:
        return self._enabled

    async def click(self, timeout: float | None = None) -> None:
        self.click_called = True


class _FakePage:
    """Minimal Page stand-in: enough surface for BrowserSession.click()."""

    def __init__(self, locator: _FakeLocator) -> None:
        self.url = "https://example.com/tickets"
        self._locator = locator

    def locator(self, selector: str) -> _FakeLocator:
        return self._locator

    def is_closed(self) -> bool:
        return False

    async def wait_for_load_state(self, *args, **kwargs) -> None:
        return None


def _session(locator: _FakeLocator) -> BrowserSession:
    session = BrowserSession(context=cast(object, object()), page=cast(Page, _FakePage(locator)))
    session._ref_map = {"el_5": '[data-paddington-ref="el_5"]'}
    return session


async def test_click_disabled_element_fails_fast_without_clicking() -> None:
    # The maxed "+" quantity stepper is disabled: click() must NOT block on it and must NOT
    # attempt the click — it returns an informative error the agent can act on.
    locator = _FakeLocator(enabled=False)
    session = _session(locator)

    result = await session.click("el_5")

    assert result.success is False
    assert "disabled" in (result.error or "")
    assert locator.click_called is False
    # Returns essentially instantly — nowhere near the 30s actionability timeout.
    assert result.elapsed_ms < 1_000


async def test_click_enabled_element_is_clicked() -> None:
    locator = _FakeLocator(enabled=True)
    session = _session(locator)

    result = await session.click("el_5")

    assert result.success is True
    assert result.error is None
    assert locator.click_called is True
