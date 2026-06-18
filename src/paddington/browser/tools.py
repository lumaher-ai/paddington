import base64
from typing import Annotated

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command
from playwright.async_api import Error as PlaywrightError

from paddington.browser.browser_session import BrowserSession
from paddington.schemas.browser import (
    ClickResult,
    InputResult,
    NavigateResult,
    PageSnapshot,
)


def build_browser_tools(session: BrowserSession) -> list[BaseTool]:
    """Build per-session browser tools.

    Each tool closes over the live BrowserSession so it can share page state,
    _ref_map, cookies, etc. across turns of the same conversation.
    """

    @tool
    async def navigate_to(
        url: str,
        wait_until: str = "load",
        timeout_ms: int = 30_000,
    ) -> NavigateResult:
        """Navigate the browser to a URL.

        Use this to load a page before reading it or interacting with it.

        Args:
            url: Absolute URL to load (must include scheme, e.g. https://...).
            wait_until: When navigation is considered done: 'load', 'domcontentloaded',
                'networkidle', or 'commit'. Default: 'load'.
            timeout_ms: Max milliseconds before navigation is considered failed.
        """
        return await session.navigate(url, wait_until=wait_until, timeout_ms=timeout_ms)

    @tool
    async def get_snapshot(max_chars: int = 8_000) -> PageSnapshot:
        """Return the current page as cleaned markdown plus a list of refs for
        interactive elements (links, buttons, inputs).

        Call this after navigating or after a click that changed the page. The
        refs returned here are the only valid refs for `click` and `input_text`
        until the next snapshot.

        Args:
            max_chars: Cap on the markdown size; the field 'truncated' tells
                you whether it was cut.
        """
        return await session.get_snapshot(max_chars=max_chars)

    @tool
    async def click(ref: str, timeout_ms: int = 30_000) -> ClickResult:
        """Click an interactive element identified by its ref from the last snapshot.

        Args:
            ref: Element reference from a recent get_snapshot result (e.g. 'el_12').
            timeout_ms: Max milliseconds to wait for the element to be clickable.
        """
        return await session.click(ref, timeout_ms=timeout_ms)

    @tool
    async def input_text(
        ref: str,
        text: str,
        clear_first: bool = True,
        press_enter: bool = True,
        timeout_ms: int = 30_000,
    ) -> InputResult:
        """Type text into an input field identified by its ref.

        Args:
            ref: Element reference (must point to a textbox).
            text: Text to type.
            clear_first: If True (default), replace the field's contents. If False,
                append onto whatever is already there.
            press_enter: If True, press Enter after typing — useful for search boxes
                that submit on Enter.
            timeout_ms: Max milliseconds for each substep.
        """
        return await session.input_text(
            ref,
            text,
            clear_first=clear_first,
            press_enter=press_enter,
            timeout_ms=timeout_ms,
        )

    @tool
    async def take_screenshot(
        tool_call_id: Annotated[str, InjectedToolCallId],
        full_page: bool = False,
    ) -> Command:
        """Capture the current page as a PNG image you can actually see.

        Use this to visually verify a page — layout, rendered charts/canvas, or
        anything get_snapshot's text misses. Prefer get_snapshot for reading
        text and for the refs needed to click/type; use this when you need to
        *see* the page. Screenshots are token-expensive, so don't call it on
        every step.

        Args:
            full_page: If False (default), capture only the visible viewport. If
                True, capture the entire scrollable page (much larger / costlier).
        """
        try:
            png = await session.screenshot(full_page=full_page)
        except PlaywrightError as e:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Screenshot failed: {e}",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        b64 = base64.b64encode(png).decode()
        scope = "full page" if full_page else "viewport"
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Screenshot captured ({scope}); attached as an image.",
                        tool_call_id=tool_call_id,
                    ),
                    HumanMessage(
                        content=[
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            }
                        ]
                    ),
                ]
            }
        )

    return [navigate_to, get_snapshot, click, input_text, take_screenshot]
