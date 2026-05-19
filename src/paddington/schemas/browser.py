from pydantic import BaseModel, Field

# These are the tool entry/exit contracts:


class NavigateResult(BaseModel):
    final_url: str = Field(
        description="URL after redirects (may differ from the requested URL).",
    )
    title: str = Field(description="Document title of the loaded page.")
    status: int = Field(
        ge=0,
        description="HTTP status of the main response (0 if navigation failed before a response).",
    )
    ok: bool = Field(
        description="True when status < 400 and the page is still open.",
    )
    elapsed_ms: int = Field(
        ge=0,
        description="Actual browsing duration in milliseconds.",
    )
    error: str | None = Field(
        default=None,
        description="Error message if ok=False (timeout, DNS, etc.).",
    )


# What the LLM sees (is serialized to the context)
class InteractiveElement(BaseModel):
    ref: str = Field(
        description="Stable reference id used to click/input later, e.g. 'el_12'.",
    )
    role: str = Field(
        description="ARIA role of the element, e.g. 'link', 'button', 'textbox'.",
    )
    name: str = Field(description="Accessible name shown to the user.")


class PageSnapshot(BaseModel):
    url: str = Field(description="Current URL of the page.")
    title: str = Field(description="Document title.")
    markdown: str = Field(
        repr=False,
        description="Textual content, simplified and possibly truncated.",
    )
    interactive_elements: list[InteractiveElement] = Field(
        default_factory=list,
        description="Clickable / typeable elements available on the page.",
    )
    truncated: bool = Field(description="True if markdown was cut due to size limits.")
    total_chars: int = Field(
        ge=0,
        description="Original size of the markdown before truncating.",
    )


class ClickResult(BaseModel):
    success: bool = Field(description="Whether the click was dispatched without error.")
    previous_url: str = Field(description="URL before clicking.")
    current_url: str = Field(description="URL after clicking.")
    navigated: bool = Field(description="True when previous_url != current_url.")
    elapsed_ms: int = Field(
        ge=0,
        description="Duration of the click action in milliseconds.",
    )
    error: str | None = Field(
        default=None,
        description="Error message if success=False.",
    )


class InputResult(BaseModel):
    success: bool = Field(description="Whether the input action completed.")
    ref: str = Field(description="Reference id of the element that was written into.")
    value_set: str = Field(description="The value finally set on the element.")
    elapsed_ms: int = Field(
        ge=0,
        description="Duration of the input action in milliseconds.",
    )
    error: str | None = Field(
        default=None,
        description="Error message if success=False.",
    )
