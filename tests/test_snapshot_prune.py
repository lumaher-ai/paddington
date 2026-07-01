"""Unit tests for the snapshot-history pruning helper.

``_prune_snapshots`` is a pure function over a message list, so it is tested directly
without an LLM. It must (a) keep only the most-recent get_snapshot ToolMessage and the
most-recent screenshot image, (b) replace (never remove) stale snapshot content so every
tool_call keeps its response, and (c) not mutate the input messages.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from paddington.agent.agent_loop import (
    _SCREENSHOT_PLACEHOLDER,
    _SNAPSHOT_PLACEHOLDER,
    _prune_snapshots,
)


def _snapshot_turn(call_id: str, page: str) -> list:
    """An AIMessage calling get_snapshot + the ToolMessage answering it."""
    return [
        AIMessage(
            content="",
            tool_calls=[{"id": call_id, "name": "get_snapshot", "args": {}, "type": "tool_call"}],
        ),
        ToolMessage(content=page, tool_call_id=call_id, name="get_snapshot"),
    ]


def _screenshot_turn(call_id: str) -> list:
    """A take_screenshot tool result: a ToolMessage note + the image HumanMessage."""
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"id": call_id, "name": "take_screenshot", "args": {}, "type": "tool_call"}
            ],
        ),
        ToolMessage(content="Screenshot captured (viewport).", tool_call_id=call_id),
        HumanMessage(
            content=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        ),
    ]


def test_keeps_only_last_snapshot() -> None:
    messages = [
        HumanMessage(content="find showtimes"),
        *_snapshot_turn("c1", "PAGE ONE"),
        *_snapshot_turn("c2", "PAGE TWO"),
        *_snapshot_turn("c3", "PAGE THREE"),
    ]

    pruned = _prune_snapshots(messages)

    snapshots = [m for m in pruned if isinstance(m, ToolMessage)]
    assert [m.content for m in snapshots] == [
        _SNAPSHOT_PLACEHOLDER,
        _SNAPSHOT_PLACEHOLDER,
        "PAGE THREE",
    ]


def test_stale_snapshots_keep_tool_call_ids() -> None:
    # Every tool_call must retain a matching ToolMessage response, or the API rejects it.
    messages = [*_snapshot_turn("c1", "PAGE ONE"), *_snapshot_turn("c2", "PAGE TWO")]

    pruned = _prune_snapshots(messages)

    answered = {m.tool_call_id for m in pruned if isinstance(m, ToolMessage)}
    assert answered == {"c1", "c2"}  # none dropped
    # The stale snapshot is still a ToolMessage (content replaced, message not removed).
    assert all(isinstance(m, ToolMessage) for m in pruned if not isinstance(m, AIMessage))


def test_keeps_only_last_screenshot() -> None:
    messages = [
        HumanMessage(content="look at the page"),
        *_screenshot_turn("s1"),
        *_screenshot_turn("s2"),
    ]

    pruned = _prune_snapshots(messages)

    images = [m for m in pruned if isinstance(m, HumanMessage) and isinstance(m.content, list)]
    texts = [m for m in pruned if isinstance(m, HumanMessage) and isinstance(m.content, str)]
    # The first screenshot's image HumanMessage is collapsed to the placeholder string...
    assert _SCREENSHOT_PLACEHOLDER in [m.content for m in texts]
    # ...and exactly one image block survives (the latest).
    assert len(images) == 1


def test_snapshots_and_screenshots_pruned_independently() -> None:
    messages = [
        *_snapshot_turn("c1", "PAGE ONE"),
        *_screenshot_turn("s1"),
        *_snapshot_turn("c2", "PAGE TWO"),
        *_screenshot_turn("s2"),
    ]

    pruned = _prune_snapshots(messages)

    live_snapshots = [
        m for m in pruned if isinstance(m, ToolMessage) and m.name == "get_snapshot"
    ]
    live_images = [
        m for m in pruned if isinstance(m, HumanMessage) and isinstance(m.content, list)
    ]
    assert [m.content for m in live_snapshots] == [_SNAPSHOT_PLACEHOLDER, "PAGE TWO"]
    assert len(live_images) == 1


def test_does_not_mutate_input() -> None:
    original = [*_snapshot_turn("c1", "PAGE ONE"), *_snapshot_turn("c2", "PAGE TWO")]
    contents_before = [m.content for m in original]

    _prune_snapshots(original)

    assert [m.content for m in original] == contents_before  # originals untouched


def test_noop_returns_same_list_identity() -> None:
    # A single snapshot / no snapshots: nothing to prune, list passes through by identity.
    single = [HumanMessage(content="hi"), *_snapshot_turn("c1", "PAGE ONE")]
    assert _prune_snapshots(single) is single

    none = [HumanMessage(content="hi"), AIMessage(content="hello")]
    assert _prune_snapshots(none) is none


def test_snapshot_identified_without_toolmessage_name() -> None:
    # Fallback path: ToolMessage.name unset, identity comes from the AIMessage tool_call.
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "get_snapshot", "args": {}, "type": "tool_call"}],
        ),
        ToolMessage(content="PAGE ONE", tool_call_id="c1"),  # no name=
        AIMessage(
            content="",
            tool_calls=[{"id": "c2", "name": "get_snapshot", "args": {}, "type": "tool_call"}],
        ),
        ToolMessage(content="PAGE TWO", tool_call_id="c2"),
    ]

    pruned = _prune_snapshots(messages)

    snapshots = [m for m in pruned if isinstance(m, ToolMessage)]
    assert [m.content for m in snapshots] == [_SNAPSHOT_PLACEHOLDER, "PAGE TWO"]
