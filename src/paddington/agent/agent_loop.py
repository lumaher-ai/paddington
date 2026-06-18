from dataclasses import dataclass
from typing import Any

import litellm
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from paddington.config import get_settings
from paddington.exceptions import PaddingtonError
from paddington.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
Use them when needed to answer the user's question accurately.
If you can answer without tools, do so directly.
Be concise and cite your sources when using search results.

## Browser tools

You can navigate and interact with real websites.
Follow this workflow:

1. Call navigate_to to go to a URL
2. Call get_snapshot to see the page content and available interactive elements
3. Each element has a ref (like "el_9") — use these refs in click_element and input_text
4. After EVERY action (click, input), call get_snapshot again — the page has changed
5. NEVER guess a ref — only use refs from the most recent snapshot
6. If a ref fails, call get_snapshot to get fresh refs and retry

Use take_screenshot to visually verify a page explicitly when markdown isn't enough, 
not something that happens automatically on every step. — layout,CAPTCHA 
rendered charts/canvas, or other visual state get_snapshot can't convey. Keep
using get_snapshot for reading text and for the refs needed to click/type;
screenshots are costly, so use them sparingly.

When asked to find information on a website, navigate there, interact as needed,
and extract the specific data requested. Return structured results.
"""


class AgentBudgetExceededError(PaddingtonError):
    status_code = 429


class AgentRecursionLimitError(PaddingtonError):
    status_code = 422


class AgentExecutionError(PaddingtonError):
    status_code = 502


@dataclass
class AgentResult:
    answer: str
    iterations: int
    tools_used: list[str]
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float


@dataclass
class AgentConfig:
    model: str = "gpt-4o-mini"
    max_iterations: int = 3
    max_cost_usd: float = 0.50
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT


@dataclass
class AgentInvocationContext:
    """Per-invocation runtime context (NOT persisted in the checkpoint).

    Tells the budget middleware how many messages already existed at the
    start of this turn, so it can compute cost/tokens only from new messages.
    """

    baseline_message_count: int


_INTERRUPTED_TOOL_RESULT = (
    "The previous tool call did not complete because the run was interrupted. "
    "Retry the action if it is still needed."
)


def _dangling_tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    """Return synthetic ToolMessages for any tool_call left without a result.

    An interruption mid-tool-execution (a crashing tool, a process kill, a
    recursion/budget abort) can leave an AIMessage whose tool_calls were never
    answered by a ToolMessage. Replaying that history makes the OpenAI/Anthropic
    APIs reject the request ("tool_call_ids did not have response messages"). We
    synthesize a placeholder result per dangling id so the history is valid again.
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    repairs: list[ToolMessage] = []
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        for call in m.tool_calls or []:
            call_id = call.get("id")
            if call_id and call_id not in answered:
                repairs.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        name=call.get("name"),
                        content=_INTERRUPTED_TOOL_RESULT,
                    )
                )
                answered.add(call_id)
    return repairs


class AgentLoop:
    """ReAct agent powered by LangGraph + ChatLiteLLM."""

    def __init__(
        self,
        tools: list[BaseTool],
        config: AgentConfig | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self._config = config or AgentConfig()
        self._checkpointer = checkpointer
        self._graph = self._build_graph(tools)

    def _build_graph(self, tools: list[BaseTool]) -> CompiledStateGraph[Any, Any, Any, Any]:
        settings = get_settings()
        model = ChatLiteLLM(
            model=self._config.model,
            temperature=0.0,
            max_retries=3,
            model_kwargs={"fallbacks": [settings.fallback_model]},
        )
        return create_agent(
            model,
            tools=tools,
            system_prompt=self._config.system_prompt,
            middleware=[BudgetMiddleware(self._config.max_cost_usd, self._config.model)],
            context_schema=AgentInvocationContext,
            checkpointer=self._checkpointer,
        )

    async def run(self, user_message: str, thread_id: str) -> AgentResult:
        invoke_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._config.max_iterations * 2,
        }

        baseline = 0
        dangling: list[ToolMessage] = []
        if self._checkpointer is not None:
            snapshot = await self._graph.aget_state({"configurable": {"thread_id": thread_id}})
            if snapshot and snapshot.values:
                messages = snapshot.values.get("messages", [])
                baseline = len(messages)

                # Repair any tool_call left unanswered by a prior interrupted run.
                # Without this, the persisted history is invalid and every future
                # request on this thread is rejected by the LLM API. We inject the
                # synthetic results into this turn's input so the add_messages
                # reducer appends them right after the dangling tool_call, before
                # the new user message.
                dangling = _dangling_tool_messages(messages)
                if dangling:
                    logger.warning(
                        "repaired_dangling_tool_calls",
                        thread_id=thread_id,
                        count=len(dangling),
                    )

        try:
            final = await self._graph.ainvoke(
                {"messages": [*dangling, HumanMessage(content=user_message)]},
                config=invoke_config,
                context=AgentInvocationContext(baseline_message_count=baseline),
            )
        except GraphRecursionError as exc:
            logger.warning(
                "agent_recursion_limit_reached",
                max_iterations=self._config.max_iterations,
                recursion_limit=invoke_config["recursion_limit"],
                thread_id=thread_id,
            )
            raise AgentRecursionLimitError(
                f"Agent stopped after reaching its step limit "
                f"(max_iterations={self._config.max_iterations}) without finishing. "
                f"Retry with a higher max_iterations for multi-step tasks like browsing."
            ) from exc

        new_messages = final["messages"][baseline:]
        total_cost, total_input, total_output = _accumulate_usage(new_messages, self._config.model)

        iterations = sum(1 for m in new_messages if isinstance(m, AIMessage))
        tools_used = [
            tc["name"]
            for m in new_messages
            if isinstance(m, AIMessage)
            for tc in (m.tool_calls or [])
        ]
        answer = next(
            (
                m.content
                for m in reversed(new_messages)
                if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content
            ),
            "",
        )

        if total_cost > self._config.max_cost_usd:
            raise AgentBudgetExceededError(
                f"Agent exceeded budget: ${total_cost:.4f} > ${self._config.max_cost_usd:.2f}"
            )

        logger.info(
            "agent_completed",
            iterations=iterations,
            tools_used=tools_used,
            total_cost=round(total_cost, 6),
        )

        return AgentResult(
            answer=answer,
            iterations=iterations,
            tools_used=tools_used,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cost_usd=round(total_cost, 6),
        )


def _accumulate_usage(messages: list, model: str) -> tuple[float, int, int]:
    """Sum cost + tokens across every AIMessage in the given slice."""
    total_cost = 0.0
    total_input = 0
    total_output = 0
    for msg in messages:
        if not isinstance(msg, AIMessage) or not msg.usage_metadata:
            continue
        input_tokens = msg.usage_metadata.get("input_tokens", 0)
        output_tokens = msg.usage_metadata.get("output_tokens", 0)
        total_input += input_tokens
        total_output += output_tokens
        try:
            in_cost, out_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            total_cost += in_cost + out_cost
        except Exception as e:
            logger.warning("cost_calculation_failed", model=model, error=str(e))
    return total_cost, total_input, total_output


class BudgetMiddleware(AgentMiddleware):
    """Halt the agent when *this turn's* LLM cost exceeds the configured budget.

    Reads `runtime.context.baseline_message_count` to ignore messages from
    previous turns when computing cost. If we're over budget and the model
    just requested tools, strip the tool_calls so the loop terminates;
    `AgentLoop.run` then raises `AgentBudgetExceededError`.
    """

    def __init__(self, max_cost_usd: float, model: str) -> None:
        super().__init__()
        self._max_cost_usd = max_cost_usd
        self._model = model

    def after_model(
        self, state: AgentState, runtime: Runtime[AgentInvocationContext]
    ) -> dict[str, Any] | None:
        baseline = runtime.context.baseline_message_count if runtime.context else 0
        new_messages = state["messages"][baseline:]
        total_cost, _, _ = _accumulate_usage(new_messages, self._model)
        if total_cost <= self._max_cost_usd:
            return None

        logger.warning(
            "agent_budget_exceeded",
            cost=round(total_cost, 6),
            budget=self._max_cost_usd,
        )
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            stripped = AIMessage(
                id=last.id,
                content=last.content or "Stopped: budget exceeded before completing the answer.",
                response_metadata=last.response_metadata,
                usage_metadata=last.usage_metadata,
            )
            return {"messages": [stripped]}
        return None
