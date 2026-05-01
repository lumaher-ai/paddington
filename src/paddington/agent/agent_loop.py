import json
from dataclasses import dataclass

from openai import AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletionMessageParam,
)

from paddington.agent.tools import PaddingtonTools
from paddington.config import get_settings
from paddington.exceptions import PaddingtonError
from paddington.logging_config import get_logger

logger = get_logger(__name__)


class AgentBudgetExceededError(PaddingtonError):
    status_code = 429


@dataclass
class AgentResult:
    """The final result of an agent run."""

    answer: str
    iterations: int
    tools_used: list[str]
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float


@dataclass
class AgentConfig:
    """Configuration for an agent run."""

    model: str = "gpt-4o-mini"
    max_iterations: int = 10
    max_cost_usd: float = 0.50
    system_prompt: str = (
        "You are a helpful assistant with access to tools. "
        "Use them when needed to answer the user's question accurately. "
        "If you can answer without tools, do so directly. "
        "Be concise and cite your sources when using search results."
    )


class AgentLoop:
    """A production-grade agent loop with tool calling, budgeting, and logging."""

    def __init__(self, tools: PaddingtonTools, config: AgentConfig | None = None) -> None:
        self._tools = tools
        self._config = config or AgentConfig()
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def run(self, user_message: str) -> AgentResult:
        """Execute the agent loop until completion or budget exhaustion."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._config.system_prompt},
            {"role": "user", "content": user_message},
        ]

        tools_used: list[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        iteration = 0

        tool_schemas = self._tools.get_openai_schemas()

        while iteration < self._config.max_iterations:
            iteration += 1

            logger.info(
                "agent_iteration_start",
                iteration=iteration,
                message_count=len(messages),
                accumulated_cost=round(total_cost, 6),
            )

            # OBSERVE + THINK
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                tools=tool_schemas if tool_schemas else omit,
                temperature=0.0,
            )

            choice = response.choices[0]

            # Track tokens and cost
            if response.usage:
                total_input_tokens += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens
                # Simplified cost calc — in production use your LLMClient's pricing
                from paddington.llm.client import _calculate_cost

                iteration_cost = _calculate_cost(
                    self._config.model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
                total_cost += iteration_cost

            # CHECK budget
            if total_cost > self._config.max_cost_usd:
                logger.warning(
                    "agent_budget_exceeded",
                    cost=round(total_cost, 6),
                    budget=self._config.max_cost_usd,
                    iteration=iteration,
                )
                raise AgentBudgetExceededError(
                    f"Agent exceeded budget: ${total_cost:.4f} > ${self._config.max_cost_usd:.2f}"
                )

            # CHECK: is the LLM done?
            if choice.finish_reason == "stop":
                logger.info(
                    "agent_completed",
                    iterations=iteration,
                    tools_used=tools_used,
                    total_cost=round(total_cost, 6),
                )
                return AgentResult(
                    answer=choice.message.content or "",
                    iterations=iteration,
                    tools_used=tools_used,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    total_cost_usd=round(total_cost, 6),
                )

            # ACT: execute tool calls
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(choice.message)  # type: ignore

                for tool_call in choice.message.tool_calls:
                    if tool_call.type != "function":
                        continue
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}

                    logger.info(
                        "agent_tool_call",
                        tool=func_name,
                        args=func_args,
                        iteration=iteration,
                    )

                    handler = self._tools.get_handler(func_name)
                    if handler is None:
                        result = f"Error: unknown tool '{func_name}'"
                    else:
                        try:
                            result = await handler(**func_args)
                        except Exception as e:
                            logger.error(
                                "agent_tool_error",
                                tool=func_name,
                                error=str(e),
                            )
                            result = f"Error executing {func_name}: {str(e)}"

                    tools_used.append(func_name)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )

        # Exceeded max iterations
        logger.warning(
            "agent_max_iterations",
            iterations=iteration,
            tools_used=tools_used,
            total_cost=round(total_cost, 6),
        )
        return AgentResult(
            answer="I wasn't able to complete the task within the allowed number of steps.",
            iterations=iteration,
            tools_used=tools_used,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost_usd=round(total_cost, 6),
        )
