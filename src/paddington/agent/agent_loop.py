import json
from dataclasses import dataclass
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from paddington.agent.tools import PaddingtonTools
from paddington.exceptions import PaddingtonError
from paddington.llm.client import LLMClient
from paddington.logging_config import get_logger

logger = get_logger(__name__)


class AgentBudgetExceededError(PaddingtonError):
    status_code = 429


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
    max_iterations: int = 10
    max_cost_usd: float = 0.50
    system_prompt: str = (
        "You are a helpful assistant with access to tools. "
        "Use them when needed to answer the user's question accurately. "
        "If you can answer without tools, do so directly. "
        "Be concise and cite your sources when using search results."
    )


class AgentState(BaseModel):
    """State that flows through the LangGraph agent."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    iteration: int = 0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    is_budget_exceeded: bool = False


class AgentLoop:
    """Production agent loop powered by LangGraph."""

    def __init__(
        self,
        tools: PaddingtonTools,
        llm_client: LLMClient,
        config: AgentConfig | None = None,
    ) -> None:
        self._tools = tools
        self._llm = llm_client
        self._config = config or AgentConfig()
        self._graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("call_llm", self._call_llm_node)
        graph.add_node("execute_tools", self._execute_tools_node)
        graph.add_node("check_budget", self._check_budget_node)

        graph.add_edge(START, "call_llm")
        graph.add_conditional_edges(
            "call_llm",
            self._route_after_llm,
            {
                "execute_tools": "check_budget",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "check_budget",
            self._route_after_budget_check,
            {
                "continue": "execute_tools",
                "budget_exceeded": END,
            },
        )
        graph.add_edge("execute_tools", "call_llm")

        return graph.compile()

    # ─── Nodes ───

    async def _call_llm_node(self, state: AgentState) -> dict:
        """Send conversation to the LLM via LiteLLM."""
        messages_for_llm = self._convert_messages_for_litellm(state.messages)
        tool_schemas = self._tools.get_tool_schemas()

        response = await self._llm.chat_with_tools(
            messages=messages_for_llm,
            tools=tool_schemas,
            model=self._config.model,
            system=self._config.system_prompt,
            temperature=0.0,
        )

        choice = response.choices[0]

        # Track cost
        cost_increment = 0.0
        input_tokens = 0
        output_tokens = 0
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            try:
                from litellm import completion_cost

                cost_increment = completion_cost(completion_response=response)
            except Exception:
                pass

        # Convert to LangChain message format
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments),
                    }
                )

        ai_message = AIMessage(
            content=choice.message.content or "",
            tool_calls=tool_calls,
        )

        new_iteration = state.iteration + 1

        logger.info(
            "agent_iteration",
            iteration=new_iteration,
            has_tool_calls=len(tool_calls) > 0,
            cost_increment=round(cost_increment, 6),
            accumulated_cost=round(state.total_cost_usd + cost_increment, 6),
        )

        return {
            "messages": [ai_message],
            "iteration": new_iteration,
            "total_cost_usd": state.total_cost_usd + cost_increment,
            "total_input_tokens": state.total_input_tokens + input_tokens,
            "total_output_tokens": state.total_output_tokens + output_tokens,
        }

    async def _execute_tools_node(self, state: AgentState) -> dict:
        """Execute all tool calls from the last AI message."""
        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}

        tool_messages = []
        new_tools_used = []

        for tool_call in last_message.tool_calls:
            func_name = tool_call["name"]
            func_args = tool_call["args"]

            logger.info(
                "agent_tool_call",
                tool=func_name,
                args=func_args,
                iteration=state.iteration,
            )

            handler = self._tools.get_handler(func_name)
            if handler is None:
                result = f"Error: unknown tool '{func_name}'"
            else:
                try:
                    result = await handler(**func_args)
                except Exception as e:
                    logger.error("agent_tool_error", tool=func_name, error=str(e))
                    result = f"Error executing {func_name}: {str(e)}"

            tool_messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            new_tools_used.append(func_name)

        return {
            "messages": tool_messages,
            "tools_used": state.tools_used + new_tools_used,
        }

    async def _check_budget_node(self, state: AgentState) -> dict:
        """Check if the agent has exceeded its budget."""
        if state.total_cost_usd > self._config.max_cost_usd:
            logger.warning(
                "agent_budget_exceeded",
                cost=round(state.total_cost_usd, 6),
                budget=self._config.max_cost_usd,
            )
            return {"is_budget_exceeded": True}
        return {}

    # ─── Routing functions ───

    def _route_after_llm(self, state: AgentState) -> str:
        """Decide: did the LLM finish or does it want tools?"""
        if state.iteration >= self._config.max_iterations:
            logger.warning("agent_max_iterations", iterations=state.iteration)
            return "end"

        last_message = state.messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "execute_tools"

        return "end"

    def _route_after_budget_check(self, state: AgentState) -> str:
        """Decide: is the budget still OK?"""
        if state.is_budget_exceeded:
            return "budget_exceeded"
        return "continue"

    # ─── Message conversion ───

    def _convert_messages_for_litellm(self, messages: list[BaseMessage]) -> list[dict]:
        """Convert LangChain messages to LiteLLM/OpenAI format."""
        result = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                msg_dict = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(msg_dict)
            elif isinstance(msg, ToolMessage):
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
        return result

    # ─── Public API (same interface as before) ───

    async def run(self, user_message: str) -> AgentResult:
        """Execute the agent. Same interface as the manual loop."""
        initial_state = {
            "messages": [HumanMessage(content=user_message)],
        }

        final_state = await self._graph.ainvoke(initial_state)

        # Extract the answer from the last AI message
        answer = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
                answer = msg.content
                break

        if final_state.get("is_budget_exceeded"):
            raise AgentBudgetExceededError(
                f"""Agent exceeded budget: ${final_state["total_cost_usd"]:.4f} > ${
                    self._config.max_cost_usd:.2f}"""
            )

        logger.info(
            "agent_completed",
            iterations=final_state["iteration"],
            tools_used=final_state["tools_used"],
            total_cost=round(final_state["total_cost_usd"], 6),
        )

        return AgentResult(
            answer=answer,
            iterations=final_state["iteration"],
            tools_used=final_state["tools_used"],
            total_input_tokens=final_state["total_input_tokens"],
            total_output_tokens=final_state["total_output_tokens"],
            total_cost_usd=round(final_state["total_cost_usd"], 6),
        )
