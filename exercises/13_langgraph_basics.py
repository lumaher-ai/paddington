import asyncio
import json
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from paddington.llm.client import LLMClient


# Step 1: Define the State
class AgentState(BaseModel):
    """Everything the agent needs to remember between steps."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    iteration: int = 0
    is_done: bool = False


# Step 2: Define the tools

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression (e.g, '234 * 567')",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in a timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone_name": {
                        "type": "string",
                        "description": "IANA timezone (e.g., 'America/Bogota')",
                    },
                },
                "required": ["timezone_name"],
            },
        },
    },
]

TOOL_HANDLERS = {
    "calculate": lambda expression: str(eval(expression)),
    "get_current_time": lambda timezone_name: (
        __import__("datetime")
        .datetime.now(__import__("zoneinfo").ZoneInfo(timezone_name))
        .strftime("%Y-%m-%d %H:%M:%S %Z")
    ),
}

# Step 3: Define the Nodes

llm_client = LLMClient()


async def call_llm(state: AgentState) -> dict:
    """Node: send the conversation to the LLM"""
    # Convert LangChain message to dict from LiteLLM
    messages_for_llm = []
    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            messages_for_llm.append({"role": "user", "content": msg.content})
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
            messages_for_llm.append(msg_dict)
        elif isinstance(msg, ToolMessage):
            messages_for_llm.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                }
            )
    response = await llm_client.chat_with_tools(
        messages=messages_for_llm,
        tools=TOOL_SCHEMAS,
        system="You are a helpful assistant. Use tools when needed.",
    )

    choice = response.choices[0]

    # Convert the response back to a LangChain AIMessage
    tool_calls = []

    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            tool_calls.append(
                {"id": tc.id, "name": tc.function.name, "args": json.loads(tc.function.arguments)}
            )
    ai_message = AIMessage(
        content=choice.message.content or "",
        tool_calls=tool_calls,
    )

    return {
        "messages": [ai_message],
        "iteration": state.iteration + 1,
    }


async def execute_tools(state: AgentState) -> dict:
    """Node: execute all tool calls from the last AI message."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    tool_messages = []
    tools_used = []

    for tool_call in last_message.tool_calls:
        func_name = tool_call["name"]
        func_args = tool_call["args"]

        handler = TOOL_HANDLERS.get(func_name)
        if handler is None:
            result = f"Error: unknown tool '{func_name}'"
        else:
            try:
                result = handler(**func_args)
            except Exception as e:
                result = f"Error: {e}"

        tool_messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
        tools_used.append(func_name)

    return {
        "messages": tool_messages,
        "tools_used": state.tools_used + tools_used,
    }


# ─── Step 4: Define the Conditional Edge ───
def should_continue(state: AgentState) -> str:
    """Decide whether to execute tools or finish."""
    last_message = state.messages[-1]

    # Safety: max iterations
    if state.iteration >= 4:
        return "end"

    # If the last message has tool calls, execute them
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "execute_tools"

    # Otherwise, the LLM is done
    return "end"


# Step 5: Build the Graph


def build_agent_graph() -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("call_llm", call_llm)
    graph.add_node("execute_tools", execute_tools)

    # Add edges
    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges(
        "call_llm",
        should_continue,
        {
            "execute_tools": "execute_tools",
            "end": END,
        },
    )
    graph.add_edge("execute_tools", "call_llm")

    return graph.compile()


# ─── Step 6: Run it ───


async def main() -> None:
    agent = build_agent_graph()

    print("=" * 60)
    print("Test 1: Simple question (no tools)")
    print("=" * 60)
    result = await agent.ainvoke(
        {
            "messages": [HumanMessage(content="What is 2 + 2?")],
        }
    )
    print(f"Answer: {result['messages'][-1].content}")
    print(f"Iterations: {result['iteration']}")

    print("\n" + "=" * 60)
    print("Test 2: Question requiring tools")
    print("=" * 60)
    result = await agent.ainvoke(
        {
            "messages": [HumanMessage(content="What is 234 * 567 and what time is it in Bogotá?")],
        }
    )
    print(f"Answer: {result['messages'][-1].content}")
    print(f"Iterations: {result['iteration']}")
    print(f"Tools used: {result['tools_used']}")

    # Print the full message history to see the flow
    print("\n--- Full message flow ---")
    for msg in result["messages"]:
        role = type(msg).__name__
        content = msg.content[:100] if msg.content else "(no content)"
        print(f"  {role}: {content}")


if __name__ == "__main__":
    asyncio.run(main())
