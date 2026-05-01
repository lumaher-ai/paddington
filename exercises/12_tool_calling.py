import asyncio
import json
from datetime import datetime, timezone

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
)

from paddington.config import get_settings

# ─── Step 1: Define your tools as regular Python functions ───


def get_current_time(timezone_name: str) -> str:
    """Get the current time in a specified timezone."""
    # Simplified: only supports UTC and America/Bogota
    if timezone_name == "America/Bogota":
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("America/Bogota"))
    else:
        now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Only allow safe math operations
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: expression contains invalid characters"
        result = eval(expression)  # Safe because we validated chars above
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    # Simulated for now — in production this would call your RAG pipeline
    fake_results = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "bogota": "Bogotá is the capital of Colombia with a population of approximately 7.4 million.",
        "paddington": "Paddington is a web agent built with FastAPI, LangGraph, and MCP.",
    }
    query_lower = query.lower()
    for key, value in fake_results.items():
        if key in query_lower:
            return value
    return "No relevant information found in the knowledge base."


# ─── Step 2: Tool registry — maps names to functions + schemas ───

TOOLS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_knowledge_base": search_knowledge_base,
}

# These schemas tell the LLM what tools exist and how to call them
TOOL_SCHEMAS: list[ChatCompletionFunctionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time in a specified timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone_name": {
                        "type": "string",
                        "description": "IANA timezone name (e.g., 'America/Bogota', 'UTC')",
                    },
                },
                "required": ["timezone_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Supports +, -, *, /, parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate (e.g., '234 * 567')",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the internal knowledge base for information about a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ─── Step 3: The agent loop ───


async def run_agent(user_message: str) -> str:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools. "
                "Use them when needed to answer the user's question accurately. "
                "If you can answer without tools, do so directly."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    max_iterations = 10  # Safety limit to prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        # OBSERVE + THINK: send messages to the LLM
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_SCHEMAS,
        )

        choice = response.choices[0]

        # CHECK: did the LLM want to call tools or respond with text?
        if choice.finish_reason == "stop":
            # The LLM is done — return the text response
            print("Agent finished with text response")
            return choice.message.content or ""

        if choice.finish_reason == "tool_calls":
            # The LLM wants to call one or more tools
            tool_calls = choice.message.tool_calls or []
            print(f"Agent requested {len(tool_calls)} tool call(s)")

            # IMPORTANT: append the assistant message WITH tool_calls to the history
            messages.append(choice.message)  # type: ignore

            # ACT: execute each tool call
            for tool_call in tool_calls:
                if tool_call.type != "function":
                    continue
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"  Calling: {func_name}({func_args})")

                # Look up and execute the function
                func = TOOLS.get(func_name)
                result = f"Error: unknown tool '{func_name}'" if func is None else func(**func_args)

                print(f"  Result: {result}")

                # APPEND: add the tool result to the messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

            # Loop back to OBSERVE — the LLM will see the tool results

    return "Error: agent exceeded maximum iterations"


# ─── Step 4: Test it ───


async def main() -> None:
    print("=" * 60)
    print("Test 1: Simple question (no tools needed)")
    print("=" * 60)
    result = await run_agent("What is 2 + 2?")
    print(f"\nFinal answer: {result}")

    print("\n" + "=" * 60)
    print("Test 2: Question that requires a tool")
    print("=" * 60)
    result = await run_agent("What time is it in Bogotá right now?")
    print(f"\nFinal answer: {result}")

    print("\n" + "=" * 60)
    print("Test 3: Question that requires MULTIPLE tools")
    print("=" * 60)
    result = await run_agent(
        "What time is it in Bogotá, and what is 234 * 567? Also, what do you know about Python?"
    )
    print(f"\nFinal answer: {result}")


if __name__ == "__main__":
    asyncio.run(main())
