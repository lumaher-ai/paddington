import asyncio

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from paddington.config import get_settings


async def main() -> None:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions about programming. "
                "Keep responses under 3 sentences."
            ),
        },
        {
            "role": "user",
            "content": "What is the difference between a list and a tuple in Python?",
        },
    ]

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    choice = response.choices[0]
    print("=== Response ===")
    print(choice.message.content)
    print()

    print("=== Usage ===")
    if response.usage:
        print(f"Prompt tokens:     {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Total tokens:      {response.usage.total_tokens}")
    else:
        print("Usage information is not available.")
    print(f"Model:             {response.model}")

    # Continue the conversation
    messages.append({"role": "assistant", "content": choice.message.content or ""})
    messages.append({"role": "user", "content": "Give me a code example of each."})

    response2 = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    print("\n=== Follow-up Response ===")
    print(response2.choices[0].message.content)
    if response2.usage:
        print(f"Total tokens this call: {response2.usage.total_tokens}")
    else:
        print("Usage information is not available for this call.")


if __name__ == "__main__":
    asyncio.run(main())
