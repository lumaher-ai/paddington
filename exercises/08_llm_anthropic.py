import asyncio

from anthropic import AsyncAnthropic

from paddington.config import get_settings


async def main() -> None:
    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=(
            "You are a helpful assistant that answers questions about programming. "
            "Keep responses under 3 sentences."
        ),
        messages=[
            {
                "role": "user",
                "content": "What is the difference between a list and a tuple in Python?",
            },
        ],
    )

    print("=== Response ===")
    print(response.content[0])
    print()

    print("=== Usage ===")
    print(f"Input tokens:  {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")
    print(f"Model:         {response.model}")

    # Multi-turn
    response2 = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="You are a helpful assistant. Keep responses under 3 sentences.",
        messages=[
            {
                "role": "user",
                "content": "What is the difference between a list and a tuple in Python?",
            },
            {"role": "assistant", "content": str(response.content[0])},
            {"role": "user", "content": "Give me a code example of each."},
        ],
    )

    print("\n=== Follow-up Response ===")
    print(response2.content[0])
    print(f"Input tokens: {response2.usage.input_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
