import asyncio

from openai import AsyncOpenAI

from paddington.config import get_settings


async def stream_openai() -> None:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    print("=== OpenAI Streaming ===")

    stream = await client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "user", "content": "Explain what Docker is in 5 sentences."},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )

    full_response = ""
    usage = None
    async for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
            continue
        delta = chunk.choices[0].delta
        if delta.content is not None:
            print(delta.content, end="", flush=True)
            full_response += delta.content

    print("\n")
    print(f"Full response length: {len(full_response)} chars")
    print("\n")
    print("=== Usage ===")
    if usage:
        print(f"Prompt tokens:     {usage.prompt_tokens}")
        print(f"Completion tokens: {usage.completion_tokens}")
        print(f"Total tokens:      {usage.total_tokens}")
    else:
        print("Usage information is not available.")


async def stream_anthropic() -> None:
    from anthropic import AsyncAnthropic

    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    print("=== Anthropic Streaming ===")

    async with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Explain what Docker is in 5 sentences."},
        ],
    ) as stream:
        full_response = ""
        async for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response += text

    print("\n")
    print(f"Full response length: {len(full_response)} chars")

    # After stream closes, you can get usage
    message = await stream.get_final_message()
    print(f"Input tokens: {message.usage.input_tokens}")
    print(f"Output tokens: {message.usage.output_tokens}")


if __name__ == "__main__":
    asyncio.run(stream_openai())
    print("---")
    asyncio.run(stream_anthropic())
