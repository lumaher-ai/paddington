import asyncio
import base64

import httpx
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from openai import AsyncOpenAI

from paddington.config import get_settings


async def image_input_openai() -> None:
    "Send an image URL to GTP for analysis."
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    response = await client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image? Be brief."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://images.panda.org/assets/images/pages/welcome/orangutan_1600x1000_279157.jpg",
                        },
                    },
                ],
            }
        ],
    )
    print("\n=== OpenAI Image Input ===")
    print(response.choices[0].message.content)
    if response.usage:
        print(f"Tokens: {response.usage.total_tokens}")
    else:
        print("Tokens: Usage information is not available.")


async def image_input_anthropic() -> None:
    "Send a base64 image to Claude for analysis"
    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    # Download image and convert to base64 (Anthropic requires base64, not URLs)
    async with httpx.AsyncClient() as http_client:
        img_response = await http_client.get(
            "https://images.panda.org/assets/images/pages/welcome/orangutan_1600x1000_279157.jpg"
        )
        image_data = base64.b64encode(img_response.content).decode("utf-8")

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": "What do you see in this image? Be brief."},
                ],
            }
        ],
    )
    print("\n=== Anthropic Image Input ===")
    block = response.content[0]
    if isinstance(block, TextBlock):
        print(block.text)
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")


async def main() -> None:
    await image_input_openai()
    await image_input_anthropic()


if __name__ == "__main__":
    asyncio.run(main())
