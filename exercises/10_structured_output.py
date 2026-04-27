import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from paddington.config import get_settings


# Define the schema of what you want the LLM to return
class ProductInfo(BaseModel):
    name: str = Field(description="Product name")
    price_usd: float = Field(description="Price in US dollars")
    pros: list[str] = Field(description="List of advantages", max_length=3)
    cons: list[str] = Field(description="List of disadvantages", max_length=3)
    rating: float = Field(description="Rating from 1.0 to 5.0", ge=1.0, le=5.0)


async def structured_with_openai() -> None:
    """OpenAI has native structured output support with response_format."""
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    response = await client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": "Extract product information from the user's description.",
            },
            {
                "role": "user",
                "content": (
                    "The MacBook Pro M4 costs $1,599. It has amazing performance and "
                    "a beautiful display, but it's expensive and heavy. I'd rate it 4.5/5."
                ),
            },
        ],
        response_format=ProductInfo,
    )

    product_info = response.choices[0].message.parsed
    print("=== OpenAI Structured Output ===")
    if product_info is not None:
        print(f"Type: {type(product_info)}")  # <class 'ProductInfo'> — already Pydantic!
        print(product_info.model_dump_json(indent=2))
    else:
        print("Failed to parse product information.")


async def structured_with_anthropic() -> None:
    """Anthropic uses tool_use to achieve structured output."""
    from anthropic import AsyncAnthropic

    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    # Convert Pydantic schema to JSON Schema for Anthropic
    tool_schema = ProductInfo.model_json_schema()

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    "The MacBook Pro M4 costs $1,599. It has amazing performance and "
                    "a beautiful display, but it's expensive and heavy. I'd rate it 4.5/5."
                ),
            },
        ],
        tools=[
            {
                "name": "extract_product_info",
                "description": "Extract structured product information from text.",
                "input_schema": tool_schema,
            },
        ],
        tool_choice={"type": "tool", "name": "extract_product_info"},
    )

    # Find the tool_use block in the response
    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_use_block is None:
        raise ValueError("Claude did not return a tool_use block")

    # Validate with Pydantic
    product = ProductInfo.model_validate(tool_use_block.input)
    print("\n=== Anthropic Structured Output ===")
    print(f"Type: {type(product)}")
    print(product.model_dump_json(indent=2))


async def main() -> None:
    await structured_with_openai()
    await structured_with_anthropic()


if __name__ == "__main__":
    asyncio.run(main())
