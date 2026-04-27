import time
from dataclasses import dataclass

import tiktoken
from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import AsyncAnthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import (
    APIConnectionError as OpenAIConnectionError,
)
from openai import (
    AsyncOpenAI,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from paddington.config import get_settings
from paddington.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    "Unified response from any LLM provider."

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    provider: str


# Pricing per 1M tokens
# Last updated: 2026-04-22
PRICING = {
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 1.25, "output": 5.00},
}


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD based on token counts and model pricing."""
    pricing = PRICING.get(model)
    if pricing is None:
        logger.warning("unknown_model_pricing", model=model)
        return 0.0
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 3)


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens for a string using tiktoken (OpenAI models only)."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # For non-OpenAI models, approximate: 1 token ≈ 4 chars
        return len(text) // 4


class LLMClient:
    """Unified client for OpenAI and Anthropic with error handling, retries, and cost tracking."""

    def __init__(self) -> None:
        settings = get_settings()
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self._anthropic = AsyncAnthropic(api_key=settings.anthropic_api_key)

        @retry(
            retry=retry_if_exception_type(
                (
                    OpenAIRateLimitError,
                    OpenAIConnectionError,
                    AnthropicRateLimitError,
                    AnthropicConnectionError,
                )
            ),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(3),
            before_sleep=lambda retry_state: logger.warning(
                "llm_retry",
                attempt=retry_state.attempt_number,
                wait=getattr(retry_state.next_action, "sleep", None),
                error=str(
                    retry_state.outcome.exception() if retry_state.outcome else "Unknown error"
                ),
            ),
        )
        async def chat(
            self,
            messages: list[dict],
            model: str = "gpt-4o-mini",
            system: str | None = None,
            max_tokens: int = 1024,
            temperature: float = 0.0,
        ) -> LLMResponse:
            start_time = time.perf_counter()

            if model.startswith("gpt") or model.startswith("o"):
                response = await self._call_openai(
                    messages=messages,
                    model=model,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            elif model.startswith("claude"):
                response = await self._call_anthropic(
                    messages=messages,
                    model=model,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                raise ValueError(
                    f"Unknown model: {model}. Must start with 'gpt', 'o', or 'claude'."
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            result = LLMResponse(
                content=response["content"],
                model=model,
                input_tokens=response["input_tokens"],
                output_tokens=response["output_tokens"],
                total_tokens=response["input_tokens"] + response["output_tokens"],
                cost_usd=_calculate_cost(
                    model, response["input_tokens"], response["output_tokens"]
                ),
                latency_ms=round(latency_ms, 1),
                provider=response["provider"],
            )

            logger.info(
                "llm_call_completed",
                model=result.model,
                provider=result.provider,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
            )

            return result

        async def _call_openai(
            self,
            messages: list[dict],
            model: str,
            system: str | None,
            max_tokens: int,
            temperature: float,
        ) -> dict:
            all_messages = []
            if system:
                all_messages.append({"role": "system", "content": system})
            all_messages.extend(messages)

            response = await self._openai.chat.completions.create(
                model=model,
                messages=all_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return {
                "content": response.choices[0].message.content or "",
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "provider": "openai",
            }

        async def _call_anthropic(
            self,
            messages: list[dict],
            model: str,
            system: str | None,
            max_tokens: int,
            temperature: float,
        ) -> dict:
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system:
                kwargs["system"] = system

            response = await self._anthropic.messages.create(**kwargs)

            return {
                "content": response.content[0].text if response.content else "",
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "provider": "anthropic",
            }
