import time

from openai import AsyncOpenAI

from paddington.config import get_settings
from paddington.logging_config import get_logger

logger = get_logger(__name__)

# Pricing per 1M tokens (text-embedding-3-small)
EMBEDDING_PRICE_PER_1M = 0.02


class EmbeddingService:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = "text-embedding-3-small"

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text and return the vector."""
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call."""
        start_time = time.perf_counter()

        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        total_tokens = response.usage.total_tokens
        cost_usd = (total_tokens / 1_000_000) * EMBEDDING_PRICE_PER_1M

        logger.info(
            "embedding_completed",
            model=self._model,
            text_count=len(texts),
            total_tokens=total_tokens,
            cost_usd=round(cost_usd, 6),
            latency_ms=round(latency_ms, 1),
        )

        # Sort by index to maintain order (API doesn't guarantee order)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
