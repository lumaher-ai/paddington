import asyncio
import time

import httpx


async def fetch_post(post_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://jsonplaceholder.typicode.com/posts/{post_id}")
        response.raise_for_status()
        return response.json()


async def fetch_posts_sequentially(post_ids: list[int]) -> list[dict]:
    results = []
    for post_id in post_ids:
        post = await fetch_post(post_id)
        results.append(post)
    return results


async def fetch_posts_concurrently(post_ids: list[int]) -> list[dict]:
    return await asyncio.gather(*(fetch_post(pid) for pid in post_ids))


if __name__ == "__main__":
    post_ids = [1, 2, 3, 4, 5]

    async def main():
        start = time.perf_counter()
        seq_results = await fetch_posts_sequentially(post_ids)
        seq_elapsed = time.perf_counter() - start
        print(f"Sequential: {seq_elapsed:.2f}s ({len(seq_results)} posts)")

        start = time.perf_counter()
        conc_results = await fetch_posts_concurrently(post_ids)
        conc_elapsed = time.perf_counter() - start
        print(f"Concurrent: {conc_elapsed:.2f}s ({len(conc_results)} posts)")

        print(f"Speedup: {seq_elapsed / conc_elapsed:.1f}x")

    asyncio.run(main())
