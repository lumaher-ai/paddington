import asyncio
import time


# hello_async()
async def hello_async() -> None:
    print("starting")
    await asyncio.sleep(1)
    print("done")


# fetch_three_things_sequentially()
async def fetch_three_things_sequentially() -> None:
    start = time.perf_counter()
    await asyncio.sleep(1)
    await asyncio.sleep(1)
    await asyncio.sleep(1)
    elapsed = time.perf_counter() - start
    print(f"Sequential took {elapsed:.2f}s")


# fetch_three_things_concurrently()
async def fetch_three_things_concurrently() -> None:
    start = time.perf_counter()
    await asyncio.gather(
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
    )
    elapsed = time.perf_counter() - start
    print(f"Concurrent took {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(hello_async())
    asyncio.run(fetch_three_things_sequentially())
    asyncio.run(fetch_three_things_concurrently())
