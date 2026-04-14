## Day 2 — [04/13/2026]

**Done:**
- async/await fundamentals + asyncio.gather
- httpx async client with JSONPlaceholder
- First FastAPI app with /health and /echo endpoints validated with Pydantic
- pytest setup + 3 tests covering happy and sad paths
- Branch workflow adopted + PR

**Learned:**
- asyncio.gather is Promise.all
- sequential vs concurrent makes a 5x differece visibly
- FastAPI generates Swagger UI from Pydantic models automatically
- Defining the Pydantic model once generates validation, documentation, serialization, and type safety in the code that consumes the model. This is dramatically more productive than the Express + Joi + swagger-jsdoc approach.
- FastAPI's TestClient lets you make HTTP requests to your app without starting a real server. It's synchronous internally withou matter if my app is async — FastAPI handles the translation. Also it's incredibly fast.
- pytest assert-based tests have introspections which makes errors clearer


## Things I've seen but don't deeply understand yet (and that's OK)
- typing.Optional / typing.Union — know they're for "X or None", will learn more when I see them in real code
- typing.Generic, TypeVar — saw them mentioned, will revisit when I write something that needs them
- async/await internals — using them at surface level, will go deep when I hit a real problem
- __name__ == "__main__" - this is something important when using sample data and other files that import content must ignore the case use with data and only access to the abstraction, I will be more relevant when creating tests, but I do not how this is used when importing files and why this works
- Some modern conventions like ":.2f" used in the time.perf_counter() - start
- Why the design pattern decorater is extensively used in Pyton? ( decorators to define endpoints,  decorator @router.post(), @app.exception_handler(MyCustomError), lru_cache)
