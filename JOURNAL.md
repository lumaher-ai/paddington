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

## Day 3 — [04/14/2026]

**Done:**
- Refactored FastAPI app into routes/schemas/services layers
- Added Pydantic Settings with .env file support
- Implemented dependency injection pattern for config
- Set up structlog with dev (colored) and prod (JSON) modes
- Added Field validations (min/max length, value ranges)
- Added tests for validation edge cases

**Learned:**
- Layered architecture separates "what data looks like" (schemas), "what the system does" (services), and "how HTTP maps to it" (routes)
- Pydantic Settings applies the same "validated typed data" pattern to configuration as Pydantic does to API data
- `@lru_cache` is the standard way to make settings behave as a singleton
- Structured logs are objects with fields, not strings — they make filtering in production observavility tools possible

**Things I've seen but don't deeply understand yet (and that's OK):**
- FastAPI's dependency injection internals (just using it at surface level)
- structlog's processor chain (trusting the setup, will revisit when I hit a problem)
- the asynccontextmanager decorator in the lifespan, and the yield keword(works for now, will learn more when I need custom lifecycle)

**Tomorrow:**
- Day 4: preparation for week 2 — Postgres with Docker, SQLAlchemy 2.0 async, Alembic migrations

## Day 4 — [04/15/2026]

**Done:**
- Postgres 16 running via Docker Compose with healthcheck and persistent volume
- asyncpg connection tested with raw SQL
- SQLAlchemy 2.0 async setup: engine, session factory, declarative Base
- First model: User with UUID PK, unique email
- Alembic initialized with async template; first migration generated and applied
- Repository pattern with UserRepository encapsulating DB access
- UserService wrapping the repository
- POST /users endpoint with full DI chain: route → service → repository → session
- Tests with SQLite in-memory and dependency_overrides
- 8 tests total passing (5 from before + 3 new)

**Learned:**
- Docker Compose isolates Postgres from my system; tear-down is `docker compose down -v`
- SQLAlchemy has 2 layers: Core (SQL builder) and ORM (class-to-table mapping); I'm using ORM
- `Mapped[type]` + `mapped_column(...)` is the modern 2.0 syntax; replaces old `Column(...)` style
- Alembic doesn't detect models unless you explicitly import them in env.py
- `session.flush()` triggers INSERT to detect IntegrityError early; commit closes the transaction
- Repository pattern keeps services agnostic of SQLAlchemy specifics
- `dependency_overrides` in FastAPI tests is what makes DI worth the ceremony

**Things I've seen but don't deeply understand yet (and that's OK):**
- SQLAlchemy session lifecycle in detail (when exactly does flush vs commit happen)
- Connection pool tuning (just using defaults)
- Alembic migration conflict resolution (haven't hit it yet)

**Tomorrow:**
- Day 5: relationships between models, more endpoints (GET /users, GET /users/{id})

## Day 5 — [04/16/2026]

**Done:**
- Added updated_at field with onupdate to User model + migration
- Completed CRUD: GET list (paginated), GET by id, PATCH (partial), DELETE
- Built centralized exception handling with PaddingtonError hierarchy
- Single handler catches all domain exceptions automatically
- 12+ tests covering all endpoints, happy and sad paths
- Swagger UI verified end-to-end

**Learned:**
- model_dump(exclude_unset=True) distinguishes "field not sent" from "field sent as null" — critical for PATCH
- Query(ge=1, le=100) validates query params declaratively, same pattern as Field for body
- Centralized exception handlers eliminate try/except duplication in routes
- Base exception with status_code attribute lets one handler manage all domain errors
- onupdate in mapped_column auto-updates timestamps on every UPDATE

**Things I've seen but don't deeply understand yet (and that's OK):**
- SQLAlchemy session internals: flush vs commit timing in nested operations
- How select().order_by().limit().offset() translates to actual SQL
- Alembic migration conflict resolution (still haven't hit it)

**Tomorrow:**
- Day 6: Auth — hashing, JWT, Bearer tokens, login/signup, protected endpoints

## Day 6 — [04/17/2026]

**Done:**
- Studied security theory: encoding, encryption, hashing (bcrypt, scrypt, argon2), and JWT principle of integrity, not confidentiality
- Implemented argon2 password hashing with passlib
- Built POST /auth/signup and POST /auth/login endpoints
- Generated JWTs with python-jose including sub, email, exp, iat claims
- Built get_current_user dependency using HTTPBearer
- Protected GET /users/me with Bearer token
- Moved shared test fixtures to conftest.py
- 23 tests passing

**Key security concepts I can now explain:**
- Why bcrypt and not SHA-256 for passwords (intentionally slow prevents brute force)
- Why JWT payload is readable but secure (signature prevents tampering)
- Why login error messages should be identical for wrong email vs wrong password
- What "Bearer" means in Authorization header (RFC 6750, "whoever bears this token")

**Tomorrow:**
- Day 7: RBAC (roles), refresh tokens, and OAuth conceptual overview

## Day 7 — [04/20/2026]

**Done:**
- Added UserRole enum (user, admin) and role field to User model
- Built require_role dependency factory using closures
- Protected DELETE and list endpoints with admin-only access
- Built PATCH /users/{id}/role for admin promotion
- Created CLI tool for seeding first admin
- Implemented refresh tokens with DB-backed rotation
- POST /auth/refresh revokes old token and issues new pair
- Studied OAuth 2.0 Authorization Code flow conceptually
- Updated all tests for new auth requirements

**Key concepts I can now explain:**
- RBAC: assign permissions to roles, assign roles to users, check roles not individual permissions
- Why require_role returns a function (closure pattern for parameterized dependencies)
- Why refresh tokens exist (short-lived access for security + long-lived refresh for UX)
- Token rotation: revoke on use, detect stolen tokens when legitimate user gets 401
- OAuth 2.0 Authorization Code flow: why the code intermediate step exists (separates browser-visible redirect from server-to-server token exchange)

**Tomorrow:**
- Day 8: week 2 wrap-up, code cleanup, full test suite verification, plan week 3 (LLM APIs + RAG)