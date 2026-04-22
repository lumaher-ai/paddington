from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from paddington.config import get_settings
from paddington.database import close_db, init_db
from paddington.exception_handlers import register_exception_handlers
from paddington.logging_config import configure_logging, get_logger
from paddington.routes import auth, echo, health, users

configure_logging()

logger = get_logger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    logger.info("application_started")
    yield
    await close_db()
    logger.info("application_stopped")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

register_exception_handlers(app)

app.include_router(health.router)
app.include_router(echo.router)
app.include_router(users.router)
app.include_router(auth.router)
