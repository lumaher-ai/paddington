from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from paddington.config import get_settings
from paddington.logging_config import configure_logging, get_logger
from paddington.routes import echo, health

configure_logging()

logger = get_logger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup logic
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )

    yield

    # Shutdown logic
    logger.info("application_shutting_down", app_name=settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(echo.router)
