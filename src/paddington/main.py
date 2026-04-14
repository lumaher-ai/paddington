from fastapi import FastAPI

from paddington.config import get_settings
from paddington.routes import echo, health

settings = get_settings()

app = FastAPI(title=settings.app_name, version=settings.app_version, debug=settings.debug)

app.include_router(health.router)
app.include_router(echo.router)
