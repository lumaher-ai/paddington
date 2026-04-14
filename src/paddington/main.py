from fastapi import FastAPI

from paddington.routes import echo, health

app = FastAPI(title="paddington", version="0.1.0")

app.include_router(health.router)
app.include_router(echo.router)
