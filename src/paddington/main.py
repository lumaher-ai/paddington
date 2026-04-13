from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="paddington", version="0.1.0")


class HealthResponse(BaseModel):
    status: str
    version: str


class EchoRequest(BaseModel):
    message: str
    repeat: int = 1


class EchoResponse(BaseModel):
    original: str
    echoed: list[str]
    received_at: datetime


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")


@app.post("/echo", response_model=EchoResponse)
async def echo(request: EchoRequest) -> EchoResponse:
    return EchoResponse(
        original=request.message,
        echoed=[request.message] * request.repeat,
        received_at=datetime.now(timezone.utc),
    )
