from fastapi import APIRouter, Depends

from paddington.config import Settings, get_settings
from paddington.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", version=settings.app_version)
