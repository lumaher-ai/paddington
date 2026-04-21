from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from paddington.exceptions import PaddingtonError
from paddington.logging_config import get_logger

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(PaddingtonError)
    async def paddington_error_handler(request: Request, exc: PaddingtonError) -> JSONResponse:
        logger.warning(
            "domain_error",
            error_type=type(exc).__name__,
            detail=exc.detail,
            status_code=exc.status_code,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "error_type": type(exc).__name__},
        )
