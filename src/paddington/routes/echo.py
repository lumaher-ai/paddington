from fastapi import APIRouter

from paddington.schemas.echo import EchoRequest, EchoResponse
from paddington.services.echo_service import create_echo

router = APIRouter(tags=["echo"])


@router.post("/echo", response_model=EchoResponse)
async def echo(request: EchoRequest) -> EchoResponse:
    return create_echo(request)
