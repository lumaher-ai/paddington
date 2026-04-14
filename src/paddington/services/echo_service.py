from datetime import datetime, timezone

from paddington.schemas.echo import EchoRequest, EchoResponse


def create_echo(request: EchoRequest) -> EchoResponse:
    return EchoResponse(
        original=request.message,
        echoed=[request.message] * request.repeat,
        received_at=datetime.now(timezone.utc),
    )
