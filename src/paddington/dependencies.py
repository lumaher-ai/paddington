from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from paddington.database import get_session
from paddington.models import User
from paddington.repositories.user_repository import UserRepository
from paddington.services.auth_service import InvalidTokenError, decode_access_token
from paddington.services.user_service import UserService

security_scheme = HTTPBearer()


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with get_session() as session:
        yield session


def get_user_repository(
    session: AsyncSession = Depends(get_db_session),
) -> UserRepository:
    return UserRepository(session)


def get_user_service(
    repository: UserRepository = Depends(get_user_repository),
) -> UserService:
    return UserService(repository)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    service: UserService = Depends(get_user_service),
) -> User:
    payload = decode_access_token(credentials.credentials)

    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise InvalidTokenError("Token missing 'sub' claim")

    try:
        user_id = UUID(user_id_str)
    except ValueError:
        raise InvalidTokenError("Token 'sub' is not a valid UUID")

    return await service.get_user(user_id)
