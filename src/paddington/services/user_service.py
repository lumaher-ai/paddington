from uuid import UUID

from paddington.models import User
from paddington.repositories.user_repository import (
    UserAlreadyExistsError,
    UserNotFoundError,
    UserRepository,
)
from paddington.schemas.user import UserCreate, UserUpdate


class UserService:
    def __init__(self, repository: UserRepository) -> None:
        self._repository = repository

    async def create_user(self, data: UserCreate) -> User:
        return await self._repository.create(
            name=data.name,
            email=data.email,
        )

    async def get_user(self, user_id: UUID) -> User:
        return await self._repository.get_by_id(user_id)

    async def list_users(self, limit: int = 20, offset: int = 0) -> tuple[list[User], int]:
        users = await self._repository.list_all(limit=limit, offset=offset)
        total = await self._repository.count()
        return users, total

    async def update_user(self, user_id: UUID, data: UserUpdate) -> User:
        update_data = data.model_dump(exclude_unset=True)
        if not update_data:
            return await self._repository.get_by_id(user_id)
        return await self._repository.update(user_id, **update_data)

    async def delete_user(self, user_id: UUID) -> None:
        await self._repository.delete(user_id)


__all__ = ["UserService", "UserAlreadyExistsError", "UserNotFoundError"]
