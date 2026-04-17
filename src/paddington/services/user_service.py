from paddington.models import User
from paddington.repositories.user_repository import UserAlreadyExistsError, UserRepository
from paddington.schemas.user import UserCreate


class UserService:
    def __init__(self, repository: UserRepository) -> None:
        self._repository = repository

    async def create_user(self, user_data: UserCreate) -> User:
        return await self._repository.create(
            name=user_data.name,
            email=user_data.email,
        )


# Re-export for convenience in routes
__all__ = ["UserService", "UserAlreadyExistsError"]
