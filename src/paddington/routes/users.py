from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from paddington.dependencies import get_current_user, get_user_service
from paddington.models import User
from paddington.schemas.user import (
    UserCreate,
    UserListResponse,
    UserResponse,
    UserUpdate,
)
from paddington.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["users"])


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_user(
    data: UserCreate,
    service: UserService = Depends(get_user_service),
) -> UserResponse:
    user = await service.create_user(data)
    return UserResponse.model_validate(user)


@router.get("", response_model=UserListResponse)
async def list_users(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    service: UserService = Depends(get_user_service),
) -> UserListResponse:
    users, total = await service.list_users(limit=limit, offset=offset)
    return UserListResponse(
        users=[UserResponse.model_validate(u) for u in users],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    return UserResponse.model_validate(current_user)


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    service: UserService = Depends(get_user_service),
) -> UserResponse:
    user = await service.get_user(user_id)
    return UserResponse.model_validate(user)


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    data: UserUpdate,
    service: UserService = Depends(get_user_service),
) -> UserResponse:
    user = await service.update_user(user_id, data)
    return UserResponse.model_validate(user)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: UUID,
    service: UserService = Depends(get_user_service),
) -> None:
    await service.delete_user(user_id)
