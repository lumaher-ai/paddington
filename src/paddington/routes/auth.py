from fastapi import APIRouter, Depends

from paddington.dependencies import get_user_service
from paddington.schemas.auth import LoginRequest, SignupRequest, TokenResponse
from paddington.schemas.user import UserResponse
from paddington.services.auth_service import (
    create_access_token,
)
from paddington.services.user_service import UserService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=UserResponse, status_code=201)
async def signup(
    data: SignupRequest,
    service: UserService = Depends(get_user_service),
) -> UserResponse:
    user = await service.create_user_with_password(
        name=data.name,
        email=data.email,
        password=data.password,
    )
    return UserResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    data: LoginRequest,
    service: UserService = Depends(get_user_service),
) -> TokenResponse:
    user = await service.authenticate(
        email=data.email,
        password=data.password,
    )
    token = create_access_token(user_id=user.id, email=user.email, role=user.role)
    return TokenResponse(access_token=token)
