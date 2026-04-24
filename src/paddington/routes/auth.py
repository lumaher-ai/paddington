from fastapi import APIRouter, Depends

from paddington.dependencies import get_refresh_token_repository, get_user_service
from paddington.repositories.refresh_token_repository import RefreshTokenRepository
from paddington.schemas.auth import LoginRequest, RefreshRequest, SignupRequest, TokenResponse
from paddington.schemas.user import UserResponse
from paddington.services.auth_service import create_access_token
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
    refresh_repo: RefreshTokenRepository = Depends(get_refresh_token_repository),
) -> TokenResponse:
    user = await service.authenticate(email=data.email, password=data.password)
    access_token = create_access_token(user_id=user.id, email=user.email, role=user.role)
    refresh_token = await refresh_repo.create(user_id=user.id)
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token.token,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    data: RefreshRequest,
    refresh_repo: RefreshTokenRepository = Depends(get_refresh_token_repository),
    service: UserService = Depends(get_user_service),
) -> TokenResponse:
    # Validate and revoke the old refresh token
    old_token = await refresh_repo.get_valid_token(data.refresh_token)
    await refresh_repo.revoke(old_token)

    # Get the user and generate new tokens
    user = await service.get_user(old_token.user_id)
    new_access_token = create_access_token(user_id=user.id, email=user.email, role=user.role)
    new_refresh_token = await refresh_repo.create(user_id=user.id)

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token.token,
    )
