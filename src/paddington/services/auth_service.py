from datetime import datetime, timedelta, timezone
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext

from paddington.config import get_settings
from paddington.exceptions import PaddingtonError

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


class InvalidCredentialsError(PaddingtonError):
    status_code = 401


class InvalidTokenError(PaddingtonError):
    status_code = 401


def hash_password(pasword: str) -> str:
    return pwd_context.hash(pasword)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: UUID, email: str) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)

    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }

    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return payload
    except JWTError as e:
        raise InvalidTokenError("Ivalid or expired token") from e
