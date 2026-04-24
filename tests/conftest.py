from collections.abc import AsyncIterator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from paddington.database import Base
from paddington.dependencies import get_db_session
from paddington.main import app

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def client(test_session: AsyncSession) -> Generator[TestClient, None, None]:
    async def override_get_db_session():
        yield test_session

    app.dependency_overrides[get_db_session] = override_get_db_session
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def admin_token(test_session: AsyncSession) -> str:
    from paddington.models import User
    from paddington.models.enums import UserRole
    from paddington.services.auth_service import create_access_token, hash_password

    admin = User(
        name="Admin",
        email="admin@test.local",
        hashed_password=hash_password("adminpass123"),
        role=UserRole.ADMIN.value,
    )
    test_session.add(admin)
    await test_session.commit()
    await test_session.refresh(admin)

    return create_access_token(user_id=admin.id, email=admin.email, role=admin.role)
