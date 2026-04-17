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


def test_create_user_returns_201(client: TestClient) -> None:
    response = client.post(
        "/users",
        json={"name": "Test User", "email": "test@example.com"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test User"
    assert data["email"] == "test@example.com"
    assert "id" in data
    assert "created_at" in data


def test_create_user_duplicate_email_returns_409(client: TestClient) -> None:
    client.post("/users", json={"name": "First", "email": "dup@example.com"})
    response = client.post(
        "/users",
        json={"name": "Second", "email": "dup@example.com"},
    )
    assert response.status_code == 409


def test_create_user_invalid_email_returns_422(client: TestClient) -> None:
    response = client.post(
        "/users",
        json={"name": "Test", "email": "not-an-email"},
    )
    assert response.status_code == 422
