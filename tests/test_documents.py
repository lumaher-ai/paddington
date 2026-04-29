from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from paddington.dependencies import get_embedding_service, get_llm_client
from paddington.llm.client import LLMClient, LLMResponse
from paddington.llm.embedding_service import EmbeddingService
from paddington.main import app


def _mock_embedding_service() -> EmbeddingService:
    mock = AsyncMock(spec=EmbeddingService)
    # Return a deterministic fake embedding (1536 dimensions)
    mock.embed_text.return_value = [0.1] * 1536
    mock.embed_batch.return_value = [[0.1] * 1536, [0.1] * 1536, [0.1] * 1536]
    return mock


def _mock_llm_client() -> LLMClient:
    mock = AsyncMock(spec=LLMClient)
    mock.chat.return_value = LLMResponse(
        content="Based on the context, the answer is 42.",
        model="gpt-4o-mini",
        input_tokens=500,
        output_tokens=20,
        total_tokens=520,
        cost_usd=0.0001,
        latency_ms=300.0,
        provider="openai",
    )
    return mock


def test_upload_document_returns_201(client: TestClient) -> None:
    mock_emb = _mock_embedding_service()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb

    # Signup + login
    client.post(
        "/auth/signup",
        json={"name": "Doc User", "email": "doc@example.com", "password": "securepass123"},
    )
    login = client.post(
        "/auth/login",
        json={"email": "doc@example.com", "password": "securepass123"},
    )
    token = login.json()["access_token"]

    # Upload document
    response = client.post(
        "/documents",
        json={
            "title": "Test Document",
            "content": "This is a test document with enough content to be chunked. " * 50,
        },
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Test Document"
    assert data["chunk_count"] > 0

    app.dependency_overrides.pop(get_embedding_service, None)


def test_query_returns_answer_with_sources(client: TestClient) -> None:
    mock_emb = _mock_embedding_service()
    mock_llm = _mock_llm_client()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    app.dependency_overrides[get_llm_client] = lambda: mock_llm

    # Signup + login
    client.post(
        "/auth/signup",
        json={"name": "Query User", "email": "query@example.com", "password": "securepass123"},
    )
    login = client.post(
        "/auth/login",
        json={"email": "query@example.com", "password": "securepass123"},
    )
    token = login.json()["access_token"]

    # Upload document first
    client.post(
        "/documents",
        json={
            "title": "Knowledge Base",
            "content": "Python is a programming language. " * 100,
        },
        headers={"Authorization": f"Bearer {token}"},
    )

    # Query
    response = client.post(
        "/documents/query",
        json={"question": "What is Python?", "top_k": 3},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
    assert data["model"] == "gpt-4o-mini"

    app.dependency_overrides.pop(get_embedding_service, None)
    app.dependency_overrides.pop(get_llm_client, None)


def test_query_requires_auth(client: TestClient) -> None:
    response = client.post(
        "/documents/query",
        json={"question": "test"},
    )
    assert response.status_code == 401


def test_upload_requires_auth(client: TestClient) -> None:
    response = client.post(
        "/documents",
        json={"title": "Test", "content": "test content here with enough length"},
    )
    assert response.status_code == 401
