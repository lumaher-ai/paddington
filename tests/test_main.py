from fastapi.testclient import TestClient

from src.paddington.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


def test_echo_repeats_message():
    response = client.post("/echo", json={"message": "hello", "repeat": 3})
    assert response.status_code == 200

    data = response.json()
    assert data["original"] == "hello"
    assert data["echoed"] == ["hello", "hello", "hello"]
    assert "received_at" in data


def test_echo_rejects_invalid_repeat():
    response = client.post("/echo", json={"message": "hi", "repeat": "not a number"})
    assert response.status_code == 422  # Pydantic validation error
