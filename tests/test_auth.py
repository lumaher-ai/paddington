from fastapi.testclient import TestClient


def test_signup_returns_201(client: TestClient) -> None:
    response = client.post(
        "/auth/signup",
        json={
            "name": "Auth User",
            "email": "auth@example.com",
            "password": "securepass123",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "auth@example.com"
    assert "hashed_password" not in data  # Never expose password hash


def test_signup_short_password_returns_422(client: TestClient) -> None:
    response = client.post(
        "/auth/signup",
        json={"name": "Test", "email": "short@example.com", "password": "123"},
    )
    assert response.status_code == 422


def test_login_returns_token(client: TestClient) -> None:
    # First signup
    client.post(
        "/auth/signup",
        json={
            "name": "Login User",
            "email": "login@example.com",
            "password": "securepass123",
        },
    )

    # Then login
    response = client.post(
        "/auth/login",
        json={"email": "login@example.com", "password": "securepass123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password_returns_401(client: TestClient) -> None:
    client.post(
        "/auth/signup",
        json={
            "name": "Wrong Pass",
            "email": "wrong@example.com",
            "password": "securepass123",
        },
    )
    response = client.post(
        "/auth/login",
        json={"email": "wrong@example.com", "password": "wrongpassword"},
    )
    assert response.status_code == 401


def test_get_me_returns_current_user(client: TestClient) -> None:
    # Signup + login
    client.post(
        "/auth/signup",
        json={
            "name": "Me User",
            "email": "me@example.com",
            "password": "securepass123",
        },
    )
    login_response = client.post(
        "/auth/login",
        json={"email": "me@example.com", "password": "securepass123"},
    )
    token = login_response.json()["access_token"]

    # Access protected endpoint
    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["email"] == "me@example.com"


def test_get_me_without_token_returns_401(client: TestClient) -> None:
    response = client.get("/users/me")
    assert response.status_code == 401


def test_get_me_with_invalid_token_returns_401(client: TestClient) -> None:
    response = client.get(
        "/users/me",
        headers={"Authorization": "Bearer invalid.token.here"},
    )
    assert response.status_code == 401
