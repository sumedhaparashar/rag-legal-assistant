
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.auth import get_current_user, require_admin


ADMIN = {
    "sub": "admin-test-uuid",
    "role": "authenticated",
}

REGULAR_USER = {
    "sub": "regular-test-uuid",
    "role": "authenticated",
}

PROTECTED_ROUTES = [
    ("GET", "/api/analytics"),
    ("POST", "/ingest"),
    ("POST", "/auto-ingest"),
]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.auth.AUTH_ENABLED", True)
    monkeypatch.setenv("ADMIN_USER_IDS", ADMIN["sub"])

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_regular_user_cannot_access_admin_routes(
    client, method, path
):
    app.dependency_overrides[get_current_user] = (
        lambda: REGULAR_USER
    )

    response = client.request(method, path)

    assert response.status_code == 403


@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_anonymous_user_cannot_access_admin_routes(
    client, method, path
):
    app.dependency_overrides[get_current_user] = (
        lambda: {"sub": "anonymous", "role": "anonymous"}
    )

    response = client.request(method, path)

    assert response.status_code == 403


@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_admin_dependency_is_attached_to_routes(
    client, method, path
):
    app.dependency_overrides[get_current_user] = (
        lambda: ADMIN
    )

    # Override the final authorization dependency so that
    # the test verifies route wiring without triggering
    # actual ingestion or analytics operations.
    app.dependency_overrides[require_admin] = (
        lambda: ADMIN
    )

    # Verify that the route is registered.
    matching_routes = [
        route for route in app.routes
        if getattr(route, "path", None) == path
        and method in getattr(route, "methods", set())
    ]

    assert len(matching_routes) == 1

    route = matching_routes[0]
    dependencies = [
        dependency.call
        for dependency in route.dependant.dependencies
    ]

    assert require_admin in dependencies
@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_missing_token_returns_401(monkeypatch, method, path):
    monkeypatch.setattr("app.auth.AUTH_ENABLED", True)

    # Don't override authentication: test the real dependency.
    app.dependency_overrides.clear()

    try:
        # No lifespan startup, so this test won't load FAISS.
        client = TestClient(app)

        response = client.request(method, path)

        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()