
import asyncio

import pytest
from fastapi import HTTPException

from app import auth


ADMIN_ID = "admin-test-uuid"
REGULAR_ID = "regular-test-uuid"


def check_admin(user):
    return asyncio.run(auth.require_admin(user=user))


def test_admin_is_allowed(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_ENABLED", True)
    monkeypatch.setenv("ADMIN_USER_IDS", ADMIN_ID)

    user = {"sub": ADMIN_ID, "role": "authenticated"}

    result = check_admin(user)

    assert result == user


def test_regular_user_is_denied(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_ENABLED", True)
    monkeypatch.setenv("ADMIN_USER_IDS", ADMIN_ID)

    user = {"sub": REGULAR_ID, "role": "authenticated"}

    with pytest.raises(HTTPException) as exc:
        check_admin(user)

    assert exc.value.status_code == 403


def test_anonymous_user_is_denied(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_ENABLED", True)
    monkeypatch.setenv("ADMIN_USER_IDS", ADMIN_ID)

    with pytest.raises(HTTPException) as exc:
        check_admin({"sub": "anonymous", "role": "anonymous"})

    assert exc.value.status_code == 403


def test_missing_admin_configuration_denies_access(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_ENABLED", True)
    monkeypatch.delenv("ADMIN_USER_IDS", raising=False)

    with pytest.raises(HTTPException) as exc:
        check_admin({"sub": ADMIN_ID, "role": "authenticated"})

    assert exc.value.status_code == 403


def test_disabled_auth_denies_admin_access(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_ENABLED", False)
    monkeypatch.setenv("ADMIN_USER_IDS", ADMIN_ID)

    with pytest.raises(HTTPException) as exc:
        check_admin({"sub": ADMIN_ID, "role": "authenticated"})

    assert exc.value.status_code == 403
