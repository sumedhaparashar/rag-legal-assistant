"""
Supabase authentication middleware for FastAPI.

Verifies JWT tokens issued by Supabase Auth using the project's
JWT secret (HS256). Provides FastAPI dependencies for protected
endpoints.

When SUPABASE_JWT_SECRET is not set, auth is bypassed entirely
(development mode) — all requests are treated as anonymous.
"""

import logging
from typing import Optional

import jwt
from fastapi import HTTPException, Header

from src.config import SUPABASE_JWT_SECRET

logger = logging.getLogger("rag_legal_assistant.auth")

# ── Whether auth is enabled ──────────────────────────────────────
AUTH_ENABLED = bool(SUPABASE_JWT_SECRET)

if not AUTH_ENABLED:
    logger.warning(
        "SUPABASE_JWT_SECRET is not set — authentication is DISABLED. "
        "All requests will be treated as anonymous. "
        "Set SUPABASE_JWT_SECRET to enable auth."
    )


def _decode_token(token: str) -> dict:
    """Decode and verify a Supabase JWT token.

    Returns the decoded payload dict on success.
    Raises HTTPException on any verification failure.
    """
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token expired. Please sign in again.",
        )
    except jwt.InvalidTokenError as exc:
        logger.debug("JWT verification failed: %s", exc)
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token.",
        )


# ── Anonymous user placeholder ───────────────────────────────────
_ANONYMOUS_USER = {
    "sub": "anonymous",
    "email": "anonymous@local",
    "role": "anonymous",
}


async def get_current_user(
    authorization: Optional[str] = Header(None),
) -> dict:
    """FastAPI dependency — returns the authenticated user payload.

    When auth is disabled (no JWT secret), returns an anonymous
    placeholder so endpoints can still access ``user["sub"]``.

    When auth is enabled, requires a valid ``Authorization: Bearer <token>``
    header. Returns 401 if missing or invalid.
    """
    if not AUTH_ENABLED:
        return _ANONYMOUS_USER

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign in.",
        )

    token = authorization.split(" ", 1)[1]
    return _decode_token(token)


async def get_optional_user(
    authorization: Optional[str] = Header(None),
) -> Optional[dict]:
    """FastAPI dependency — like get_current_user but returns None
    instead of raising 401 when auth is missing/invalid.

    Useful for endpoints that work differently for authed vs anon users.
    """
    if not AUTH_ENABLED:
        return _ANONYMOUS_USER

    if not authorization or not authorization.startswith("Bearer "):
        return None

    try:
        return _decode_token(authorization.split(" ", 1)[1])
    except HTTPException:
        return None
