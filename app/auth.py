"""
Supabase authentication middleware for FastAPI.

Verifies JWT tokens issued by Supabase Auth. Supports BOTH:
  - HS256 (legacy shared-secret signing) — verified with SUPABASE_JWT_SECRET
  - ES256/RS256 (modern asymmetric signing) — verified with Supabase's JWKS

The JWKS endpoint is discovered automatically from SUPABASE_URL.
"""


import logging
import os
from typing import Optional

import jwt
import requests
from fastapi import HTTPException, Header, Depends

from src.config import SUPABASE_JWT_SECRET, SUPABASE_URL

logger = logging.getLogger("rag_legal_assistant.auth")

# ── Whether auth is enabled ──────────────────────────────────────
AUTH_ENABLED = bool(SUPABASE_URL)

# Cache for the JWKS keys (fetched once, reused for the process lifetime)
_JWKS_CACHE = None


def _fetch_jwks():
    """Fetch and cache Supabase's JWKS (public keys for ES256/RS256)."""
    global _JWKS_CACHE
    if _JWKS_CACHE is not None:
        return _JWKS_CACHE

    if not SUPABASE_URL:
        raise HTTPException(status_code=500, detail="SUPABASE_URL not configured")

    url = SUPABASE_URL.rstrip("/") + "/auth/v1/.well-known/jwks.json"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        _JWKS_CACHE = r.json()
        logger.info("Fetched Supabase JWKS (%d keys)", len(_JWKS_CACHE.get("keys", [])))
        return _JWKS_CACHE
    except Exception as exc:
        logger.error("Failed to fetch Supabase JWKS: %s", exc)
        raise HTTPException(status_code=500, detail="Auth verification unavailable")


def _get_key_for_token(token: str):
    """Inspect the token header and return (key, algorithm) for verification."""
    try:
        header = jwt.get_unverified_header(token)
    except Exception as exc:
        logger.debug("Could not read token header: %s", exc)
        raise HTTPException(status_code=401, detail="Malformed token")

    alg = header.get("alg", "HS256")
    kid = header.get("kid")

    if alg == "HS256":
        if not SUPABASE_JWT_SECRET:
            raise HTTPException(
                status_code=500,
                detail="Received HS256 token but SUPABASE_JWT_SECRET is not set",
            )
        return SUPABASE_JWT_SECRET, alg

    # Asymmetric — look up the matching public key in the JWKS
    jwks = _fetch_jwks()
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            if alg.startswith("RS"):
                public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
            else:
                public_key = jwt.algorithms.ECAlgorithm.from_jwk(key)
            return public_key, alg

    raise HTTPException(status_code=401, detail="Signing key not found in JWKS")


def _decode_token(token: str) -> dict:
    """Decode and verify a Supabase JWT token. Raises HTTPException on failure."""
    try:
        key, alg = _get_key_for_token(token)
        payload = jwt.decode(
            token,
            key,
            algorithms=[alg],
            audience="authenticated",
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired. Please sign in again.")
    except jwt.InvalidTokenError as exc:
       logger.warning("JWT verification failed: %s", exc)
       raise HTTPException(status_code=401, detail="Invalid authentication token.")

# ── Anonymous user placeholder ───────────────────────────────────
_ANONYMOUS_USER = {
    "sub": "anonymous",
    "email": "anonymous@local",
    "role": "anonymous",
}


async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """FastAPI dependency — returns the authenticated user payload."""
    if not AUTH_ENABLED:
        return _ANONYMOUS_USER

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required. Please sign in.")

    token = authorization.split(" ", 1)[1]
    return _decode_token(token)


async def get_optional_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Like get_current_user but returns None instead of raising 401."""
    if not AUTH_ENABLED:
        return _ANONYMOUS_USER

    if not authorization or not authorization.startswith("Bearer "):
        return None

    try:
        return _decode_token(authorization.split(" ", 1)[1])
    except HTTPException:
        return None
    # ── Admin authorization ─────────────────────────────────────────

def _admin_ids() -> set[str]:
    """Read trusted administrator user IDs from environment variables."""
    return {
        user_id.strip()
        for user_id in os.getenv("ADMIN_USER_IDS", "").split(",")
        if user_id.strip()
    }


async def require_admin(
    user: dict = Depends(get_current_user),
) -> dict:
    """Allow only authenticated administrators to access this route."""

    # Never permit admin access when authentication is disabled.
    if not AUTH_ENABLED or user.get("sub") == "anonymous":
        raise HTTPException(
            status_code=403,
            detail="Admin access denied",
        )

    if user.get("sub") not in _admin_ids():
        raise HTTPException(
            status_code=403,
            detail="Admin access denied",
        )

    return user