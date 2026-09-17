"""Token-based authentication module for LemGendary Dataset Compiler API."""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger("lemgendary.api.auth")

_TOKEN_FILE = Path(".lgd_server/token")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_security = HTTPBearer(auto_error=False)


def get_or_create_token() -> str:
    """Retrieve existing server security token or generate a persistent local token."""
    env_token = os.environ.get("LEMGENDARY_API_TOKEN", "").strip()
    if env_token:
        return env_token

    if _TOKEN_FILE.exists():
        try:
            stored = _TOKEN_FILE.read_text(encoding="utf-8").strip()
            if stored:
                return stored
        except OSError as exc:
            logger.warning("Failed to read server token file %s: %s", _TOKEN_FILE, exc)

    new_token = secrets.token_urlsafe(32)
    try:
        _TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        _TOKEN_FILE.write_text(new_token, encoding="utf-8")
        logger.info("Generated new API access token at %s", _TOKEN_FILE)
    except OSError as exc:
        logger.warning("Failed to persist server token to %s: %s", _TOKEN_FILE, exc)

    return new_token


async def verify_token(
    header_key: Optional[str] = Security(_api_key_header),
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Security(_bearer_security),
) -> str:
    """Validate incoming HTTP request authentication token against server master token."""
    if os.environ.get("LEMGENDARY_DISABLE_AUTH", "").lower() in ("1", "true", "yes"):
        return "auth_disabled"

    expected = get_or_create_token()
    provided = header_key or (bearer_creds.credentials if bearer_creds else None)

    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API token. Provide via 'X-API-Key' header or 'Authorization: Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API token provided.",
        )

    return provided
