"""Authoritative ecosystem secrets loader for LemGendary Dataset Compiler.

Resolves credentials directly from the workspace root ``.secrets.yaml`` as the
Single Source of Truth (SSOT).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from core.paths import PROJECT_ROOT

_log = logging.getLogger("lemgendary_datasets.secrets")

SECRETS_FILE_NAME = ".secrets.yaml"


def find_secrets_yaml() -> Path | None:
    """Locate the workspace root ``.secrets.yaml`` file."""
    candidates = [
        PROJECT_ROOT.parent / SECRETS_FILE_NAME,
        PROJECT_ROOT / SECRETS_FILE_NAME,
        Path.cwd() / SECRETS_FILE_NAME,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def load_secrets_records() -> list[dict[str, Any]]:
    """Return all secret records parsed from ``.secrets.yaml``."""
    path = find_secrets_yaml()
    if not path:
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        _log.warning("Failed parsing %s: %s", path, exc)
        return []
    records = data.get("secrets", []) if isinstance(data, dict) else []
    return [r for r in records if isinstance(r, dict)]


def get_secret_record(service: str, username: str | None = None) -> dict[str, Any] | None:
    """Find a secret record by service and optional username.

    If username is omitted, returns the default entry for that service (or the first
    usable entry if none is marked default).
    """
    records = [
        r for r in load_secrets_records() if str(r.get("service", "")).lower() == service.lower()
    ]
    if not records:
        return None
    if username:
        for r in records:
            if str(r.get("username", "")).lower() == username.lower():
                return r
    default_record = next((r for r in records if bool(r.get("is_default", False))), records[0])
    return default_record


def get_secret_value(
    service: str, username: str | None = None, default: str | None = None
) -> str | None:
    """Retrieve secret value for a service."""
    record = get_secret_record(service, username)
    if record:
        val = str(record.get("secret_value") or "").strip()
        if val:
            return val
    return default


def get_kaggle_vault_credentials(
    username: str | None = None,
) -> tuple[str | None, str | None]:
    """Retrieve (token, username) pair from .secrets.yaml for kaggle service."""
    rec = get_secret_record("kaggle", username)
    if not rec:
        return None, None
    token = str(rec.get("secret_value") or "").strip() or None
    user = str(rec.get("username") or "").strip() or None
    return token, user
