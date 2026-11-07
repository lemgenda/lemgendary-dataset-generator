"""
Kaggle authentication and dataset-handle helpers for the sync core.

Owns credential bootstrap (explicit args > environment > .kaggle_token), the
TLS CA bundle wiring, Kaggle handle normalization, and the legacy *large
manifold safeguard shared by every upload/metadata path.
"""

import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path

import certifi
from kagglehub import clients as kagglehub_clients

from core.kaggle_errors import KAGGLE_CLIENT_ERRORS
from core.kaggle_identity import default_owner
from core.lazy_imports import load_kaggle_api
from core.secrets import get_kaggle_vault_credentials

logger = logging.getLogger(__name__)


def normalize_kaggle_handle(repo_id: str) -> str:
    """Strip Kaggle dataset URLs and ``kaggle://`` URIs down to ``owner/slug``."""
    clean = repo_id.strip()
    if "kaggle.com/datasets/" in clean:
        clean = clean.split("kaggle.com/datasets/")[-1].split("?")[0].strip("/")
    elif clean.startswith("kaggle://"):
        clean = clean.replace("kaggle://", "").strip("/")
    return clean


def split_owner_slug(handle: str) -> tuple[str, str]:
    """Split ``owner/slug`` into its parts; a bare slug gets the default owner."""
    if "/" in handle:
        parts = handle.split("/")
        return parts[0], parts[1]
    return default_owner(), handle


def kaggle_cli_creationflags() -> int:
    """Subprocess creation flags that keep the Kaggle CLI from opening a console on Windows."""
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return 0


def is_kaggle_large_manifold(repo_id: str) -> bool:
    """Detect whether a repository handle points to a legacy *large manifold on Kaggle."""
    clean = normalize_kaggle_handle(repo_id)
    slug = clean.split("/")[-1].lower() if "/" in clean else clean.lower()
    return (
        slug.endswith("large")
        or "manifoldlarge" in slug
        or "large" in slug.split("_")
        or "large" in slug.split("-")
    )


def _token_file() -> Path:
    """Locate the repository ``.kaggle_token`` (repo root first, then this package)."""
    root = (
        Path(__file__).resolve().parent.parent
        if Path(__file__).parent.name == "core"
        else Path(__file__).resolve().parent
    )
    token_file = root / ".kaggle_token"
    if not token_file.exists():
        token_file = Path(__file__).parent / ".kaggle_token"
    return token_file


def _apply_token_file(default_user: str | None) -> None:
    """Export Kaggle credentials from .secrets.yaml or legacy .kaggle_token into env."""
    token, user = get_kaggle_vault_credentials()
    if not token:
        token_file = _token_file()
        if token_file.exists():
            token = token_file.read_text().strip() or None

    if token and "KAGGLE_API_TOKEN" not in os.environ:
        os.environ["KAGGLE_API_TOKEN"] = token
    if "KAGGLE_USERNAME" not in os.environ:
        chosen_user = default_user or user
        if chosen_user:
            os.environ["KAGGLE_USERNAME"] = chosen_user


def _apply_pip_certifi_fallback() -> None:
    """Point TLS at pip's vendored certifi bundle when the main certifi bundle is missing."""
    try:
        pip_cert_mod = importlib.import_module("pip._vendor.certifi")
        pip_cert_where = getattr(pip_cert_mod, "where", None)
        if pip_cert_where:
            pip_path = pip_cert_where()
            if os.path.exists(pip_path):
                os.environ["SSL_CERT_FILE"] = pip_path
                os.environ["REQUESTS_CA_BUNDLE"] = pip_path
    except (ImportError, AttributeError, OSError) as fallback_exc:
        logger.debug("Pip certifi fallback resolution skipped: %s", fallback_exc)


def _configure_ca_bundle() -> None:
    """Export SSL_CERT_FILE / REQUESTS_CA_BUNDLE from certifi unless already set."""
    try:
        cert_path = certifi.where()
        if os.path.exists(cert_path):
            os.environ.setdefault("SSL_CERT_FILE", cert_path)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)
        else:
            _apply_pip_certifi_fallback()
    except (ImportError, AttributeError, OSError) as cert_exc:
        logger.debug("TLS CA certificate bundle check notice: %s", cert_exc)


def _suppress_version_warnings() -> None:
    """Silence the kagglehub / KaggleApi "newer version available" banners."""
    kagglehub_clients.already_printed_version_warning = True
    try:
        load_kaggle_api().already_printed_version_warning = True
    except KAGGLE_CLIENT_ERRORS as exc:
        logger.debug("KaggleApi warning flag suppression skipped: %s", exc)


def setup_kaggle_auth(
    default_user: str | None = None,
    custom_user: str | None = None,
    custom_token: str | None = None,
) -> bool:
    """Ensure Kaggle credentials are initialized (explicit args > environment > .kaggle_token).

    ``default_user`` defaults to the Kaggle owner in the ecosystem file.
    Returns True when an API token or username/key pair is available afterwards.
    """
    if custom_token:
        os.environ["KAGGLE_API_TOKEN"] = custom_token
    if custom_user:
        os.environ["KAGGLE_USERNAME"] = custom_user
    if not custom_token:
        _apply_token_file(default_user or default_owner())

    _configure_ca_bundle()
    _suppress_version_warnings()

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return bool(
        os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY") or kaggle_json.exists()
    )
