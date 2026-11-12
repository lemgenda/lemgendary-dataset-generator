"""
Kaggle authentication and dataset-handle helpers for the sync core.

Owns credential bootstrap (explicit args, else the vault secret, exported by
kind through core.kaggle_credentials), the
TLS CA bundle wiring, Kaggle handle normalization, and the legacy *large
manifold safeguard shared by every upload/metadata path.
"""

import importlib
import logging
import os
import subprocess
import sys

import certifi

from core.kaggle_credentials import (
    ENV_USERNAME,
    SOURCE_EXPLICIT,
    KaggleCredential,
    export_credential,
    has_kaggle_credentials,
    stored_credential,
)
from core.kaggle_identity import default_owner

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


def setup_kaggle_auth(
    default_user: str | None = None,
    custom_user: str | None = None,
    custom_token: str | None = None,
) -> bool:
    """Export Kaggle credentials (explicit args, else the vault's default Kaggle record).

    A ``KGAT_`` access token goes to ``KAGGLE_API_TOKEN`` unchanged, a legacy
    key to ``KAGGLE_KEY`` (see :mod:`core.kaggle_credentials`). ``default_user``
    (default: the Kaggle owner in the ecosystem file) names the account when
    the stored secret does not. With no stored secret the environment and the
    ``~/.kaggle`` files are left to the Kaggle client.

    Returns True when the Kaggle client will find a token or key afterwards.
    """
    if custom_token:
        credential: KaggleCredential | None = KaggleCredential(
            secret=custom_token, username=custom_user, source=SOURCE_EXPLICIT
        )
    else:
        credential = stored_credential()
    if credential is not None:
        export_credential(credential, default_user or default_owner())
    elif custom_user:
        os.environ[ENV_USERNAME] = custom_user
    else:
        os.environ.setdefault(ENV_USERNAME, default_user or default_owner())

    _configure_ca_bundle()
    return has_kaggle_credentials()
