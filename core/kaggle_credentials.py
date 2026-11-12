"""Kaggle credential kinds, stores and export: the one place that knows them.

The installed Kaggle client (kaggle 2.x with kagglesdk, and kagglehub) knows
two kinds of secret:

- an API access token, which starts with ``KGAT_``. It is read unchanged from
  ``KAGGLE_API_TOKEN`` or from ``~/.kaggle/access_token`` (or
  ``access_token.txt``); no username is needed (the client asks Kaggle whose
  token it is);
- a legacy API key, read as the pair ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` or
  from ``kaggle.json`` in the Kaggle config folder.

A ``KGAT_`` token sent as a legacy key is rejected (HTTP 401), so a stored
secret is exported by kind and never with its prefix stripped.

The stored secret is the workspace vault's (``.secrets.yaml``) default Kaggle
record. Ambient credentials (the environment variables and files above) are
left to the Kaggle client.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from core.secrets import get_kaggle_vault_credentials

logger = logging.getLogger(__name__)

ACCESS_TOKEN_PREFIX = "KGAT_"
ENV_API_TOKEN = "KAGGLE_API_TOKEN"
ENV_USERNAME = "KAGGLE_USERNAME"
ENV_KEY = "KAGGLE_KEY"
ENV_CONFIG_DIR = "KAGGLE_CONFIG_DIR"

SOURCE_EXPLICIT = "explicit"
SOURCE_VAULT = "secrets_vault"


def is_access_token(secret: str) -> bool:
    """True for a Kaggle API access token (``KGAT_`` prefix), False for a legacy key."""
    return secret.startswith(ACCESS_TOKEN_PREFIX)


@dataclass(frozen=True)
class KaggleCredential:
    """One Kaggle secret, the account it belongs to (if known) and where it came from."""

    secret: str
    username: str | None
    source: str

    @property
    def is_access_token(self) -> bool:
        """True when the secret is a ``KGAT_`` access token."""
        return is_access_token(self.secret)


def access_token_files() -> tuple[Path, Path]:
    """``~/.kaggle/access_token`` and ``access_token.txt``, where kagglesdk looks."""
    folder = Path.home() / ".kaggle"
    return folder / "access_token", folder / "access_token.txt"


def legacy_config_file() -> Path:
    """``kaggle.json`` in ``KAGGLE_CONFIG_DIR`` (default ``~/.kaggle``)."""
    config_dir = os.environ.get(ENV_CONFIG_DIR)
    base = Path(config_dir) if config_dir else Path.home() / ".kaggle"
    return base / "kaggle.json"


def _read_token_file(path: Path) -> str | None:
    """Stripped content of ``path``; None when it is missing, empty or unreadable."""
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Cannot read Kaggle token file %s: %s", path, exc)
        return None


def _has_access_token_file() -> bool:
    """True when one of the kagglesdk access token files holds a token."""
    return any(_read_token_file(path) is not None for path in access_token_files())


def stored_credential() -> KaggleCredential | None:
    """The vault's default Kaggle record, else None."""
    secret, username = get_kaggle_vault_credentials()
    if secret:
        return KaggleCredential(secret=secret, username=username, source=SOURCE_VAULT)
    return None


def export_credential(credential: KaggleCredential, default_user: str | None = None) -> None:
    """Put ``credential`` into the environment variables its kind belongs to.

    An access token goes to ``KAGGLE_API_TOKEN`` as is; a legacy key goes to
    ``KAGGLE_KEY``. ``KAGGLE_USERNAME`` is the credential's own account, else
    ``default_user`` (a legacy key only works with the account that owns it).
    """
    if credential.is_access_token:
        os.environ[ENV_API_TOKEN] = credential.secret
    else:
        os.environ[ENV_KEY] = credential.secret
    username = credential.username or default_user
    if username:
        os.environ[ENV_USERNAME] = username


def environment_has_credential() -> bool:
    """True when the environment holds an access token or a username/key pair."""
    return bool(
        os.environ.get(ENV_API_TOKEN)
        or (os.environ.get(ENV_USERNAME) and os.environ.get(ENV_KEY))
    )


def credential_sources() -> dict[str, bool]:
    """Which credential stores hold a Kaggle secret (no secret is returned)."""
    return {
        "environment_variables": environment_has_credential(),
        SOURCE_VAULT: bool(get_kaggle_vault_credentials()[0]),
        "user_access_token": _has_access_token_file(),
        "user_kaggle_json": legacy_config_file().is_file(),
    }


def has_kaggle_credentials() -> bool:
    """True when the Kaggle client will find a token or key in the environment or files."""
    return (
        environment_has_credential()
        or _has_access_token_file()
        or legacy_config_file().is_file()
    )
