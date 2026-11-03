"""Kaggle account identity shared by every sync, publish and metadata path.

The owner handle is defined here only; registry ``kaggle_ref`` values that
name an owner explicitly take precedence over it.
"""

from __future__ import annotations

DEFAULT_KAGGLE_OWNER = "lemtreursi"


def default_handle(slug: str) -> str:
    """``owner/slug`` under the default Kaggle owner."""
    return f"{DEFAULT_KAGGLE_OWNER}/{slug}"
