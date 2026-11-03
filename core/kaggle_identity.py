"""Kaggle account identity shared by every sync, publish and metadata path.

The owner handle comes from ``identity.kaggle_owner`` in the shared
ecosystem file (see :mod:`core.ecosystem`); registry ``kaggle_ref`` values
that name an owner explicitly take precedence over it.
"""

from __future__ import annotations

from core import ecosystem


def default_owner() -> str:
    """Kaggle account that owns the published manifolds."""
    return ecosystem.kaggle_owner()


def default_handle(slug: str) -> str:
    """``owner/slug`` under the default Kaggle owner."""
    return f"{default_owner()}/{slug}"
