"""Registry lookup of the loose Kaggle manifolds a recompile is packed from.

A ``unified_data.yaml`` entry lists them under ``source_manifolds`` (for
example the RetinaFace entry names
``kaggle://<owner>/lemgendizedretinafacemobilenetlarge``). Pulling such
a source lands it in the entry's ``modernized_folder`` and packs its text
labels with the entry's ``label_json`` strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.paths import UNIFIED_DATA

__all__ = ["SourceManifold", "find_source_manifold"]

_REF_PREFIX = "kaggle://"


@dataclass(frozen=True)
class SourceManifold:
    """The registry entry a pulled source manifold belongs to."""

    key: str
    modernized_folder: str
    label_json: str | None


def _slug(ref: str) -> str:
    return ref.removeprefix(_REF_PREFIX).strip("/").split("/")[-1].lower()


def find_source_manifold(slug: str, registry: Path = UNIFIED_DATA) -> SourceManifold | None:
    """The entry whose ``source_manifolds`` lists the Kaggle dataset ``slug``, else None.

    ``slug`` may be ``owner/name`` or ``name``. Raises ``OSError`` or
    ``yaml.YAMLError`` when the registry cannot be read.
    """
    wanted = _slug(slug)
    with open(registry, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    for key, entry in (data.get("datasets") or {}).items():
        refs = entry.get("source_manifolds") or []
        if any(_slug(str(ref)) == wanted for ref in refs):
            folder = entry.get("modernized_folder") or f"LemGendized{entry.get('name', key)}"
            return SourceManifold(key, str(folder), entry.get("label_json"))
    return None
