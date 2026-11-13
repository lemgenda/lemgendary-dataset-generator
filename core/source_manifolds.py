"""Registry lookup of the loose Kaggle manifolds a recompile is packed from.

A ``unified_data.yaml`` entry lists them under ``source_manifolds`` (for
example the RetinaFace entry names
``kaggle://<owner>/lemgendizedretinafacemobilenetlarge``). Pulling such
a source lands it in the entry's ``modernized_folder`` and packs its text
labels with the entry's label strategy (``label_json``, else its task's). The labels-only repair
(``repair labels``) looks the strategy up by the manifold's folder name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.paths import UNIFIED_DATA
from utils.label_json import resolve_label_strategy

__all__ = [
    "SourceManifold",
    "find_registry_manifold",
    "find_source_manifold",
    "label_strategy_for_folder",
    "registry_entry_for_folder",
]

_REF_PREFIX = "kaggle://"


@dataclass(frozen=True)
class SourceManifold:
    """The registry entry a pulled source manifold belongs to."""

    key: str
    modernized_folder: str
    label_json: str | None


def _slug(ref: str) -> str:
    return ref.removeprefix(_REF_PREFIX).strip("/").split("/")[-1].lower()


def _entries(registry: Path) -> dict[str, Any]:
    with open(registry, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    return data.get("datasets") or {}


def _folder(key: str, entry: dict[str, Any]) -> str:
    return str(entry.get("modernized_folder") or f"LemGendized{entry.get('name', key)}")


def find_source_manifold(slug: str, registry: Path = UNIFIED_DATA) -> SourceManifold | None:
    """The entry whose ``source_manifolds`` lists the Kaggle dataset ``slug``, else None.

    ``slug`` may be ``owner/name`` or ``name``. Raises ``OSError`` or
    ``yaml.YAMLError`` when the registry cannot be read.
    """
    wanted = _slug(slug)
    for key, entry in _entries(registry).items():
        refs = entry.get("source_manifolds") or []
        if any(_slug(str(ref)) == wanted for ref in refs):
            return SourceManifold(key, _folder(key, entry), _strategy(entry))
    return None


def find_registry_manifold(slug: str, registry: Path = UNIFIED_DATA) -> SourceManifold | None:
    """The entry whose own ``kaggle_ref`` is the Kaggle dataset ``slug``, else None.

    A pulled compiled manifold packs its text labels with this entry's label
    strategy. Raises ``OSError`` or ``yaml.YAMLError`` when the registry
    cannot be read.
    """
    wanted = _slug(slug)
    for key, entry in _entries(registry).items():
        ref = entry.get("kaggle_ref")
        if ref and _slug(str(ref)) == wanted:
            return SourceManifold(key, _folder(key, entry), _strategy(entry))
    return None


def registry_entry_for_folder(folder: str, registry: Path = UNIFIED_DATA) -> dict[str, Any] | None:
    """The registry entry compiled into ``folder``, else None.

    Raises ``OSError`` or ``yaml.YAMLError`` when the registry cannot be read.
    """
    for key, entry in _entries(registry).items():
        if _folder(key, entry) == folder:
            return entry
    return None


def _strategy(entry: dict[str, Any]) -> str | None:
    """The entry's ``label_json``, else the label strategy of its task."""
    return resolve_label_strategy(entry.get("label_json"), entry.get("task"))


def label_strategy_for_folder(folder: str, registry: Path = UNIFIED_DATA) -> str | None:
    """The label strategy of the entry compiled into ``folder``, else None.

    The entry's ``label_json``, else its task's strategy
    (:func:`utils.label_json.resolve_label_strategy`). Raises ``OSError`` or
    ``yaml.YAMLError`` when the registry cannot be read.
    """
    entry = registry_entry_for_folder(folder, registry)
    return _strategy(entry) if entry else None
