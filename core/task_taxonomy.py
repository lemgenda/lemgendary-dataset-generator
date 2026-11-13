"""Read ``task_taxonomy.json``: task ids, layout folders, strategies and name patterns.

``task_taxonomy.json`` (schema: ``task_taxonomy.schema.json``) at the
repository root is the single source of truth for:

- the compiler task ids;
- the split folder each task adds to a fresh manifold (``labels``, ``masks``,
  ``targets`` or none);
- the default label and mask generation strategy of each task;
- the name patterns of the fallback heuristic, used only when a registry
  entry has no ``task`` (every entry of ``unified_data.yaml`` has one);
- the documentation profile key of each manifold folder name.

``LEMGENDARY_TASK_TAXONOMY_FILE`` points the loader at another file (tests).
A missing or malformed file raises :class:`TaskTaxonomyError` naming the path.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from core.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

TASK_TAXONOMY_ENV = "LEMGENDARY_TASK_TAXONOMY_FILE"
"""Environment variable that overrides the taxonomy file location."""

DEFAULT_TASK_TAXONOMY_FILE = PROJECT_ROOT / "task_taxonomy.json"

LAYOUT_DIRS: tuple[str, ...] = ("labels", "masks", "targets")
"""Split folders a task may add next to ``images/``."""


class TaskTaxonomyError(RuntimeError):
    """The taxonomy file is missing, unreadable or invalid."""


@dataclass(frozen=True)
class TaskSpec:
    """One task of the taxonomy."""

    id: str
    layout_dir: str | None
    label_strategy: str | None
    mask_strategy: str | None
    name_patterns: tuple[str, ...]


@dataclass(frozen=True)
class TaskTaxonomy:
    """The parsed taxonomy file."""

    path: Path
    tasks: tuple[TaskSpec, ...]
    fallback_task: str
    unnamed_task: str
    documentation_tasks: Mapping[str, str]

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Every task id, in file order."""
        return tuple(t.id for t in self.tasks)

    def get(self, task_id: str) -> TaskSpec | None:
        """The task named ``task_id``, or None when the taxonomy has no such task."""
        return next((t for t in self.tasks if t.id == task_id), None)

    def layout_dir(self, task_id: str) -> str | None:
        """Split folder ``task_id`` adds to a fresh manifold (None: images only)."""
        spec = self.get(task_id)
        return spec.layout_dir if spec else None

    def label_strategy(self, task_id: str) -> str | None:
        """Default label-generation strategy of ``task_id``."""
        spec = self.get(task_id)
        return spec.label_strategy if spec else None

    def mask_strategy(self, task_id: str) -> str | None:
        """Default mask-generation strategy of ``task_id``."""
        spec = self.get(task_id)
        return spec.mask_strategy if spec else None

    def infer_from_name(self, name: str | None) -> str:
        """Heuristic task of a dataset key: the first task with a pattern in the name."""
        if not name:
            return self.unnamed_task
        lowered = name.lower()
        return next(
            (t.id for t in self.tasks if any(p in lowered for p in t.name_patterns)),
            self.fallback_task,
        )


def taxonomy_file() -> Path:
    """Path of the taxonomy file that :func:`taxonomy` reads."""
    override = os.environ.get(TASK_TAXONOMY_ENV)
    return Path(override) if override else DEFAULT_TASK_TAXONOMY_FILE


def _fail(path: Path, message: str) -> TaskTaxonomyError:
    return TaskTaxonomyError(f"Invalid task taxonomy {path}: {message}")


def _optional_str(raw: dict[str, Any], key: str, path: Path) -> str | None:
    if key not in raw:
        raise _fail(path, f"task '{raw.get('id')}' has no '{key}'")
    value = raw[key]
    if value is not None and not (isinstance(value, str) and value):
        raise _fail(path, f"task '{raw.get('id')}': '{key}' must be a string or null")
    return value


def _parse_task(raw: Any, path: Path) -> TaskSpec:
    if not isinstance(raw, dict):
        raise _fail(path, "task entry is not an object")
    task_id = raw.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise _fail(path, "task entry has no string 'id'")
    layout_dir = _optional_str(raw, "layout_dir", path)
    if layout_dir is not None and layout_dir not in LAYOUT_DIRS:
        raise _fail(path, f"task '{task_id}': layout_dir must be one of {LAYOUT_DIRS} or null")
    patterns = raw.get("name_patterns")
    if not isinstance(patterns, list) or not all(isinstance(p, str) and p for p in patterns):
        raise _fail(path, f"task '{task_id}': name_patterns must be a list of strings")
    return TaskSpec(
        id=task_id,
        layout_dir=layout_dir,
        label_strategy=_optional_str(raw, "label_strategy", path),
        mask_strategy=_optional_str(raw, "mask_strategy", path),
        name_patterns=tuple(p.lower() for p in patterns),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TaskTaxonomyError(
            f"Task taxonomy not found: expected {path} (or set {TASK_TAXONOMY_ENV})"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskTaxonomyError(f"Cannot read task taxonomy {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise _fail(path, "top level is not an object")
    return raw


@cache
def _load(path: Path) -> TaskTaxonomy:
    raw = _read_json(path)
    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise _fail(path, "'tasks' must be a non-empty list")
    tasks = tuple(_parse_task(t, path) for t in raw_tasks)
    ids = [t.id for t in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise _fail(path, f"duplicate task ids {duplicates}")

    heuristic = raw.get("heuristic")
    if not isinstance(heuristic, dict):
        raise _fail(path, "'heuristic' must be an object")
    fallback, unnamed = heuristic.get("fallback_task"), heuristic.get("unnamed_task")
    for key, value in (("fallback_task", fallback), ("unnamed_task", unnamed)):
        if value not in ids:
            raise _fail(path, f"heuristic.{key} '{value}' is not a task id")

    doc_tasks = raw.get("documentation_tasks")
    if not isinstance(doc_tasks, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in doc_tasks.items()
    ):
        raise _fail(path, "'documentation_tasks' must map folder names to strings")

    return TaskTaxonomy(
        path=path,
        tasks=tasks,
        fallback_task=str(fallback),
        unnamed_task=str(unnamed),
        documentation_tasks=MappingProxyType(dict(doc_tasks)),
    )


def taxonomy() -> TaskTaxonomy:
    """Return the parsed taxonomy file, cached per path."""
    return _load(taxonomy_file())


def clear_cache() -> None:
    """Forget parsed files (tests that switch ``LEMGENDARY_TASK_TAXONOMY_FILE``)."""
    _load.cache_clear()
    _logged_heuristic.cache_clear()


def entry_task(entry: Mapping[str, Any]) -> str | None:
    """The task of one registry entry: ``task_override``, else ``task``; None when unset."""
    registered = entry.get("task_override") or entry.get("task")
    return str(registered) if registered else None


def registry_task(datasets: Mapping[str, Any], name: str) -> str | None:
    """The registry task of a dataset key or name: ``task_override``, else ``task``."""
    lowered = name.lower()
    for ds_key, ds_cfg in datasets.items():
        if not isinstance(ds_cfg, dict):
            continue
        if ds_key.lower() == lowered or str(ds_cfg.get("name", "")).lower() == lowered:
            registered = entry_task(ds_cfg)
            if registered:
                return registered
    return None


@cache
def _logged_heuristic(name: str) -> str:
    task = taxonomy().infer_from_name(name)
    logger.info("No registry task for %r; inferred %r from its name", name, task)
    return task


def resolve_task(datasets: Mapping[str, Any], model_key: str | None) -> str:
    """Task of a dataset key: the registry ``task_override``/``task``, else the heuristic.

    The heuristic is a logged fallback for keys the registry does not type
    (custom or unknown entries); every ``unified_data.yaml`` entry sets ``task``.
    """
    if not model_key:
        return taxonomy().unnamed_task
    return registry_task(datasets, model_key) or _logged_heuristic(model_key.lower())
