"""Write API for the unified_data.yaml dataset registry (single source of truth).

Adding a manifold must never damage the registry:

* Existing text is preserved byte for byte (comments, folding, ordering).
  The new entry is serialized on its own and inserted at the end of the
  top-level ``datasets:`` block; nothing else is re-emitted.
* An existing key is never overwritten (``RegistryEntryExistsError``).
* The merged document is validated with the compiler's own schema
  (``UnifiedData``) before anything is written, so a bad entry cannot make
  every later compile exit at start-up.
* The write is atomic: a temp file in the same directory is fsynced and
  then moved over the registry with ``os.replace``.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import yaml

from core.config_schema import UnifiedData

_log = logging.getLogger(__name__)

DEFAULT_UNIFIED_DATA_PATH = Path(__file__).resolve().parent.parent / "unified_data.yaml"

_WRITE_LOCK = threading.Lock()
_ENTRY_INDENT = "  "


class RegistryEntryExistsError(ValueError):
    """Raised when a dataset key is already defined in the registry."""


class RegistryWriteError(RuntimeError):
    """Raised when the registry cannot be updated safely."""


def _find_datasets_block_end(lines: list[str]) -> int:
    """Return the line index where a new entry must be inserted.

    That is just after the last non-blank, non-comment line of the
    top-level ``datasets:`` block (before the next top-level key, or EOF).
    """
    start = next((i for i, line in enumerate(lines) if line.rstrip() == "datasets:"), None)
    if start is None:
        raise RegistryWriteError("Registry has no block-style top-level 'datasets:' key")

    insert_at = start + 1
    for i in range(start + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line[0].isspace():
            break
        insert_at = i + 1
    return insert_at


def _render_entry(key: str, entry: dict[str, Any], newline: str) -> list[str]:
    """Serialize one dataset entry indented for the ``datasets:`` block."""
    body = yaml.safe_dump({key: entry}, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return [f"{_ENTRY_INDENT}{line}{newline}" for line in body.splitlines()]


def _atomic_write(path: Path, text: str) -> None:
    """Write text to path via a same-directory temp file and os.replace."""
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except OSError:
        _log.error("Atomic write to %s failed; removing temp file %s", path, tmp_path)
        tmp_path.unlink(missing_ok=True)
        raise


def add_dataset_entry(key: str, entry: dict[str, Any], path: Path | None = None) -> None:
    """Add a new dataset entry to unified_data.yaml without touching existing text.

    Raises:
        FileNotFoundError: the registry file does not exist.
        RegistryEntryExistsError: ``key`` is already defined.
        pydantic.ValidationError: the merged registry fails the compiler schema.
        RegistryWriteError: the registry layout does not allow a safe insert.
        OSError: the atomic write failed (the registry is left unchanged).
    """
    target = path or DEFAULT_UNIFIED_DATA_PATH
    with _WRITE_LOCK:
        # Decode bytes directly so the file's own line endings are preserved.
        original = target.read_bytes().decode("utf-8")
        newline = "\r\n" if "\r\n" in original else "\n"
        raw = yaml.safe_load(original)
        if not isinstance(raw, dict):
            raise RegistryWriteError(f"Registry root of {target} must be a mapping")

        datasets = raw.get("datasets")
        if not isinstance(datasets, dict):
            raise RegistryWriteError(f"Registry {target} has no 'datasets' mapping")
        if key in datasets:
            raise RegistryEntryExistsError(f"Dataset key '{key}' already exists in {target.name}")

        merged = dict(raw)
        merged["datasets"] = {**datasets, key: entry}
        UnifiedData.model_validate(merged)

        lines = original.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += newline
        insert_at = _find_datasets_block_end(lines)
        new_text = "".join(lines[:insert_at] + _render_entry(key, entry, newline) + lines[insert_at:])

        if yaml.safe_load(new_text) != merged:
            raise RegistryWriteError(
                f"Inserting '{key}' into {target.name} would not yield the validated registry; nothing written"
            )

        _atomic_write(target, new_text)
        _log.info("Added dataset '%s' to %s", key, target)
