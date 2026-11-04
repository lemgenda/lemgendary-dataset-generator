"""Parquet schema mapper. Extracted from compiler_core.py in Phase 1.4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.lazy_imports import load_parquet

# Canonical role -> candidate column names; the first present column wins.
_FIRST_MATCH_ROLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("file_name", ("image", "pixel_values")),
    ("url", ("url",)),
    ("key", ("key",)),
    ("class", ("label",)),
)

# Restoration targets: later candidates override earlier ones when present.
_TARGET_COLUMNS: tuple[str, ...] = ("target", "sharp", "ground_truth")

# Bbox/seg roles -> substrings matched against lowercased column names; the
# last matching column wins.
_SUBSTRING_ROLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("xmin", ("xmin", "x1")),
    ("ymin", ("ymin", "y1")),
    ("width", ("width", "w")),
    ("height", ("height", "h")),
)


def _map_columns(cols: list[str]) -> dict[str, Any]:
    """Build the canonical-role -> column-name mapping for a schema."""
    mapping: dict[str, Any] = {}
    for role, candidates in _FIRST_MATCH_ROLES:
        found = next((c for c in candidates if c in cols), None)
        if found is not None:
            mapping[role] = found

    for target in _TARGET_COLUMNS:
        if target in cols:
            mapping["target"] = target

    for c in cols:
        cl = c.lower()
        for role, needles in _SUBSTRING_ROLES:
            if any(x in cl for x in needles):
                mapping[role] = c
    return mapping


def parse_parquet(
    pq_path: str | Path,
) -> tuple[Path, dict[str, Any], list[str]]:
    """Inspect a Parquet file's schema and return a column-name mapping.

    Returns:
        pq_path:  the resolved path (as `Path`)
        mapping:  canonical-role -> actual-column-name (e.g. "file_name" -> "image")
        cols:     the schema's full column-name list, in declaration order
    """
    pq = load_parquet()

    path = Path(pq_path)
    schema = pq.read_schema(str(path))
    cols: list[str] = list(schema.names)

    return path, _map_columns(cols), cols
