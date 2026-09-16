"""Parquet schema mapper. Extracted from compiler_core.py in Phase 1.4."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_parquet(
    pq_path: str | Path,
) -> tuple[Path, dict[str, Any], list[str]]:
    """Inspect a Parquet file's schema and return a column-name mapping.

    Returns:
        pq_path:  the resolved path (as `Path`)
        mapping:  canonical-role -> actual-column-name (e.g. "file_name" -> "image")
        cols:     the schema's full column-name list, in declaration order
    """
    import pyarrow.parquet as pq

    path = Path(pq_path)
    schema = pq.read_schema(str(path))
    cols: list[str] = list(schema.names)

    mapping: dict[str, Any] = {}
    if "image" in cols:
        mapping["file_name"] = "image"
    elif "pixel_values" in cols:
        mapping["file_name"] = "pixel_values"
    if "url" in cols:
        mapping["url"] = "url"
    if "key" in cols:
        mapping["key"] = "key"
    if "label" in cols:
        mapping["class"] = "label"

    # Restoration targets
    if "target" in cols:
        mapping["target"] = "target"
    if "sharp" in cols:
        mapping["target"] = "sharp"
    if "ground_truth" in cols:
        mapping["target"] = "ground_truth"

    # Additional mappings for bbox/seg if needed
    for c in cols:
        cl = c.lower()
        if any(x in cl for x in ["xmin", "x1"]):
            mapping["xmin"] = c
        if any(x in cl for x in ["ymin", "y1"]):
            mapping["ymin"] = c
        if any(x in cl for x in ["width", "w"]):
            mapping["width"] = c
        if any(x in cl for x in ["height", "h"]):
            mapping["height"] = c

    return path, mapping, cols