"""Parquet schema mapper. Copied verbatim from compiler_core.py in Phase 1.4."""

from __future__ import annotations


def parse_parquet(pq_path):
    import pyarrow.parquet as pq
    schema = pq.read_schema(pq_path)
    cols = schema.names
    # Detect common schemas
    mapping = {}
    if "image" in cols: mapping["file_name"] = "image"
    elif "pixel_values" in cols: mapping["file_name"] = "pixel_values"
    if "url" in cols: mapping["url"] = "url"
    if "key" in cols: mapping["key"] = "key"
    if "label" in cols: mapping["class"] = "label"

    # Restoration Targets
    if "target" in cols: mapping["target"] = "target"
    if "sharp" in cols: mapping["target"] = "sharp"
    if "ground_truth" in cols: mapping["target"] = "ground_truth"

    # Additional mappings for bbox/seg if needed
    for c in cols:
        cl = c.lower()
        if any(x in cl for x in ["xmin", "x1"]): mapping["xmin"] = c
        if any(x in cl for x in ["ymin", "y1"]): mapping["ymin"] = c
        if any(x in cl for x in ["width", "w"]): mapping["width"] = c
        if any(x in cl for x in ["height", "h"]): mapping["height"] = c
    return pq_path, mapping, cols