"""
Annotation format detection.

Extracted from compiler_core.py in Phase 1.4. Preserves the exact probe
order and heuristics of the original `_detect_in_subdir` and
`detect_annotations` functions.
"""

from __future__ import annotations

from pathlib import Path


def _detect_in_subdir(sub: Path) -> tuple[str | None, Path | None]:
    for f in sub.glob("*.json"):
        if "coco" in f.name.lower() or "instances" in f.name.lower():
            return "coco", f
    for f in sub.glob("*.parquet"):
        return "parquet", f
    for ext, fmt in [("*.xml", "xml"), ("*.txt", "yolo"), ("*.npz", "npz")]:
        if any(sub.glob(ext)):
            return fmt, sub
    return None, None


def detect_annotations(path) -> tuple[str | None, Path | None]:
    path = Path(path)
    # 2026 Resilience: Multi-format annotation discovery
    for f in path.glob("*.json"):
        if "coco" in f.name.lower() or "instances" in f.name.lower():
            return "coco", f
    for f in path.glob("*.parquet"):
        return "parquet", f
    for f in path.glob("*.mat"):
        return "matlab", f

    # Check one level deeper for common structures
    candidates = [
        path / "annotations", path / "Annotations", path / "labels",
        path / "metadata", path / "data", path / "landmarks"
    ]
    for sub in candidates:
        if sub.exists():
            fmt, match = _detect_in_subdir(sub)
            if fmt is not None:
                return fmt, match

    return None, None