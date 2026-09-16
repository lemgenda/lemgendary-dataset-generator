"""YOLO format label parser. Extracted from compiler_core.py in Phase 1.4."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_yolo(
    txt_path: str | Path,
    img_w: int,
    img_h: int,
) -> list[dict[str, Any]]:
    """Parse a YOLO-format label file into a list of annotation dicts.

    Each returned dict:
        {"class": <str>, "bbox": [xmin, ymin, width, height]}
        optionally includes "keypoints": [x, y, v, ...]

    Pixel-space coordinates are reconstructed from YOLO's normalized
    center-x, center-y, width, height representation.
    """
    annotations: list[dict[str, Any]] = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = parts[0]
                    cx, cy, nw, nh = map(float, parts[1:5])
                    w = nw * img_w
                    h = nh * img_h
                    xmin = (cx * img_w) - (w / 2.0)
                    ymin = (cy * img_h) - (h / 2.0)
                    item: dict[str, Any] = {"class": cls_id, "bbox": [xmin, ymin, w, h]}
                    if len(parts) > 5:
                        item["keypoints"] = list(map(float, parts[5:]))
                    annotations.append(item)
    except OSError:
        # Missing / unreadable label file → return empty list. The caller
        # treats an empty result the same as "no labels available".
        pass
    return annotations