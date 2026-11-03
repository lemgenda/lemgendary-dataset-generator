"""YOLO format label parser. Extracted from compiler_core.py in Phase 1.4."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _parse_line(line: str, img_w: int, img_h: int) -> dict[str, Any] | None:
    """Convert one YOLO label line to an annotation dict, or None if too short."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cx, cy, nw, nh = map(float, parts[1:5])
    w = nw * img_w
    h = nh * img_h
    xmin = (cx * img_w) - (w / 2.0)
    ymin = (cy * img_h) - (h / 2.0)
    item: dict[str, Any] = {"class": parts[0], "bbox": [xmin, ymin, w, h]}
    if len(parts) > 5:
        item["keypoints"] = list(map(float, parts[5:]))
    return item


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
                item = _parse_line(line, img_w, img_h)
                if item is not None:
                    annotations.append(item)
    except OSError as exc:
        # Missing / unreadable label file → log diagnostic and return empty list.
        logger.debug("Label file unreadable or absent at %s: %s", txt_path, exc)
    return annotations
