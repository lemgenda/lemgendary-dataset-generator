"""COCO annotation parser. Extracted from compiler_core.py in Phase 1.4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_coco(
    json_path: str | Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    """Parse a COCO JSON file into (images_by_id, annotations_by_image_id).

    Returns:
        images_by_id:            mapping of image id -> raw image metadata dict
        annotations_by_image_id: mapping of image id -> list of raw annotation dicts
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    images: dict[int, dict[str, Any]] = {x["id"]: x for x in data.get("images", [])}
    anns: dict[int, list[dict[str, Any]]] = {}
    for a in data.get("annotations", []):
        img_id = a["image_id"]
        if img_id not in anns:
            anns[img_id] = []
        anns[img_id].append(a)
    return images, anns