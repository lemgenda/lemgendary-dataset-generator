"""COCO annotation parser. Copied verbatim from compiler_core.py in Phase 1.4."""

from __future__ import annotations

import json


def parse_coco(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    images = {x["id"]: x for x in data.get("images", [])}
    anns = {}
    for a in data.get("annotations", []):
        img_id = a["image_id"]
        if img_id not in anns: anns[img_id] = []
        anns[img_id].append(a)
    return images, anns