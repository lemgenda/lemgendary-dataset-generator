"""YOLO format label parser. Copied verbatim from compiler_core.py in Phase 1.4."""

from __future__ import annotations


def parse_yolo(txt_path, img_w, img_h):
    annotations = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = parts[0]
                    # YOLO is center_x, center_y, width, height (normalized)
                    cx, cy, nw, nh = map(float, parts[1:5])
                    w = nw * img_w
                    h = nh * img_h
                    xmin = (cx * img_w) - (w / 2.0)
                    ymin = (cy * img_h) - (h / 2.0)
                    item = {"class": cls_id, "bbox": [xmin, ymin, w, h]}
                    if len(parts) > 5:
                        kpts = list(map(float, parts[5:]))
                        item["keypoints"] = kpts
                    annotations.append(item)
    except Exception:
        pass
    return annotations