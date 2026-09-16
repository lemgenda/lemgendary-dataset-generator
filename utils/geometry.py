"""Bounding-box and keypoint coordinate transforms. Extracted in Phase 1.5.5."""

from __future__ import annotations


def convert_bbox_xywh_to_yolo(bbox, w, h):
    """Convert [x, y, w, h] pixel coords to YOLO-normalized [cx, cy, w, h]."""
    x, y, bw, bh = bbox
    return [
        round((x + bw / 2) / w, 6),
        round((y + bh / 2) / h, 6),
        round(bw / w, 6),
        round(bh / h, 6),
    ]


def normalize_points(points, w, h, stride=2):
    """Normalize a flat [x0, y0, x1, y1, ...] list. stride=3 handles pose (x,y,visibility)."""
    norm = []
    for i in range(0, len(points), stride):
        norm.append(round(points[i] / w, 6))
        norm.append(round(points[i + 1] / h, 6))
        if stride == 3:
            norm.append(points[i + 2])
    return norm