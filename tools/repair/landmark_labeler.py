"""Face boxes and 5-point landmarks from per-file source landmarks (``repair face-landmarks``).

For each photo the source landmark file (the scheme of
:mod:`core.source_labels`, SFHQ ``sfhq_110``) gives one face: the box and
the five points of :mod:`core.face_geometry` (WIDER FACE / RetinaFace
convention), checked for plausibility (eye order, eyes above nose above
mouth, every point inside the box). A photo whose landmarks fail that
check is dropped from the manifold (a failed landmark fit gives no usable
face label, and a detector must not learn it as a face without a box).

The JSON label's ``objects`` is replaced by that one face (``face_count``
1) and ``landmark_source`` is recorded; other keys are kept. The detector
boxes of the older compile are not kept: they are whole-head boxes from a
different convention, and their landmarks sat on a fixed template, so one
consistent source for every box and point is better than a mix. The plan
reports how the new labels differ from the old ones (``box_area_ratio`` new
over old area, ``box_iou``, ``landmark_nme``: mean point distance over the
new inter-ocular distance, for old faces with landmarks; ``old_faces``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.face_geometry import FaceGeometry, face_geometry, plausibility_problems
from core.source_labels import LandmarkScheme
from tools.repair.source_label_fix import (
    DROPPED,
    LABELLED,
    UNCHANGED,
    SampleInput,
    SampleUpdate,
)

__all__ = ["IMPLAUSIBLE", "INVALID_LANDMARKS", "LandmarkLabeler"]

INVALID_LANDMARKS = "invalid landmarks"
IMPLAUSIBLE = "implausible landmarks"
_BOX_KEY = "bbox_cxcywh_normalized"
_LANDMARK_KEY = "landmarks_normalized"
_LANDMARK_VALUES = 10
_BOX_VALUES = 4


def _corners(box: list[float]) -> tuple[float, float, float, float]:
    cx, cy, w, h = box
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = _corners(a)
    bx0, by0, bx1, by1 = _corners(b)
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _old_faces(meta: dict[str, Any]) -> list[dict[str, Any]]:
    objects = meta.get("objects")
    if not isinstance(objects, list):
        return []
    return [o for o in objects if isinstance(o, dict) and isinstance(o.get(_BOX_KEY), list)
            and len(o[_BOX_KEY]) == _BOX_VALUES]


def _comparison(meta: dict[str, Any], box: list[float], points: list[float],
                size: tuple[int, int]) -> dict[str, float]:
    """How the new face differs from the best-matching old one."""
    old = _old_faces(meta)
    metrics: dict[str, float] = {"old_faces": float(len(old))}
    if not old:
        return metrics
    best = max(old, key=lambda o: _iou(o[_BOX_KEY], box))
    old_box = [float(v) for v in best[_BOX_KEY]]
    if old_box[2] * old_box[3] > 0:
        metrics["box_area_ratio"] = box[2] * box[3] / (old_box[2] * old_box[3])
    metrics["box_iou"] = _iou(old_box, box)
    old_points = best.get(_LANDMARK_KEY)
    if isinstance(old_points, list) and len(old_points) == _LANDMARK_VALUES and any(old_points):
        scale = np.array(size, dtype=np.float64)
        new = np.asarray(points, dtype=np.float64).reshape(5, 2) * scale
        prev = np.asarray(old_points, dtype=np.float64).reshape(5, 2) * scale
        ocular = float(np.linalg.norm(new[1] - new[0]))
        if ocular > 0:
            metrics["landmark_nme"] = float(np.linalg.norm(new - prev, axis=1).mean()) / ocular
    return metrics


@dataclass(frozen=True)
class LandmarkLabeler:
    """Replaces each photo's face objects with the box and 5 points of its source landmarks."""

    scheme: LandmarkScheme
    source: str

    @property
    def file_suffix(self) -> str:
        """Suffix of the source landmark files."""
        return self.scheme.file_suffix

    def info_fields(self) -> dict[str, Any]:
        """dataset_info.yaml ``landmark_source``."""
        return {"landmark_source": self.source}

    def _face(self, sample: SampleInput) -> FaceGeometry:
        with np.load(sample.require_label_file(), allow_pickle=False) as data:
            points = np.asarray(data[self.scheme.array_key], dtype=np.float64)
        return face_geometry(points, self.scheme, *sample.size)

    def label(self, sample: SampleInput) -> SampleUpdate:
        """The JSON label of ``sample`` with its rebuilt face."""
        try:
            face = self._face(sample)
        except (OSError, KeyError, ValueError) as exc:
            return SampleUpdate(INVALID_LANDMARKS, problem=str(exc))
        problems = plausibility_problems(face)
        if problems:
            return SampleUpdate(DROPPED, problem=f"{IMPLAUSIBLE}: {'; '.join(problems)}")
        width, height = sample.size
        box, points = face.box_cxcywh(width, height), face.landmarks(width, height)
        meta = dict(sample.meta or {})
        objects = [{"category_id": 0, _BOX_KEY: box, _LANDMARK_KEY: points}]
        if meta.get("objects") == objects and meta.get("landmark_source") == self.source \
                and meta.get("face_count") == 1:
            return SampleUpdate(UNCHANGED)
        metrics = _comparison(meta, box, points, sample.size)
        meta.update({"face_count": 1, "objects": objects, "landmark_source": self.source})
        return SampleUpdate(LABELLED, meta=meta, metrics=metrics)
