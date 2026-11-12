"""JSON label sidecars derived from a sample's text label, by named strategy.

A manifold whose registry entry sets ``label_json: <strategy>`` gets a
``<key>.json`` member next to every ``<key>.txt`` label when its shards are
packed. Adding a strategy means adding a builder to
:data:`LABEL_JSON_BUILDERS`; the shard writers never change.

``face_landmarks`` reads RetinaFace-style YOLO lines
``cls cx cy w h [lx0 ly0 ... lx4 ly4]`` (all normalised to 0..1) and writes
the layout the training suite's container decoder reads
(``data/sample_labels.face_target_from_json``)::

    {"face_count": 1,
     "objects": [{"category_id": 0,
                  "bbox_cxcywh_normalized": [cx, cy, w, h],
                  "landmarks_normalized": [lx0, ly0, ..., lx4, ly4]}]}

``landmarks_normalized`` (left eye, right eye, nose, left and right mouth
corner) is present only when the line carries all ten coordinates and they
are not all zero: the Kaggle RetinaFace source writes ten zeros for a face
whose landmarks were not detected, and the training suite masks absent
landmarks out instead of training towards the image corner.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = [
    "FACE_LANDMARK_VALUES",
    "LABEL_JSON_BUILDERS",
    "build_label_json",
    "face_landmarks_json",
]

FACE_LANDMARK_VALUES = 10
_BOX_END = 5

LabelJsonBuilder = Callable[[str], dict[str, Any] | None]


def _face_object(values: list[float]) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "category_id": int(values[0]),
        "bbox_cxcywh_normalized": values[1:_BOX_END],
    }
    landmarks = values[_BOX_END:_BOX_END + FACE_LANDMARK_VALUES]
    if len(landmarks) == FACE_LANDMARK_VALUES and any(landmarks):
        obj["landmarks_normalized"] = landmarks
    return obj


def face_landmarks_json(text: str) -> dict[str, Any] | None:
    """Face objects of a RetinaFace YOLO label; None when no line has a class and a box.

    Raises ``ValueError`` when a line holds a non-numeric value.
    """
    objects = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < _BOX_END:
            continue
        objects.append(_face_object([float(p) for p in parts]))
    if not objects:
        return None
    return {"face_count": len(objects), "objects": objects}


LABEL_JSON_BUILDERS: dict[str, LabelJsonBuilder] = {
    "face_landmarks": face_landmarks_json,
}


def build_label_json(strategy: str, text: str) -> dict[str, Any] | None:
    """The JSON sidecar ``strategy`` derives from label ``text`` (None: nothing to write).

    Raises ``KeyError`` for an unknown strategy and ``ValueError`` for a
    label the strategy cannot read.
    """
    return LABEL_JSON_BUILDERS[strategy](text)
