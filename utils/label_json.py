"""JSON label documents: the one label format inside WebDataset shards (SSOT).

Every labelled sample in a shard carries its label as ``<key>.json``; there
are no ``.txt`` label members. Masks stay ``<key>.mask.webp``. A text label
(a loose ``labels/<split>/<key>.txt`` written by the compiler or shipped by
a Kaggle source) becomes its JSON document through the manifold's label
strategy: the registry entry's ``label_json``, else the strategy of its
task (:data:`TASK_STRATEGIES`). The JSON may carry other keys as well
(``source``, ``hash``, ``style`` ...); the label keys are these, per
strategy (all coordinates normalised to 0..1, boxes as ``cx cy w h``):

``face_landmarks`` (face detection, RetinaFace)::

    {"face_count": 1,
     "objects": [{"category_id": 0,
                  "bbox_cxcywh_normalized": [cx, cy, w, h],
                  "landmarks_normalized": [lx0, ly0, ..., lx4, ly4]}]}

  ``landmarks_normalized`` (left eye, right eye, nose, left and right mouth
  corner) is present only when the line carries ten coordinates that are not
  all zero: the Kaggle RetinaFace source writes ten zeros for a face whose
  landmarks were not detected, and the training suite masks absent
  landmarks out (``data/sample_labels.face_target_from_json``). A label
  rebuilt from source landmarks (``repair face-landmarks``) also records
  ``landmark_source`` (for example ``sfhq-110pt``).

``yolo_boxes`` (detection, pose and the auto-labelled restoration tasks)::

    {"object_count": 2,
     "objects": [{"category_id": 3, "bbox_cxcywh_normalized": [cx, cy, w, h]},
                 {"category_id": 0, "bbox_cxcywh_normalized": [cx, cy, w, h],
                  "keypoints": [x0, y0, v0, ...]}]}

  ``keypoints`` holds the values after the box when a line has any (pose
  lines). An empty label is ``{"object_count": 0, "objects": []}``: an
  image without objects, a valid detection negative.

``yolo_polygons`` (segmentation polygons)::

    {"object_count": 1,
     "objects": [{"category_id": 1, "polygon_normalized": [x0, y0, x1, y1, ...]}]}

``segmentation_mask`` (face parsing, ParseNet): the class mask is the
``<key>.mask.webp`` member (lossless, one class id per pixel); the JSON
records its scheme::

    {"num_classes": 19, "class_names": ["background", "skin", ...],
     "mask_source": "sfhq-bisenet"}

  A mask has no text label: a non-empty text label is rejected.

``nima`` (quality)::

    {"nima_scores": {"distribution": [p1, ..., p10], "mean_score": 5.42}}

``class_label`` (classification): ``{"class_id": 3}`` for a numeric label,
``{"class_name": "cat"}`` for a named one.

``caption`` (diffusion): ``{"caption": "a red house by a lake"}``.

:func:`label_document` builds a document from label text,
:func:`label_problems` validates one (the shard verifier,
:mod:`utils.label_audit`, runs it on every ``.json`` member).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "FACE_LANDMARK_VALUES",
    "LABEL_JSON_BUILDERS",
    "LABEL_STRATEGIES",
    "NIMA_BINS",
    "TASK_STRATEGIES",
    "LabelStrategy",
    "build_label_json",
    "face_landmarks_json",
    "label_document",
    "label_problems",
    "resolve_label_strategy",
]

FACE_LANDMARK_VALUES = 10
NIMA_BINS = 10
_BOX_END = 5
_BOX_VALUES = 4
_POLYGON_MIN_VALUES = 6
_BOX_KEY = "bbox_cxcywh_normalized"
_LANDMARK_KEY = "landmarks_normalized"
_POLYGON_KEY = "polygon_normalized"

LabelJsonBuilder = Callable[[str], dict[str, Any] | None]


def _numbers(line: str) -> list[float]:
    """The values of one label line; raises ``ValueError`` for a non-numeric value."""
    values = [float(p) for p in line.split()]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"non-finite value in label line {line!r}")
    return values


def _category(value: float) -> int:
    if value != int(value) or value < 0:
        raise ValueError(f"class id {value} is not a non-negative integer")
    return int(value)


def _box_lines(text: str) -> list[list[float]]:
    """Numeric lines with at least a class and a box."""
    lines = []
    for line in text.splitlines():
        values = _numbers(line)
        if not values:
            continue
        if len(values) < _BOX_END:
            raise ValueError(f"label line {line!r} has fewer than 5 values")
        lines.append(values)
    return lines


# ─── Builders: label text -> JSON document ─────────────────────────────────
def _face_object(values: list[float]) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "category_id": _category(values[0]),
        _BOX_KEY: values[1:_BOX_END],
    }
    landmarks = values[_BOX_END:_BOX_END + FACE_LANDMARK_VALUES]
    if len(landmarks) == FACE_LANDMARK_VALUES and any(landmarks):
        obj[_LANDMARK_KEY] = landmarks
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
        objects.append(_face_object(_numbers(line)))
    if not objects:
        return None
    return {"face_count": len(objects), "objects": objects}


def _yolo_boxes_json(text: str) -> dict[str, Any]:
    objects = []
    for values in _box_lines(text):
        obj: dict[str, Any] = {"category_id": _category(values[0]),
                               _BOX_KEY: values[1:_BOX_END]}
        if len(values) > _BOX_END:
            obj["keypoints"] = values[_BOX_END:]
        objects.append(obj)
    return {"object_count": len(objects), "objects": objects}


def _yolo_polygons_json(text: str) -> dict[str, Any]:
    objects = []
    for line in text.splitlines():
        values = _numbers(line)
        if not values:
            continue
        points = values[1:]
        if len(points) < _POLYGON_MIN_VALUES or len(points) % 2:
            raise ValueError(f"polygon line {line!r} needs at least 3 x,y points")
        objects.append({"category_id": _category(values[0]), _POLYGON_KEY: points})
    return {"object_count": len(objects), "objects": objects}


def _segmentation_mask_json(text: str) -> dict[str, Any] | None:
    if not text.strip():
        return None
    raise ValueError("a segmentation mask has no text label (the mask is <key>.mask.webp)")


def _nima_json(text: str) -> dict[str, Any] | None:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    if len(lines) > 1:
        raise ValueError(f"a NIMA label has one line, got {len(lines)}")
    distribution = _numbers(lines[0])
    if len(distribution) != NIMA_BINS:
        raise ValueError(f"expected {NIMA_BINS} NIMA values, got {len(distribution)}")
    mean_score = sum((i + 1) * p for i, p in enumerate(distribution))
    return {"nima_scores": {"distribution": distribution, "mean_score": round(mean_score, 4)}}


def _class_label_json(text: str) -> dict[str, Any] | None:
    label = text.strip()
    if not label:
        return None
    if "\n" in label:
        raise ValueError("a class label has one line")
    try:
        number = float(label)
    except ValueError:
        return {"class_name": label}
    return {"class_id": _category(number)}


def _caption_json(text: str) -> dict[str, Any] | None:
    caption = text.strip()
    return {"caption": caption} if caption else None


# ─── Validators: JSON document -> problems ─────────────────────────────────
def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _number_list(value: Any, length: int | None = None) -> bool:
    return (isinstance(value, list) and all(_is_number(v) for v in value)
            and (length is None or len(value) == length))


def _object_problems(index: int, obj: Any) -> list[str]:
    where = f"objects[{index}]"
    if not isinstance(obj, Mapping):
        return [f"{where} is not an object"]
    problems = []
    category = obj.get("category_id")
    if not isinstance(category, int) or isinstance(category, bool) or category < 0:
        problems.append(f"{where}.category_id is not a non-negative integer")
    box = obj.get(_BOX_KEY)
    polygon = obj.get(_POLYGON_KEY)
    if box is None and polygon is None:
        problems.append(f"{where} has neither {_BOX_KEY} nor {_POLYGON_KEY}")
    if box is not None and not (_number_list(box, _BOX_VALUES)
                                and all(0.0 <= v <= 1.0 for v in box)):
        problems.append(f"{where}.{_BOX_KEY} is not 4 numbers in 0..1")
    if polygon is not None and not (_number_list(polygon) and len(polygon) >= _POLYGON_MIN_VALUES
                                    and len(polygon) % 2 == 0):
        problems.append(f"{where}.{_POLYGON_KEY} is not an even list of >= 6 numbers")
    landmarks = obj.get(_LANDMARK_KEY)
    if landmarks is not None and not _number_list(landmarks, FACE_LANDMARK_VALUES):
        problems.append(f"{where}.{_LANDMARK_KEY} is not {FACE_LANDMARK_VALUES} numbers")
    keypoints = obj.get("keypoints")
    if keypoints is not None and not _number_list(keypoints):
        problems.append(f"{where}.keypoints is not a list of numbers")
    return problems


def _objects_problems(doc: Mapping[str, Any]) -> list[str]:
    objects = doc.get("objects")
    if objects is None:
        return []
    if not isinstance(objects, list):
        return ["objects is not a list"]
    problems = [p for i, obj in enumerate(objects) for p in _object_problems(i, obj)]
    for count_key in ("face_count", "object_count"):
        if count_key in doc and doc[count_key] != len(objects):
            problems.append(f"{count_key} {doc[count_key]!r} != {len(objects)} objects")
    return problems


def _scalar_problems(doc: Mapping[str, Any]) -> list[str]:
    problems = []
    nima = doc.get("nima_scores")
    if nima is not None:
        if not isinstance(nima, Mapping) or not _number_list(nima.get("distribution"), NIMA_BINS):
            problems.append(f"nima_scores.distribution is not {NIMA_BINS} numbers")
        elif "mean_score" in nima and not _is_number(nima["mean_score"]):
            problems.append("nima_scores.mean_score is not a number")
    class_id = doc.get("class_id")
    if class_id is not None and (not isinstance(class_id, int) or isinstance(class_id, bool)
                                 or class_id < 0):
        problems.append("class_id is not a non-negative integer")
    class_name = doc.get("class_name")
    if class_name is not None and not (isinstance(class_name, str) and class_name.strip()):
        problems.append("class_name is not a non-empty string")
    caption = doc.get("caption")
    if caption is not None and not isinstance(caption, str):
        problems.append("caption is not a string")
    for source_key in ("mask_source", "landmark_source"):
        value = doc.get(source_key)
        if value is not None and not (isinstance(value, str) and value.strip()):
            problems.append(f"{source_key} is not a non-empty string")
    return problems + _mask_problems(doc)


def _mask_problems(doc: Mapping[str, Any]) -> list[str]:
    problems = []
    count = doc.get("num_classes")
    if count is not None and (not isinstance(count, int) or isinstance(count, bool)
                              or count < 1):
        problems.append("num_classes is not a positive integer")
    names = doc.get("class_names")
    if names is not None:
        if not (isinstance(names, list) and names
                and all(isinstance(n, str) and n.strip() for n in names)):
            problems.append("class_names is not a list of non-empty strings")
        elif isinstance(count, int) and len(names) != count:
            problems.append(f"class_names has {len(names)} names, num_classes is {count}")
    return problems


# ─── Strategies ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LabelStrategy:
    """How one kind of text label becomes a JSON document, and what a valid one holds.

    ``build`` returns None for a label that holds nothing; ``empty`` is then
    the document written instead (None: such a label has no document and the
    sample carries no label). ``required`` lists the keys every document of
    this strategy holds (any one of ``required`` for ``class_label``).
    """

    name: str
    build: LabelJsonBuilder
    required: tuple[str, ...]
    empty: dict[str, Any] | None = None
    any_of: bool = False

    def missing(self, doc: Mapping[str, Any]) -> list[str]:
        """Required keys absent from ``doc``."""
        present = [key for key in self.required if key in doc]
        if self.any_of:
            return [] if present else [f"one of {', '.join(self.required)}"]
        return [key for key in self.required if key not in doc]


LABEL_STRATEGIES: dict[str, LabelStrategy] = {
    strategy.name: strategy for strategy in (
        LabelStrategy("face_landmarks", face_landmarks_json, ("face_count", "objects"),
                      empty={"face_count": 0, "objects": []}),
        LabelStrategy("yolo_boxes", _yolo_boxes_json, ("object_count", "objects")),
        LabelStrategy("yolo_polygons", _yolo_polygons_json, ("object_count", "objects")),
        LabelStrategy("segmentation_mask", _segmentation_mask_json,
                      ("num_classes", "class_names")),
        LabelStrategy("nima", _nima_json, ("nima_scores",)),
        LabelStrategy("class_label", _class_label_json, ("class_id", "class_name"),
                      any_of=True),
        LabelStrategy("caption", _caption_json, ("caption",)),
    )
}

LABEL_JSON_BUILDERS: dict[str, LabelJsonBuilder] = {
    name: strategy.build for name, strategy in LABEL_STRATEGIES.items()
}

# Registry task -> the strategy of its text labels (when ``label_json`` is unset).
TASK_STRATEGIES: dict[str, str] = {
    "detection": "yolo_boxes",
    "detection-face": "face_landmarks",
    "pose": "yolo_boxes",
    "segmentation": "yolo_polygons",
    "quality": "nima",
    "classification": "class_label",
    "diffusion": "caption",
    "restoration": "yolo_boxes",
    "super-resolution": "yolo_boxes",
    "parameter_prediction": "yolo_boxes",
}


def resolve_label_strategy(label_json: str | None, task: str | None) -> str | None:
    """The registry's ``label_json``, else the strategy of ``task``; None when neither."""
    return label_json or TASK_STRATEGIES.get(task or "")


def build_label_json(strategy: str, text: str) -> dict[str, Any] | None:
    """The JSON fields ``strategy`` derives from label ``text`` (None: nothing to write).

    Raises ``KeyError`` for an unknown strategy and ``ValueError`` for a
    label the strategy cannot read.
    """
    return LABEL_STRATEGIES[strategy].build(text)


def label_document(strategy: str, text: str) -> dict[str, Any] | None:
    """The JSON document of label ``text``: built, else the strategy's empty document.

    None only for an empty label of a strategy without an empty document.
    Raises ``ValueError`` for text the strategy cannot read, including text
    that is not blank yet yields nothing (that label would be lost).
    """
    chosen = LABEL_STRATEGIES[strategy]
    doc = chosen.build(text)
    if doc is not None:
        return doc
    if text.strip():
        raise ValueError(f"{strategy} found no label in {text.strip()[:60]!r}")
    return dict(chosen.empty) if chosen.empty is not None else None


def label_problems(doc: Any, strategy: str | None = None) -> list[str]:
    """Why ``doc`` is not a valid label document (empty list: it is).

    Every label key present is checked; with ``strategy`` its required keys
    must be present too.
    """
    if not isinstance(doc, Mapping):
        return ["the JSON label is not an object"]
    problems = _objects_problems(doc) + _scalar_problems(doc)
    chosen = LABEL_STRATEGIES.get(strategy or "")
    if chosen is not None:
        problems.extend(f"missing {key}" for key in chosen.missing(doc))
    return problems
