"""Per-manifold sample policies, chosen by the registry entry (SSOT).

A registry entry gets a policy from its label strategy (``label_json``) or
its task; :func:`sample_policy` is the one lookup. A policy sorts each
sample into a kind from its JSON sidecar (``kind_of``), drops the kinds in
``drop_kinds`` and draws val from the kinds in ``val_kinds`` (None: every
kept kind). The face detection policy (``label_json: face_landmarks``, or
task ``detection-face``):

- a sample whose label holds no face (no label, empty label,
  ``objects: []``) is dropped: a detector learns nothing from an aligned
  crop without a face;
- the val split is a seeded, deterministic share (the entry's
  ``val_split``) of the samples that carry 5-point landmarks, drawn per
  source and keyed by the source file id, so val measures the landmarks
  too and a file never sits in both splits; samples with a box but no
  landmarks stay in train.

The face parsing policy (``label_json: segmentation_mask``) drops nothing by
kind (``repair masks`` gives every photo its mask) and draws val from all
photos by source file id, so a file never sits in both splits.

For a source with a key layout the val share is drawn over all ``photos``
file ids of the layout, not over the samples a manifold kept: a val
candidate goes to val when its file id is drawn. Manifolds compiled from the
same source files (ParseNet and RetinaFace) therefore share one val set.

Key layouts (``key_layouts`` in the registry) describe manifolds compiled
before the source scan skipped auxiliary folders: per source, the sample
keys ``<Prefix>_<source>_<idx>`` run through ``copies`` duplicate sample
images, then ``photos`` real images, then auxiliary images (for SFHQ the
face-parsing maps of ``segmentations/``). Copies and auxiliary images are
dropped; a photo's file id is ``idx - copies``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from utils.naming import sample_key_source

__all__ = [
    "FACE_BOX_ONLY",
    "FACE_DETECTION",
    "FACE_LANDMARKS",
    "FACE_NONE",
    "FACE_PARSING",
    "MASK_NONE",
    "MASK_PRESENT",
    "ROLE_AUXILIARY",
    "ROLE_COPY",
    "ROLE_PHOTO",
    "KeyLayout",
    "KeyRole",
    "SamplePolicy",
    "face_kind",
    "key_role",
    "mask_kind",
    "parse_key_layouts",
    "sample_policy",
    "split_rank",
]

FACE_NONE = "no_face"
FACE_BOX_ONLY = "box_only"
FACE_LANDMARKS = "landmarks"
MASK_NONE = "no_mask"
MASK_PRESENT = "mask"

ROLE_PHOTO = "photo"
ROLE_COPY = "sample_copy"
ROLE_AUXILIARY = "auxiliary_image"

_LANDMARK_VALUES = 10


def _has_landmarks(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    values = obj.get("landmarks_normalized")
    return (isinstance(values, list) and len(values) == _LANDMARK_VALUES
            and any(isinstance(v, (int, float)) and v != 0 for v in values))


def face_kind(meta: Mapping[str, Any] | None) -> str:
    """``no_face``, ``box_only`` or ``landmarks`` for a sample's JSON sidecar."""
    objects = meta.get("objects") if meta is not None else None
    if not isinstance(objects, list) or not objects:
        return FACE_NONE
    return FACE_LANDMARKS if any(_has_landmarks(o) for o in objects) else FACE_BOX_ONLY


def mask_kind(meta: Mapping[str, Any] | None) -> str:
    """``mask`` when the sidecar records a class mask (``num_classes``), else ``no_mask``."""
    classes = meta.get("num_classes") if meta is not None else None
    present = isinstance(classes, int) and not isinstance(classes, bool) and classes > 0
    return MASK_PRESENT if present else MASK_NONE


@dataclass(frozen=True)
class SamplePolicy:
    """What a manifold keeps and how it splits.

    ``kind_of`` sorts a sample by its JSON sidecar; samples of ``drop_kinds``
    are dropped; val is drawn from ``val_kinds`` (None: every kept kind).
    """

    name: str
    kind_of: Callable[[Mapping[str, Any] | None], str]
    drop_kinds: frozenset[str]
    val_kinds: frozenset[str] | None
    seed: int = 42


FACE_DETECTION = SamplePolicy("face_detection", face_kind,
                              drop_kinds=frozenset({FACE_NONE}),
                              val_kinds=frozenset({FACE_LANDMARKS}))
FACE_PARSING = SamplePolicy("face_parsing", mask_kind, drop_kinds=frozenset(), val_kinds=None)

_BY_LABEL_JSON: dict[str, SamplePolicy] = {"face_landmarks": FACE_DETECTION,
                                           "segmentation_mask": FACE_PARSING}
_BY_TASK: dict[str, SamplePolicy] = {"detection-face": FACE_DETECTION}


def sample_policy(label_json: str | None, task: str | None = None) -> SamplePolicy | None:
    """The policy of a manifold with this label strategy or task; None when it has none."""
    return _BY_LABEL_JSON.get(label_json or "") or _BY_TASK.get(task or "")


@dataclass(frozen=True)
class KeyLayout:
    """Key index layout of one source: ``copies`` duplicates, then ``photos`` images.

    ``files_dir`` names the source's folder under a local root of its
    per-file labels (``repair masks`` / ``repair face-landmarks``).
    """

    source: str
    copies: int
    photos: int
    files_dir: str = ""


@dataclass(frozen=True)
class KeyRole:
    """What a sample key stands for: its role, source and source file id."""

    role: str
    source: str
    file_id: str


def parse_key_layouts(raw: Iterable[Mapping[str, Any]] | None) -> dict[str, KeyLayout]:
    """Registry ``key_layouts`` by lower-case source."""
    layouts: dict[str, KeyLayout] = {}
    for item in raw or ():
        layout = KeyLayout(str(item["source"]), int(item["copies"]), int(item["photos"]),
                           str(item.get("files_dir") or ""))
        layouts[layout.source.lower()] = layout
    return layouts


def key_role(key: str, layouts: Mapping[str, KeyLayout]) -> KeyRole:
    """The role of ``key`` under ``layouts`` (keys of unlisted sources are photos)."""
    base = key.rpartition("/")[2]
    source = sample_key_source(base) or ""
    idx_text = base.rpartition("_")[2]
    layout = layouts.get(source.lower())
    if layout is None or not idx_text.isdigit():
        return KeyRole(ROLE_PHOTO, source, idx_text if idx_text.isdigit() else base)
    idx = int(idx_text)
    if idx < layout.copies:
        return KeyRole(ROLE_COPY, source, str(idx))
    file_id = idx - layout.copies
    if file_id >= layout.photos:
        return KeyRole(ROLE_AUXILIARY, source, str(file_id - layout.photos))
    return KeyRole(ROLE_PHOTO, source, str(file_id))


def split_rank(policy: SamplePolicy, source: str, file_id: str) -> str:
    """Seeded, stable sort key of a source file for the val draw."""
    text = f"{policy.seed}:{source.lower()}:{file_id}"
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
