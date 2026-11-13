"""Class masks from per-file source masks (``repair masks``).

For each photo the source mask (an 8-bit PNG of class ids, the scheme of
:mod:`core.source_labels`) is checked and stored as ``<key>.mask.webp``:

- it must be a single-channel index image (``L`` or ``P``) of the photo's
  size whose values are all below the scheme's ``num_classes``;
- it is encoded with the shard writers' mask encoder
  (:func:`utils.image_payload.encode_mask_payload`, lossless WebP), and the
  encoded mask is decoded again and must hold exactly the same class ids
  (when writing; a dry run checks the first point only).

The JSON label gets ``num_classes``, ``class_names`` and ``mask_source``
(other keys are kept). A sample that already has its ``.mask.webp`` and
these keys is unchanged.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from core.source_labels import MaskScheme
from tools.repair.source_label_fix import LABELLED, UNCHANGED, SampleInput, SampleUpdate
from utils.image_payload import CorruptImageError, encode_mask_payload
from utils.sample_group import MASK_SUFFIX

__all__ = ["INVALID_MASK", "MASK_MEMBER", "MaskLabeler", "decode_mask"]

MASK_MEMBER = f"{MASK_SUFFIX}.webp"
INVALID_MASK = "invalid mask"
_INDEX_MODES = ("L", "P")


def decode_mask(payload: bytes) -> np.ndarray:
    """Class ids ``[H, W]`` (uint8) of an encoded mask, as the training suite reads it (``L``)."""
    with Image.open(io.BytesIO(payload)) as img:
        return np.asarray(img.convert("L"))


def _source_ids(raw: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(raw)) as img:
        if img.mode not in _INDEX_MODES:
            raise CorruptImageError(f"mode {img.mode} is not a class-id image")
        return np.asarray(img if img.mode == "L" else Image.frombytes("L", img.size,
                                                                      img.tobytes()))


@dataclass(frozen=True)
class MaskLabeler:
    """Stores each photo's source mask losslessly and records the class scheme."""

    scheme: MaskScheme
    source: str

    @property
    def file_suffix(self) -> str:
        """Suffix of the source mask files."""
        return self.scheme.file_suffix

    def _fields(self) -> dict[str, Any]:
        return {"num_classes": self.scheme.num_classes,
                "class_names": list(self.scheme.class_names),
                "mask_source": self.source}

    def info_fields(self) -> dict[str, Any]:
        """dataset_info.yaml ``classes``: the scheme's class names."""
        return {"classes": list(self.scheme.class_names)}

    def _checked_ids(self, sample: SampleInput) -> tuple[bytes, np.ndarray]:
        """The source mask bytes and class ids; ``CorruptImageError`` when unusable."""
        raw = sample.require_label_file().read_bytes()
        ids = _source_ids(raw)
        if (ids.shape[1], ids.shape[0]) != sample.size:
            raise CorruptImageError(f"mask {ids.shape[1]}x{ids.shape[0]},"
                                    f" image {sample.size[0]}x{sample.size[1]}")
        if int(ids.max()) >= self.scheme.num_classes:
            raise CorruptImageError(f"class id {int(ids.max())} >= {self.scheme.num_classes}")
        return raw, ids

    @staticmethod
    def _encoded(raw: bytes, ids: np.ndarray, name: str) -> bytes:
        """The lossless WebP mask; ``CorruptImageError`` when the round trip changes an id."""
        payload, _ = encode_mask_payload(raw, name)
        if not np.array_equal(decode_mask(payload), ids):
            raise CorruptImageError("the WebP round trip changed class ids")
        return payload

    def label(self, sample: SampleInput) -> SampleUpdate:
        """The ``.mask.webp`` member and JSON label of ``sample``."""
        fields = self._fields()
        meta = dict(sample.meta or {})
        if MASK_MEMBER in sample.suffixes and all(meta.get(k) == v for k, v in fields.items()):
            return SampleUpdate(UNCHANGED)
        try:
            raw, ids = self._checked_ids(sample)
            members = ({MASK_MEMBER: self._encoded(raw, ids, sample.require_label_file().name)}
                       if sample.encode else {})
        except (OSError, CorruptImageError) as exc:
            return SampleUpdate(INVALID_MASK, problem=str(exc))
        meta.update(fields)
        present = np.count_nonzero(np.bincount(ids.ravel(), minlength=self.scheme.num_classes))
        return SampleUpdate(LABELLED, meta=meta, members=members,
                            metrics={"classes_present": float(present)})
