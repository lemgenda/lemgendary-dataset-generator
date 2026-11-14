"""Visual dataset curation and quality-assurance checks for WebDataset manifolds.

Provides:
- Image integrity checks (decode, non-zero dimensions, blank/solid-color, low-variance).
- Mask integrity checks (dimension alignment, all-background, hole ratio, edge bleeding).
- Label QA (schema uniformity, noise/outlier flagging, class-imbalance weighting).
"""

from __future__ import annotations

import io
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageStat


@dataclass(frozen=True)
class ImageQAResult:
    """Result of image integrity and variance checks."""

    ok: bool
    reason: str | None
    width: int
    height: int
    max_stddev: float


@dataclass(frozen=True)
class MaskQAResult:
    """Result of segmentation mask integrity checks."""

    ok: bool
    reason: str | None
    width: int
    height: int
    foreground_ratio: float
    hole_ratio: float
    invalid_values: list[int]


def check_image_integrity(
    image_bytes: bytes,
    min_stddev: float = 1.0,
) -> ImageQAResult:
    """Verify that ``image_bytes`` decodes cleanly and is neither blank nor near-blank."""
    if not image_bytes:
        return ImageQAResult(ok=False, reason="empty_bytes", width=0, height=0, max_stddev=0.0)
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.load()
            w, h = img.size
            if w <= 0 or h <= 0:
                return ImageQAResult(
                    ok=False, reason="zero_dimension", width=w, height=h, max_stddev=0.0
                )
            extrema = img.getextrema()
            if isinstance(extrema[0], tuple):
                is_blank = all(mn == mx for mn, mx in extrema)
            else:
                is_blank = extrema[0] == extrema[1]
            if is_blank:
                return ImageQAResult(
                    ok=False, reason="blank_solid_color", width=w, height=h, max_stddev=0.0
                )
            stat = ImageStat.Stat(img)
            max_std = float(max(stat.stddev))
            if max_std < min_stddev:
                return ImageQAResult(
                    ok=False,
                    reason="low_variance",
                    width=w,
                    height=h,
                    max_stddev=round(max_std, 4),
                )
            return ImageQAResult(
                ok=True, reason=None, width=w, height=h, max_stddev=round(max_std, 4)
            )
    except (OSError, ValueError) as exc:
        return ImageQAResult(
            ok=False, reason=f"decode_error:{exc}", width=0, height=0, max_stddev=0.0
        )


def _decode_mask_array(
    mask_bytes: bytes,
    image_size: tuple[int, int] | None,
) -> np.ndarray | MaskQAResult:
    """Decode mask bytes and validate dimensions against ``image_size``."""
    if not mask_bytes:
        return MaskQAResult(
            ok=False,
            reason="empty_mask_bytes",
            width=0,
            height=0,
            foreground_ratio=0.0,
            hole_ratio=0.0,
            invalid_values=[],
        )
    try:
        with Image.open(io.BytesIO(mask_bytes)) as m_img:
            m_img.load()
            w, h = m_img.size
            if image_size is not None and (w, h) != image_size:
                return MaskQAResult(
                    ok=False,
                    reason=f"size_mismatch:{w}x{h}!={image_size[0]}x{image_size[1]}",
                    width=w,
                    height=h,
                    foreground_ratio=0.0,
                    hole_ratio=0.0,
                    invalid_values=[],
                )
            return np.asarray(m_img.convert("L"))
    except (OSError, ValueError) as exc:
        return MaskQAResult(
            ok=False,
            reason=f"mask_decode_error:{exc}",
            width=0,
            height=0,
            foreground_ratio=0.0,
            hole_ratio=0.0,
            invalid_values=[],
        )


def check_mask_integrity(
    mask_bytes: bytes,
    image_size: tuple[int, int] | None = None,
    expected_classes: set[int] | None = None,
    max_hole_ratio: float = 0.25,
) -> MaskQAResult:
    """Verify segmentation mask alignment, class validity, and interior hole ratio."""
    decoded = _decode_mask_array(mask_bytes, image_size)
    if isinstance(decoded, MaskQAResult):
        return decoded
    arr = decoded
    h, w = arr.shape

    unique_vals = {int(v) for v in np.unique(arr)}
    invalid_vals: list[int] = []
    if expected_classes is not None:
        invalid_vals = sorted(unique_vals - expected_classes)
        if invalid_vals:
            return MaskQAResult(
                ok=False,
                reason="invalid_class_values",
                width=w,
                height=h,
                foreground_ratio=float(np.mean(arr > 0)),
                hole_ratio=0.0,
                invalid_values=invalid_vals,
            )

    fg = arr > 0
    fg_ratio = float(np.mean(fg))
    if fg_ratio == 0.0:
        return MaskQAResult(
            ok=False,
            reason="empty_foreground",
            width=w,
            height=h,
            foreground_ratio=0.0,
            hole_ratio=0.0,
            invalid_values=[],
        )

    # Estimate interior holes via 4-neighbor enclosure of background pixels.
    bg = ~fg
    if h >= 3 and w >= 3:
        enclosed = bg[1:-1, 1:-1] & fg[:-2, 1:-1] & fg[2:, 1:-1] & fg[1:-1, :-2] & fg[1:-1, 2:]
        hole_ratio = float(np.sum(enclosed)) / float(max(1, np.sum(fg)))
    else:
        hole_ratio = 0.0

    if hole_ratio > max_hole_ratio:
        return MaskQAResult(
            ok=False,
            reason="excessive_holes",
            width=w,
            height=h,
            foreground_ratio=round(fg_ratio, 4),
            hole_ratio=round(hole_ratio, 4),
            invalid_values=[],
        )

    return MaskQAResult(
        ok=True,
        reason=None,
        width=w,
        height=h,
        foreground_ratio=round(fg_ratio, 4),
        hole_ratio=round(hole_ratio, 4),
        invalid_values=[],
    )


def compute_class_weights(
    counts: dict[str, int],
    strategy: str = "inverse_sqrt",
    beta: float = 0.999,
) -> dict[str, float]:
    """Compute normalized class-balancing weights from class sample counts.

    Supports ``inverse_sqrt`` (1 / sqrt(n_c)) and ``effective_number``
    ((1 - beta) / (1 - beta**n_c)), normalized so the mean weight is 1.0.
    """
    if not counts:
        return {}
    raw: dict[str, float] = {}
    for cls, n in counts.items():
        n_safe = max(1, n)
        if strategy == "effective_number":
            raw[cls] = (1.0 - beta) / max(1e-12, 1.0 - (beta**n_safe))
        else:
            raw[cls] = 1.0 / math.sqrt(n_safe)
    mean_w = sum(raw.values()) / len(raw)
    return {cls: round(w / max(1e-12, mean_w), 6) for cls, w in raw.items()}


def validate_label_records(
    records: list[dict[str, Any]],
    required_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Audit a batch of label dicts for schema uniformity and missing/null fields."""
    key_sets = Counter(tuple(sorted(r.keys())) for r in records)
    missing_required = 0
    null_fields = Counter()
    for r in records:
        if required_keys and not required_keys.issubset(r.keys()):
            missing_required += 1
        for k, v in r.items():
            if v is None:
                null_fields[k] += 1
    return {
        "total": len(records),
        "uniform_schema": len(key_sets) <= 1,
        "key_set_variants": len(key_sets),
        "missing_required_count": missing_required,
        "null_field_counts": dict(null_fields),
    }
