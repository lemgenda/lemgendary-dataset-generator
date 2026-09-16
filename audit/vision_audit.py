"""
Vision audit primitives.

Single responsibility: decide whether an image (or image/target pair, or mask)
passes the compile-time quality gates. Returns structured results with a
canonical reject code so downstream consumers (reject_log, doc_generator,
dataset_info.yaml) can build a per-manifold rejection breakdown.

Extracted from compiler_core.py in Phase 2 of the 2026 modernization roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


# ─── Canonical reject codes ─────────────────────────────────────────────────
# Every code is a stable identifier. Adding codes is safe; renaming them
# requires a registry migration.
ERR_RES_UNDERFLOW = "ERR_RES_UNDERFLOW"
ERR_RES_OVERFLOW = "ERR_RES_OVERFLOW"
ERR_BLACK_FRAME = "ERR_BLACK_FRAME"
ERR_WHITE_FRAME = "ERR_WHITE_FRAME"
ERR_ASPECT_EXTREME = "ERR_ASPECT_EXTREME"
ERR_PAIR_MISMATCH = "ERR_PAIR_MISMATCH"
ERR_PAIR_MODE_MISMATCH = "ERR_PAIR_MODE_MISMATCH"
ERR_MASK_INVALID = "ERR_MASK_INVALID"
ERR_BBOX_OUT_OF_BOUNDS = "ERR_BBOX_OUT_OF_BOUNDS"
ERR_BBOX_DEGENERATE = "ERR_BBOX_DEGENERATE"
ERR_KEYPOINT_OUT_OF_BOUNDS = "ERR_KEYPOINT_OUT_OF_BOUNDS"
ERR_NIMA_BELOW_THRESHOLD = "ERR_NIMA_BELOW_THRESHOLD"
ERR_HEADER_INVALID = "ERR_HEADER_INVALID"
ERR_TRUNCATED = "ERR_TRUNCATED"
ERR_ZERO_BYTES = "ERR_ZERO_BYTES"


# ─── Magic number signatures ────────────────────────────────────────────────
_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
    b"RIFF": "webp",       # RIFF....WEBP (checked below)
    b"II*\x00": "tiff_le",
    b"MM\x00*": "tiff_be",
    b"\x89HDF\r\n\x1a\n": "hdf5",
    b"\x93NUMPY": "numpy",  # .npy
}


@dataclass
class AuditResult:
    """Structured outcome of an audit check."""
    valid: bool
    code: str | None = None
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class VisionAuditor:
    """Per-worker auditor instance.

    Constructed once in `init_worker`; methods are called from `process_image`
    for every candidate sample. Holds no per-sample state, so instances are
    safe to share across threads.
    """

    # Minimum resolution floors per task. Kept as a dict so future tasks can
    # register their own floors without editing this class.
    _RES_FLOORS: dict[str, int] = {
        "diffusion": 512,
        "quality": 224,
        "classification": 224,
        "restoration": 224,
        "super-resolution": 224,
        "parameter_prediction": 128,
        "detection": 128,
        "segmentation": 128,
        "pose": 128,
    }

    # Aspect ratios beyond this bound are rejected. A 10:1 crop is not a
    # useful training sample for most tasks; restoration is more forgiving.
    _MAX_ASPECT: float = 10.0

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.black_threshold: float = float(cfg.get("black_threshold", 0.1))
        self.strict_ground_truth: bool = bool(cfg.get("strict_ground_truth", False))
        self.nima_threshold: float = float(cfg.get("nima_threshold", 4.0))
        self.debug_rejects: bool = bool(cfg.get("audit_debug_rejects", False))

    # ── Header verification ─────────────────────────────────────────────────
    def verify_header(self, img_bytes: bytes) -> AuditResult:
        """Confirm the byte stream starts with a known image magic number.

        Full decode is deferred to PIL; this method only rejects files whose
        first 8 bytes are unambiguously not an image.
        """
        if not img_bytes:
            return AuditResult(valid=False, code=ERR_ZERO_BYTES, reason="empty byte stream")
        head = img_bytes[:12]
        for sig, fmt in _MAGIC.items():
            if head.startswith(sig):
                if fmt == "webp":
                    # RIFF container must carry WEBP in bytes 8..12
                    if head[8:12] != b"WEBP":
                        continue
                return AuditResult(valid=True, details={"format": fmt})
        return AuditResult(
            valid=False,
            code=ERR_HEADER_INVALID,
            reason=f"unrecognized magic: {head[:8].hex()}",
        )

    # ── Image-level audit ───────────────────────────────────────────────────
    def audit_image(
        self,
        img: Image.Image,
        task: str,
        slug: str,
        strict_resolution: bool = True,
    ) -> AuditResult:
        """Check resolution, black-frame, and aspect-ratio gates."""
        w, hgt = img.size

        if strict_resolution:
            floor = self._RES_FLOORS.get(task, 128)
            # 'artifact' slugs accept a lower floor (128px) as an intentional
            # escape hatch used by small restoration datasets.
            if "artifact" in slug.lower():
                floor = 128
            if w < floor or hgt < floor:
                return AuditResult(
                    valid=False,
                    code=ERR_RES_UNDERFLOW,
                    reason=f"{w}x{hgt} below {floor}px floor for task={task}",
                    details={"w": w, "h": hgt, "floor": floor},
                )

        # Black-frame gate — skipped for LAION/AVA because those sources
        # intentionally include extreme-exposure reference material.
        if task == "quality" and "laion" not in slug and "ava" not in slug:
            if self._is_black(img):
                return AuditResult(
                    valid=False,
                    code=ERR_BLACK_FRAME,
                    reason=f">{1.0 - self.black_threshold:.0%} of pixels near-black",
                )

        # Aspect-ratio guard
        aspect = max(w, hgt) / max(1, min(w, hgt))
        if aspect > self._MAX_ASPECT:
            return AuditResult(
                valid=False,
                code=ERR_ASPECT_EXTREME,
                reason=f"aspect {aspect:.2f} exceeds {self._MAX_ASPECT}",
            )

        return AuditResult(valid=True, details={"w": w, "h": hgt, "mode": img.mode})

    def _is_black(self, img: Image.Image) -> bool:
        """Return True when the near-black pixel ratio exceeds the threshold."""
        thumb = img.resize((64, 64), Image.Resampling.NEAREST) if img.size[0] > 64 else img
        gray = np.asarray(thumb.convert("L"))
        black_ratio = float(np.sum(gray < 10) / gray.size)
        return (black_ratio > (1.0 - self.black_threshold))

    # ── Pair audit (restoration / SR / segmentation) ────────────────────────
    def audit_pairs(
        self,
        img: Image.Image,
        tgt: Image.Image,
    ) -> AuditResult:
        """Confirm paired images have matching dimensions and modes."""
        if img.size != tgt.size:
            return AuditResult(
                valid=False,
                code=ERR_PAIR_MISMATCH,
                reason=f"dimension mismatch: {img.size} vs {tgt.size}",
            )
        if img.mode != tgt.mode:
            return AuditResult(
                valid=False,
                code=ERR_PAIR_MODE_MISMATCH,
                reason=f"mode mismatch: {img.mode} vs {tgt.mode}",
            )
        return AuditResult(valid=True)

    def audit_mask(self, mask: Image.Image) -> AuditResult:
        """Confirm a mask is single-channel and carries at least two classes."""
        if mask.mode not in ("L", "1", "P"):
            return AuditResult(
                valid=False,
                code=ERR_MASK_INVALID,
                reason=f"mask mode {mask.mode} not in (L, 1, P)",
            )
        arr = np.asarray(mask.convert("L"))
        unique = np.unique(arr)
        if unique.size < 2:
            return AuditResult(
                valid=False,
                code=ERR_MASK_INVALID,
                reason=f"mask has {unique.size} class(es); need >= 2",
            )
        return AuditResult(valid=True, details={"classes": unique.size})

    # ── Bounding box audit ──────────────────────────────────────────────────
    def audit_bbox(
        self,
        bbox: tuple[float, float, float, float],
        img_w: int,
        img_h: int,
    ) -> AuditResult:
        """Reject degenerate or out-of-bounds boxes.

        bbox is (x, y, w, h) in pixel space. Both origin and extents must be
        non-negative; the box must lie fully within the image; width and
        height must exceed zero.
        """
        x, y, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            return AuditResult(
                valid=False,
                code=ERR_BBOX_DEGENERATE,
                reason=f"non-positive extent {bw}x{bh}",
            )
        if x < 0 or y < 0 or x + bw > img_w or y + bh > img_h:
            return AuditResult(
                valid=False,
                code=ERR_BBOX_OUT_OF_BOUNDS,
                reason=f"box [{x},{y},{bw},{bh}] outside {img_w}x{img_h}",
            )
        return AuditResult(valid=True)

    def audit_keypoints(
        self,
        points: list[float],
        img_w: int,
        img_h: int,
        stride: int = 3,
    ) -> AuditResult:
        """Confirm every (x, y) pair lies within the image bounds.

        `points` is a flat list; stride=2 for pure coordinate pairs, stride=3
        when a visibility flag follows each pair (COCO pose format).
        """
        n = len(points) // stride
        for i in range(n):
            x = points[i * stride]
            y = points[i * stride + 1]
            if x < 0 or x > img_w or y < 0 or y > img_h:
                return AuditResult(
                    valid=False,
                    code=ERR_KEYPOINT_OUT_OF_BOUNDS,
                    reason=f"keypoint {i} = ({x},{y}) outside {img_w}x{img_h}",
                )
        return AuditResult(valid=True)

    # ── NIMA quality gate ───────────────────────────────────────────────────
    def passes_nima(
        self,
        task: str,
        slug: str,
        is_authenticity: bool,
        nima_score: float,
        nima_probs: list[float],
        current_threshold: float,
        idx: int,
    ) -> bool:
        """Apply the NIMA quality gate.

        Retains the pre-Phase-2 semantics:
          - In strict_ground_truth mode, human-label-only quality manifolds
            reject samples whose NIMA distribution is the default fallback
            (i.e. no human rating was found for them).
          - Below-threshold scores are rejected unless the sample is part of
            an authenticity dataset (where distribution shape matters more
            than scalar score).
        """
        # Strict ground truth: quality manifolds outside LAION must carry a
        # real human rating; the default fallback (all mass on bin 1) is
        # treated as "no rating available."
        if (
            nima_probs[0] == 1.0
            and task in ("quality", "diffusion")
            and not is_authenticity
        ):
            if self.strict_ground_truth and task == "quality" and "laion" not in slug:
                return False

        if (
            task in ("quality", "diffusion")
            and nima_score < current_threshold
            and not is_authenticity
        ):
            if self.debug_rejects and idx < 5:
                print(f"[AUDIT] {slug} idx={idx} nima={nima_score} < {current_threshold}")
            return False

        return True