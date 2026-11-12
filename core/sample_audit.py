"""Apply the VisionAuditor gates to one sample and record every rejection.

The per-sample pipeline (:mod:`core.sample_processor`) asks this module
whether a sample's bytes, pixels, target and annotations are usable. Every
rejection is logged and, while a manifold compile has bound a
:class:`audit.reject_log.RejectLog` to ``WORKER.audit.reject_log``, written to
the registry's ``reject_log`` table.

Gates that could reject legitimate data in some sources are opt-in via
``strict_ground_truth``: pair colour-mode mismatches (an RGB input with a
palette target) and the single-channel mask check (some sources ship RGB
colour masks). Dimension mismatches, empty or unknown headers and invalid
annotations are always enforced.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from audit.vision_audit import (
    ERR_BBOX_DEGENERATE,
    ERR_PAIR_MISMATCH,
    ERR_PAIR_MODE_MISMATCH,
    AuditResult,
)
from core.compiler_worker import WORKER

logger = logging.getLogger(__name__)

__all__ = ["header_ok", "image_ok", "record_reject", "target_ok", "valid_annotations"]

# Bytes read from a file to identify its format.
_HEADER_BYTES = 12
# Slack (pixels) for boxes rebuilt from rounded YOLO floats that touch an edge.
_EDGE_SLACK = 0.5
# Pair tasks whose input and target must have the same dimensions.
_SAME_SIZE_TASKS = ("restoration", "segmentation")


def record_reject(source: str, name: str, result: AuditResult) -> None:
    """Log one rejection and persist it when a reject log is bound."""
    logger.debug("Rejected %s from %s: %s (%s)", name, source, result.code, result.reason)
    reject_log = WORKER.audit.reject_log
    if reject_log is not None:
        reject_log.record(source, name, result.code or "ERR_UNKNOWN", result.reason or "")


def _read_header(img_path: Path) -> bytes:
    with img_path.open("rb") as handle:
        return handle.read(_HEADER_BYTES)


def header_ok(source: str, name: str, data: bytes | Path) -> bool:
    """Reject empty byte streams and files whose magic number is not an image."""
    auditor = WORKER.audit.auditor
    if auditor is None:
        return True
    head = data if isinstance(data, bytes) else _read_header(data)
    result = auditor.verify_header(head)
    if not result.valid:
        record_reject(source, name, result)
    return result.valid


def image_ok(source: str, name: str, img: Image.Image, task: str) -> bool:
    """Resolution, black-frame and aspect-ratio gates."""
    auditor = WORKER.audit.auditor
    if auditor is None:
        return True
    result = auditor.audit_image(img, task, source)
    if not result.valid:
        record_reject(source, name, result)
    return result.valid


def _open_target(target: Any) -> Image.Image | None:
    if isinstance(target, Image.Image):
        return target
    if isinstance(target, (str, Path)):
        return Image.open(Path(target))
    if isinstance(target, bytes):
        return Image.open(io.BytesIO(target))
    return None


def _pair_result(task: str, img: Image.Image, tgt: Image.Image, strict: bool) -> AuditResult:
    """Dimension check always; colour-mode check for restoration under ``strict``."""
    auditor = WORKER.audit.auditor
    if auditor is None:
        return AuditResult(valid=True)
    result = auditor.audit_pairs(img, tgt)
    if result.valid or result.code == ERR_PAIR_MISMATCH:
        return result
    if result.code == ERR_PAIR_MODE_MISMATCH and strict and task == "restoration":
        return result
    return AuditResult(valid=True)


def target_ok(source: str, name: str, task: str, img: Image.Image | None, target: Any) -> bool:
    """Check a restoration target or segmentation mask against its input image.

    ``target`` is a PIL image, a path or bytes. Super-resolution pairs differ
    in size by design and are not compared; nothing is checked when the input
    pixels were not decoded.
    """
    auditor = WORKER.audit.auditor
    if auditor is None or img is None or task not in _SAME_SIZE_TASKS:
        return True
    tgt = _open_target(target)
    if tgt is None:
        return True
    strict = auditor.strict_ground_truth
    try:
        result = _pair_result(task, img, tgt, strict)
        if result.valid and task == "segmentation" and strict:
            result = auditor.audit_mask(tgt)
    finally:
        if tgt is not target:
            tgt.close()
    if not result.valid:
        record_reject(source, name, result)
    return result.valid


def _keypoint_stride(points: list[Any]) -> int:
    """3 for (x, y, visibility) triples, 2 for plain (x, y) pairs."""
    return 3 if points and len(points) % 3 == 0 else 2


def _annotation_result(ann: dict[str, Any], width: int, height: int) -> AuditResult:
    """Audit one annotation: pixel boxes against the image, normalized points against 1x1."""
    auditor = WORKER.audit.auditor
    if auditor is None:
        return AuditResult(valid=True)
    data = list(ann.get("data") or [])
    box_w, box_h = width + _EDGE_SLACK, height + _EDGE_SLACK
    if ann.get("type") == "bbox":
        return auditor.audit_bbox((data[0], data[1], data[2], data[3]), box_w, box_h)
    if ann.get("type") == "pose":
        result = auditor.audit_bbox((data[0], data[1], data[2], data[3]), box_w, box_h)
        kpts = data[4:]
        if result.valid:
            result = auditor.audit_keypoints(kpts, 1, 1, stride=_keypoint_stride(kpts))
        return result
    if ann.get("type") == "segmentation":
        return auditor.audit_keypoints(data, 1, 1, stride=2)
    return AuditResult(valid=True)


def valid_annotations(source: str, name: str, annotations: list[Any],
                      dims: tuple[int, int]) -> list[Any]:
    """Drop degenerate or out-of-bounds annotations; each drop is recorded.

    Unknown image dimensions (pixels never decoded) leave the list unchanged.
    """
    width, height = dims
    if WORKER.audit.auditor is None or width <= 0 or height <= 0:
        return annotations
    kept: list[Any] = []
    for ann in annotations:
        if len(ann.get("data") or []) < (2 if ann.get("type") == "segmentation" else 4):
            result = AuditResult(valid=False, code=ERR_BBOX_DEGENERATE,
                                 reason="annotation has too few coordinates")
        else:
            result = _annotation_result(ann, width, height)
        if result.valid:
            kept.append(ann)
        else:
            record_reject(source, name, result)
    return kept
