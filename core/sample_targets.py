"""Locate the ground-truth target (sharp / clean / mask) paired with a degraded input image.

Restoration-style tasks pair every input with a target. The target comes
from the annotation row when the source is tabular, otherwise from sibling
folders and file names on disk (``blur/`` next to ``sharp/``, ``rain_01.png``
next to ``norain_01.png``), and finally from the DPED device-to-Canon mapping.
"""

import io
import logging
import os
from pathlib import Path
from typing import Any

from PIL import Image

from core.compiler_worker import WORKER

logger = logging.getLogger(__name__)

TARGET_TASKS = ("restoration", "super-resolution", "parameter_prediction", "segmentation")

_ROW_TARGET_KEYS = ("target", "sharp", "ground_truth", "gt", "clean", "original", "mask", "masks")
_BLUR_KEYS = ("blur", "blurry", "input", "lowres", "lr", "rain",
              "hazy", "noisy", "degraded", "distorted", "low", "images")
_SHARP_KEYS = ("sharp", "gt", "ground_truth", "groundtruth", "clean",
               "clear", "original", "hr", "highres", "target", "norain",
               "high", "targets", "mask", "masks", "segmentation",
               "segmentations")
_SPLIT_KEYS = ("train", "test", "val", "images")
_DPED_DEVICES = ("iphone", "sony", "blackberry")


def _annotation_row(ann_data: Any) -> dict[str, Any] | None:
    """Return the annotation row dict for a dict or a (DataFrame, mapping) pair."""
    if isinstance(ann_data, dict):
        return ann_data
    if isinstance(ann_data, tuple) and len(ann_data) == 2:
        df_sub, _ = ann_data
        if not df_sub.empty:
            return df_sub.iloc[0].to_dict()
    return None


def _target_from_row(ann_data: Any) -> tuple[Image.Image | None, str | None]:
    """Read the target from the first target-like column of the annotation row."""
    row_dict = _annotation_row(ann_data) if ann_data else None
    if not row_dict:
        return None, None
    for k in _ROW_TARGET_KEYS:
        val = row_dict.get(k)
        if isinstance(val, bytes):
            return Image.open(io.BytesIO(val)), None
        if isinstance(val, str) and val.endswith((".png", ".jpg", ".jpeg")):
            p = Path(val)
            return None, (str(p) if p.exists() else None)
    return None, None


def _is_sharp_dir(path: Path) -> bool:
    """True for a directory whose name marks it as holding targets."""
    return path.is_dir() and any(k in path.name.lower() for k in _SHARP_KEYS)


def _renamed_candidates(img_path: Path, folder: Path, skip_same: bool) -> str | None:
    """Swap a blur-style file-name prefix for each sharp-style one and test it in ``folder``."""
    lower = img_path.name.lower()
    for b_k in _BLUR_KEYS:
        if not lower.startswith(b_k):
            continue
        for s_k in _SHARP_KEYS:
            new_name = lower.replace(b_k, s_k, 1)
            if skip_same and new_name == lower:
                continue
            if (folder / new_name).exists():
                return str(folder / new_name)
    return None


def _search_sharp_siblings(img_path: Path) -> str | None:
    """Look for the target in sharp-named siblings of a blur-named parent folder."""
    parent = img_path.parent
    if not any(k in parent.name.lower() for k in _BLUR_KEYS):
        return None
    try:
        for sibling in parent.parent.iterdir():
            if not _is_sharp_dir(sibling):
                continue
            potential = sibling / img_path.name
            if potential.exists():
                return str(potential)
            found = _renamed_candidates(img_path, sibling, skip_same=False)
            if found:
                return found
    except OSError as exc:
        logger.debug("Target sibling search OSError on %s: %s", parent, exc)
    return None


def _search_ancestor_siblings(img_path: Path) -> str | None:
    """Look for the target beside a split-named ancestor (``train/``, ``images/`` ...)."""
    parent = img_path.parent
    for ancestor in [parent, parent.parent]:
        if not any(k in ancestor.name.lower() for k in _SPLIT_KEYS):
            continue
        try:
            for sibling in ancestor.parent.iterdir():
                if not _is_sharp_dir(sibling):
                    continue
                potential = sibling / img_path.name
                if potential.exists():
                    return str(potential)
                try:
                    potential_rel = sibling / img_path.relative_to(ancestor)
                    if potential_rel.exists():
                        return str(potential_rel)
                except ValueError as exc:
                    logger.debug("Relative path computation failed: %s", exc)
        except OSError as exc:
            logger.debug("Ancestor sibling search OSError on %s: %s", ancestor, exc)
    return None


def _search_dped(img_path: Path) -> str | None:
    """Map a DPED phone capture to its Canon reference frame."""
    p_str = str(img_path).replace("\\", "/").lower()
    for device_name in _DPED_DEVICES:
        needle = f"/{device_name}/"
        if needle in p_str:
            tgt_p_str = p_str.replace(needle, "/canon/")
            if WORKER.dped_cache and tgt_p_str in WORKER.dped_cache:
                return tgt_p_str
            if os.path.exists(tgt_p_str):
                return tgt_p_str
            return None
    return None


def resolve_target(task: str, ann_data: Any, img_path: Path,
                   is_virtual: bool) -> tuple[Image.Image | None, str | None]:
    """Return ``(target_image, target_path)`` for a sample; both None when there is none."""
    if task not in TARGET_TASKS:
        return None, None
    target_img, target_img_path = _target_from_row(ann_data)
    if target_img or target_img_path or is_virtual:
        return target_img, target_img_path
    return None, (
        _search_sharp_siblings(img_path)
        or _search_ancestor_siblings(img_path)
        or _renamed_candidates(img_path, img_path.parent, skip_same=True)
        or _search_dped(img_path)
    )
