"""Vision manifold sampling and copying for :mod:`tools.manifold_reduce`.

Split out of ``tools/manifold_reduce.py``: picks a slug-balanced sample of each
split and copies it (with its labels, targets/images and masks) into the
reduced manifold until the per-split byte budget is spent.
"""

from __future__ import annotations

import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from core.compiler_core import DATASETS_META
from tools.manifold_reduce_select import sample_count

__all__ = ["VALID_EXTS", "copy_reduced_vision"]

VALID_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


@dataclass(frozen=True)
class _CopyLayout:
    """Source/target roots of one vision reduction and the primary sample folder."""

    source_root: Path
    target_root: Path
    primary_dir_name: str
    task_type: str


def _slug_of(path: Path) -> str:
    """Return the source slug encoded as the second ``_`` field of a sample name."""
    try:
        return path.name.split('_')[1]
    except IndexError:
        return "unknown"


def _count_files(directory: Path) -> int:
    """Number of files directly in ``directory`` (0 when it does not exist)."""
    return sum(1 for _ in directory.iterdir() if _.is_file()) if directory.exists() else 0


def _train_share(source_root: Path, primary_dir_name: str) -> float:
    """Fraction of the size budget given to the train split (by sample count)."""
    train_count = _count_files(source_root / primary_dir_name / "train")
    val_count = _count_files(source_root / primary_dir_name / "val")
    total_count = train_count + val_count
    return train_count / total_count if total_count > 0 else 1.0


def _task_type(source_root: Path) -> str:
    """Task of the registry dataset whose name appears in the manifold folder name."""
    for v in DATASETS_META.values():
        if v["name"] in source_root.name:
            return v.get("task", "quality")
    return "quality"


def _interleave_by_slug(all_imgs: list[Path]) -> list[Path]:
    """Shuffle each slug's samples and round-robin across slugs for a balanced order."""
    images_by_slug: defaultdict[str, list[Path]] = defaultdict(list)
    for img in all_imgs:
        images_by_slug[_slug_of(img)].append(img)

    for slug in images_by_slug:
        random.shuffle(images_by_slug[slug])

    sampled_imgs = []
    lists = list(images_by_slug.values())
    while lists:
        lists = [lst for lst in lists if lst]
        for lst in lists:
            sampled_imgs.append(lst.pop())
    return sampled_imgs


def _sample_split(layout: _CopyLayout, split: str, fraction: float | None) -> list[Path]:
    """Return the ordered, fraction-limited sample list of one split ([] if none)."""
    primary_split_dir = layout.source_root / layout.primary_dir_name / split
    if not primary_split_dir.exists():
        return []
    all_imgs = [p for p in primary_split_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    if not all_imgs:
        return []
    sampled_imgs = _interleave_by_slug(all_imgs)
    return sampled_imgs[:sample_count(len(sampled_imgs), fraction)]


def _copy_if_exists(src: Path, dest_dir: Path) -> int:
    """Copy ``src`` into ``dest_dir`` when it exists; return the bytes written."""
    if not src.exists():
        return 0
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    return dest.stat().st_size


def _copy_sample(layout: _CopyLayout, split: str, img_path: Path) -> int:
    """Copy one sample and its companions into the reduced manifold; return bytes written."""
    src, dst = layout.source_root, layout.target_root
    file_size = _copy_if_exists(img_path, dst / layout.primary_dir_name / split)
    file_size += _copy_if_exists(
        src / "labels" / split / (img_path.stem + ".txt"), dst / "labels" / split
    )

    # The companion of a primary image is its target, and vice versa.
    companion = "targets" if layout.primary_dir_name == "images" else "images"
    file_size += _copy_if_exists(src / companion / split / img_path.name, dst / companion / split)

    mask_dir = src / "masks" / split
    mask_path = mask_dir / img_path.name
    if not mask_path.exists():
        mask_path = mask_dir / (img_path.stem + ".png")
    file_size += _copy_if_exists(mask_path, dst / "masks" / split)
    return file_size


def _copy_split(
    layout: _CopyLayout, split: str, sampled_imgs: list[Path], split_limit_bytes: float
) -> list[dict[str, Any]]:
    """Copy samples until the split's byte budget is spent; return their index rows."""
    rows: list[dict[str, Any]] = []
    current_bytes = 0
    with tqdm(
        total=split_limit_bytes,
        desc=f"Copying {split}",
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for img_path in sampled_imgs:
            if current_bytes >= split_limit_bytes:
                break
            file_size = _copy_sample(layout, split, img_path)
            current_bytes += file_size
            pbar.update(file_size)
            rows.append({
                "name": img_path.stem,
                "split": split,
                "source": _slug_of(img_path),
                "task": layout.task_type,
            })
    return rows


def copy_reduced_vision(
    source_root: Path, target_root: Path, max_gb: float, fraction: float | None
) -> list[dict[str, Any]] | None:
    """Copy a reduced sample of ``source_root`` into ``target_root``.

    Returns the new ``index.json`` rows, or None when the user aborts (Ctrl+C).
    """
    for d in ["images", "labels", "targets", "masks"]:
        for s in ["train", "val"]:
            (target_root / d / s).mkdir(parents=True, exist_ok=True)

    primary_dir_name = "images" if (source_root / "images").exists() else "targets"
    layout = _CopyLayout(source_root, target_root, primary_dir_name, _task_type(source_root))
    train_prob = _train_share(source_root, primary_dir_name)

    new_index: list[dict[str, Any]] = []
    for split in ["train", "val"]:
        sampled_imgs = _sample_split(layout, split, fraction)
        if not sampled_imgs:
            continue

        split_limit_bytes = (
            max_gb * (1024**3) * (train_prob if split == "train" else (1 - train_prob))
        )
        try:
            new_index.extend(_copy_split(layout, split, sampled_imgs, split_limit_bytes))
        except KeyboardInterrupt:
            print("\n[ABORTED] Reduction cancelled by user.")
            return None
    return new_index
