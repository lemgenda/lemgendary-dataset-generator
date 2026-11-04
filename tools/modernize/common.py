"""Shared helpers of the modernization tools: image files, YAML metadata, cleanup."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from utils.container_stream import dump_yaml_file, transcode_to_webp

logger = logging.getLogger(__name__)

IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LOOSE_SUBDIRS: tuple[str, ...] = ("images", "targets", "masks")
SPLITS: tuple[str, ...] = ("train", "val", "test")

# Errors raised while reading, mutating or writing a YAML metadata file.
YAML_UPDATE_ERRORS: tuple[type[BaseException], ...] = (OSError, yaml.YAMLError, TypeError)


def merge_into_yaml(path: Path, updates: dict[str, Any]) -> None:
    """Set every key of ``updates`` in the YAML mapping at ``path`` and rewrite it.

    Raises one of :data:`YAML_UPDATE_ERRORS` when the file cannot be read,
    is not a mapping, or cannot be written.
    """
    with open(path, "r", encoding="utf-8") as f_in:
        data = yaml.safe_load(f_in) or {}
    for key, value in updates.items():
        data[key] = value
    dump_yaml_file(path, data)


def update_dataset_info(target_dir: Path, updates: dict[str, Any], summary: str) -> None:
    """Merge ``updates`` into ``target_dir/dataset_info.yaml`` when it exists.

    ``summary`` completes the ``[METADATA] Updated dataset_info.yaml ...``
    progress line. A failure is logged and reported, never raised.
    """
    info_path = target_dir / "dataset_info.yaml"
    if not info_path.exists():
        return
    try:
        merge_into_yaml(info_path, updates)
        print(f"[METADATA] Updated {info_path.name} {summary}.")
    except YAML_UPDATE_ERRORS as exc:
        logger.warning("Failed updating %s: %s", info_path, exc)
        print(f"[WARN] Failed updating dataset_info.yaml: {exc}")


def purge_loose_dirs(source_dir: Path, notice: str) -> None:
    """Delete the loose images/targets/masks folders of ``source_dir``."""
    print(f"[CLEANUP] {notice}")
    for sub in LOOSE_SUBDIRS:
        sub_p = source_dir / sub
        if sub_p.exists():
            shutil.rmtree(sub_p, ignore_errors=True)
    print("[SUCCESS] Loose directories purged.")


def encode_image(path: Path, quality: int, transcode_webp: bool) -> tuple[str, bytes, str]:
    """Read ``path``; return (stem, payload, extension), WebP-transcoded on request."""
    raw = path.read_bytes()
    if transcode_webp:
        out_b, out_ext = transcode_to_webp(raw, path.suffix, quality=quality)
        return path.stem, out_b, out_ext
    return path.stem, raw, path.suffix.lower()


def _image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


@dataclass
class LooseSplit:
    """The image and target files of one split of a loose directory."""

    name: str
    images: list[Path] = field(default_factory=list)
    targets: dict[str, Path] = field(default_factory=dict)

    @property
    def total_samples(self) -> int:
        """Number of samples: images, or targets for a target-only split."""
        return len(self.images) if self.images else len(self.targets)

    def out_dir(self, container_dir: Path) -> Path:
        """Return the output folder of this split under ``container_dir``."""
        return container_dir if self.name == "all" else container_dir / self.name


def loose_split_names(source_dir: Path) -> list[str]:
    """Return the split names of a loose directory (``["all"]`` when unsplit)."""
    images_dir = source_dir / "images"
    targets_dir = source_dir / "targets"
    has_splits = any((images_dir / s).exists() or (targets_dir / s).exists() for s in SPLITS)
    return list(SPLITS) if has_splits else ["all"]


def scan_loose_split(source_dir: Path, split: str) -> LooseSplit:
    """Collect the image files and stem->target map of one split."""
    images_dir = source_dir / "images"
    targets_dir = source_dir / "targets"
    if split != "all":
        images_dir = images_dir / split
        targets_dir = targets_dir / split
    return LooseSplit(
        name=split,
        images=_image_files(images_dir),
        targets={p.stem: p for p in _image_files(targets_dir)},
    )
