"""Shared helpers of the modernization tools: loose files, YAML metadata, cleanup."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from utils.container_stream import dump_yaml_file
from utils.sample_group import LABEL_EXTS

logger = logging.getLogger(__name__)

IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LOOSE_SUBDIRS: tuple[str, ...] = ("images", "targets", "masks", "labels")
# Loose folders whose files only survive as companions inside the shards.
COMPANION_SUBDIRS: tuple[str, ...] = ("masks", "labels")
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


def purge_loose_dirs(
    source_dir: Path, notice: str, subdirs: tuple[str, ...] = LOOSE_SUBDIRS
) -> bool:
    """Delete the loose ``subdirs`` folders of ``source_dir``; return True when all went."""
    print(f"[CLEANUP] {notice}")
    purged = True
    for sub in subdirs:
        sub_p = source_dir / sub
        if not sub_p.exists():
            continue
        try:
            shutil.rmtree(sub_p)
        except OSError as exc:
            purged = False
            logger.error("Could not purge loose folder %s: %s", sub_p, exc)
            print(f"[WARN] Could not purge {sub_p.name}: {exc}")
    if purged:
        print("[SUCCESS] Loose directories purged.")
    return purged


def _files_with(directory: Path, exts: set[str] | tuple[str, ...]) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts)


@dataclass
class LooseSplit:
    """The image, target, mask and label files of one split of a loose directory."""

    name: str
    images: list[Path] = field(default_factory=list)
    targets: dict[str, Path] = field(default_factory=dict)
    masks: dict[str, Path] = field(default_factory=dict)
    labels: dict[str, Path] = field(default_factory=dict)

    @property
    def total_samples(self) -> int:
        """Number of samples: images, or targets for a target-only split."""
        return len(self.images) if self.images else len(self.targets)

    @property
    def target_only(self) -> bool:
        """True when the split has targets but no images (targets are primary)."""
        return not self.images

    def primary_files(self) -> list[Path]:
        """Return the files that each start one sample (images, else targets)."""
        return self.images if self.images else list(self.targets.values())

    def out_dir(self, container_dir: Path) -> Path:
        """Return the output folder of this split under ``container_dir``."""
        return container_dir if self.name == "all" else container_dir / self.name


def loose_split_names(source_dir: Path) -> list[str]:
    """Return the split names of a loose directory (``["all"]`` when unsplit)."""
    has_splits = any((source_dir / sub / s).exists() for sub in LOOSE_SUBDIRS for s in SPLITS)
    return list(SPLITS) if has_splits else ["all"]


def _split_dir(source_dir: Path, sub: str, split: str) -> Path:
    base = source_dir / sub
    return base if split == "all" else base / split


def _stem_map(
    source_dir: Path, sub: str, split: str, exts: set[str] | tuple[str, ...]
) -> dict[str, Path]:
    return {p.stem: p for p in _files_with(_split_dir(source_dir, sub, split), exts)}


def scan_loose_split(source_dir: Path, split: str) -> LooseSplit:
    """Collect the images and the stem maps of targets, masks and labels of one split."""
    return LooseSplit(
        name=split,
        images=_files_with(_split_dir(source_dir, "images", split), IMAGE_EXTS),
        targets=_stem_map(source_dir, "targets", split, IMAGE_EXTS),
        masks=_stem_map(source_dir, "masks", split, IMAGE_EXTS),
        labels=_stem_map(source_dir, "labels", split, LABEL_EXTS),
    )
