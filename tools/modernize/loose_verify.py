"""Pre-purge barrier: prove every loose file landed inside the written tar shards.

``purge_loose_dirs`` deletes the loose ``images/``, ``targets/``, ``masks/``
and ``labels/`` folders. Before that is allowed, :func:`missing_loose_members`
reads the member names of every shard actually on disk and lists each loose
file whose expected member is absent. Rejected (corrupt) samples are excluded
because they were deliberately not written; orphan masks or labels that have
no image are reported as missing, so they block the purge.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from tools.modernize.common import LooseSplit
from utils.sample_group import MASK_SUFFIX, TARGET_SUFFIX

__all__ = ["expected_member_names", "missing_loose_members", "shard_member_names"]

_WEBP = ".webp"


def shard_member_names(shard_dir: Path) -> set[str]:
    """Return the names of the regular members of every tar shard directly in ``shard_dir``."""
    names: set[str] = set()
    for shard in sorted(shard_dir.glob("shard-*")):
        if not shard.is_file():
            continue
        with tarfile.open(shard, "r") as tf:
            names.update(m.name for m in tf.getmembers() if m.isfile())
    return names


def _image_ext(path: Path, transcode_webp: bool) -> str:
    return _WEBP if transcode_webp else path.suffix.lower()


def expected_member_names(split: LooseSplit, transcode_webp: bool) -> dict[str, Path]:
    """Map each tar member name the split must contain to the loose file it came from."""
    expected: dict[str, Path] = {}
    if split.target_only:
        for stem, path in split.targets.items():
            expected[f"{stem}{_image_ext(path, transcode_webp)}"] = path
    else:
        for path in split.images:
            expected[f"{path.stem}{_image_ext(path, transcode_webp)}"] = path
        for stem, path in split.targets.items():
            expected[f"{stem}{TARGET_SUFFIX}{_image_ext(path, transcode_webp)}"] = path
    for stem, path in split.masks.items():
        expected[f"{stem}{MASK_SUFFIX}{_WEBP}"] = path
    for stem, path in split.labels.items():
        expected[f"{stem}{path.suffix.lower()}"] = path
    return expected


def missing_loose_members(
    split: LooseSplit, shard_dir: Path, transcode_webp: bool, rejected: set[str]
) -> list[Path]:
    """Return the loose files of ``split`` whose member is not in the shards of ``shard_dir``."""
    present = shard_member_names(shard_dir)
    return sorted(
        path
        for name, path in expected_member_names(split, transcode_webp).items()
        if name not in present and path.stem not in rejected
    )
