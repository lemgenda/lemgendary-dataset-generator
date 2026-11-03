"""
LemGendary Dataset Compiler — Shared container-streaming helpers.

Small building blocks shared by the zip/loose-directory to WebDataset/MDS
packagers (`tools/stream_zip_to_container`, `tools/modernization_queue`):
WebP transcoding, worker-count resolution, output-directory preparation,
loose-layout MDS writer/record construction, and `dataset_info.yaml`
format stamping.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from formats.mds import LOOSE_MDS_COLUMNS

logger = logging.getLogger(__name__)

__all__ = [
    "IMAGE_WEBP_QUALITY",
    "TARGET_WEBP_QUALITY",
    "dump_yaml_file",
    "loose_mds_record",
    "open_loose_mds_writer",
    "prepare_mds_split_dir",
    "prepare_shards_dir",
    "resolve_worker_count",
    "transcode_to_webp",
    "update_dataset_info_format",
]

IMAGE_WEBP_QUALITY = 92
TARGET_WEBP_QUALITY = 95


def transcode_to_webp(raw_bytes: bytes, ext: str, quality: int = 92) -> tuple[bytes, str]:
    """Transcode image bytes to WebP format using Pillow with libwebp method 2."""
    if ext.lower() == ".webp":
        return raw_bytes, ".webp"
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            if img.mode in ("RGBA", "LA", "P"):
                if img.mode == "P":
                    img = img.convert("RGBA")
                bg = Image.new("RGB", img.size, (255, 255, 255))
                channels = img.split()
                mask = channels[-1] if len(channels) > 3 else None
                bg.paste(img, mask=mask)
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

            out = io.BytesIO()
            img.save(out, format="WEBP", quality=quality, method=2)
            return out.getvalue(), ".webp"
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        logger.warning("WebP transcode failed (%s); keeping original %s bytes", exc, ext)
        return raw_bytes, ext


def resolve_worker_count(max_workers: int | None) -> int:
    """Return `max_workers`, or a CPU-based default capped at 12 threads."""
    return max_workers or min(12, os.cpu_count() or 4)


def prepare_shards_dir(target_dir: Path) -> Path:
    """Create `target_dir` and its `shards/` subfolder; return the shards folder."""
    target_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = target_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    return shards_dir


def prepare_mds_split_dir(split_mds_dir: Path) -> str:
    """Wipe and recreate an MDS split folder; return the `out` string for MDSWriter.

    mosaicml-streaming parses `out` with urllib.parse.urlparse, so an absolute
    Windows path (C:\\...) is mistaken for a URL scheme; a relative path routes
    to LocalUploader. Falls back to the absolute path when no relative path
    exists (different drive). The folder is wiped first because LocalUploader
    refuses non-empty output directories from interrupted runs.
    """
    try:
        mds_out_str = os.path.relpath(split_mds_dir)
    except ValueError:
        mds_out_str = str(split_mds_dir)
    if split_mds_dir.exists():
        shutil.rmtree(split_mds_dir)
    split_mds_dir.mkdir(parents=True, exist_ok=True)
    return mds_out_str


def open_loose_mds_writer(mds_writer_cls: Any, out: str, size_limit: int) -> Any:
    """Construct an MDSWriter for the loose images/targets column schema."""
    return mds_writer_cls(
        out=out,
        columns=LOOSE_MDS_COLUMNS,
        compression="zstd",
        size_limit=size_limit,
        hashes=[],
    )


def loose_mds_record(split: str, name: str, image: bytes, target: bytes = b"") -> dict[str, Any]:
    """Build one restoration-task sample row for the loose MDS column schema."""
    return {
        "image": image,
        "target": target,
        "mask": b"",
        "label": "",
        "task": "restoration",
        "split": split,
        "name": name,
    }


def dump_yaml_file(path: Path, data: Any) -> None:
    """Write `data` as block-style, order-preserving, unicode YAML to `path`."""
    with open(path, "w", encoding="utf-8") as f_out:
        yaml.safe_dump(
            data,
            f_out,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def update_dataset_info_format(ds_info_path: Path, fmt: str, transcode_webp: bool) -> bool:
    """Stamp `format`/`canonical_format` (and `image_format: webp`) into dataset_info.yaml.

    Returns True when the file existed and was rewritten. A missing file returns
    False silently; a read/write failure is logged, printed as a warning and
    returns False.
    """
    if not ds_info_path.exists():
        return False
    try:
        with open(ds_info_path, "r", encoding="utf-8") as f_in:
            info_data = yaml.safe_load(f_in) or {}
        info_data["format"] = fmt
        info_data["canonical_format"] = fmt
        if transcode_webp:
            info_data["image_format"] = "webp"
        dump_yaml_file(ds_info_path, info_data)
    except (OSError, yaml.YAMLError, TypeError) as exc:
        logger.warning("Failed updating %s: %s", ds_info_path, exc)
        print(f"[WARN] Failed updating dataset_info.yaml: {exc}")
        return False
    return True
