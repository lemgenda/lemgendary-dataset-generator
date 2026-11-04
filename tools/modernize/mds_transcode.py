"""Transcode legacy MosaicML MDS shards (.mds.zstd) into WebDataset tar shards."""

from __future__ import annotations

import io
import itertools
import json
import shutil
import struct
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zstandard as zstd
from tqdm import tqdm

from tools.modernize.common import SPLITS, update_dataset_info
from tools.modernize.queue_items import _fast_has_tar_shards

TarMembers = list[tuple[str, bytes]]


def _detect_img_ext(b: bytes) -> str:
    """Sniff the image container from its magic bytes (defaults to webp)."""
    if b.startswith(b"RIFF") and len(b) > 12 and b[8:12] == b"WEBP":
        return "webp"
    if b.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if b.startswith(b"\x89PNG"):
        return "png"
    return "webp"


def _mds_sample_to_tar_members(fields: dict[str, bytes | None], key: str) -> TarMembers:
    """Map one decoded MDS sample to its WebDataset tar members (name, payload)."""
    members: TarMembers = []
    for field_name, suffix in (("image", ""), ("target", ".target"), ("mask", ".mask")):
        payload = fields.get(field_name)
        if payload:
            members.append((f"{key}{suffix}.{_detect_img_ext(payload)}", payload))

    meta_dict: dict[str, Any] = {}
    for meta_key in ("label", "split", "task"):
        raw = fields.get(meta_key)
        if raw:
            meta_dict[meta_key] = raw.decode("utf-8", errors="replace")
    if meta_dict:
        members.append((f"{key}.json", json.dumps(meta_dict, ensure_ascii=False).encode("utf-8")))
    return members


def _decode_sample(columns: list[str], s_bytes: bytes) -> dict[str, bytes | None]:
    """Split one raw MDS sample into its column payloads (None when empty)."""
    num_cols = len(columns)
    col_sizes = struct.unpack(f"<{num_cols}I", s_bytes[:num_cols * 4])
    offset = num_cols * 4
    fields: dict[str, bytes | None] = {}
    for col, sz in zip(columns, col_sizes):
        fields[col] = s_bytes[offset:offset + sz] if sz > 0 else None
        offset += sz
    return fields


def _iter_shard_samples(raw_data: bytes) -> Iterator[bytes]:
    """Yield the raw bytes of every sample of a decompressed MDS shard."""
    shard_samples = struct.unpack("<I", raw_data[:4])[0]
    sample_offsets = struct.unpack(
        f"<{shard_samples + 1}I", raw_data[4 : 4 + (shard_samples + 1) * 4]
    )
    for idx in range(shard_samples):
        yield raw_data[sample_offsets[idx]:sample_offsets[idx + 1]]


def _iter_mds_tar_members(
    split_mds_dir: Path,
    shards_meta: list[dict[str, Any]],
    dctx: zstd.ZstdDecompressor,
) -> Iterator[TarMembers]:
    """Yield the tar members of every sample in a split's MDS shards, in order."""
    sample_counter = 0
    for s_info in shards_meta:
        columns = s_info["column_names"]
        mds_file = split_mds_dir / s_info["zip_data"]["basename"]
        if not mds_file.exists():
            raise FileNotFoundError(f"MDS shard file not found: {mds_file}")

        for s_bytes in _iter_shard_samples(dctx.decompress(mds_file.read_bytes())):
            fields = _decode_sample(columns, s_bytes)
            raw_name = fields.get("name")
            key = raw_name.decode("utf-8") if raw_name else f"sample_{sample_counter:08d}"
            sample_counter += 1
            yield _mds_sample_to_tar_members(fields, key)


def _add_tar_members(tf: tarfile.TarFile, members: TarMembers) -> None:
    """Append in-memory (name, payload) members to an open tar archive."""
    for name, payload in members:
        ti = tarfile.TarInfo(name=name)
        ti.size = len(payload)
        tf.addfile(ti, io.BytesIO(payload))


def _find_mds_splits(mds_dir: Path) -> list[str] | None:
    """Return the MDS splits that have an index.json (``["all"]`` if unsplit), else None."""
    found = [s for s in SPLITS if (mds_dir / s).exists() and (mds_dir / s / "index.json").exists()]
    if found:
        return found
    if (mds_dir / "index.json").exists():
        return ["all"]
    return None


@dataclass(frozen=True)
class _ShardTarget:
    """Where and how a split's WebDataset shards are written."""

    out_dir: Path
    ext: str
    shard_size_samples: int


def _write_shards(
    samples: Iterator[TarMembers], target: _ShardTarget, pbar: Any
) -> tuple[int, int]:
    """Write ``samples`` into numbered tar shards; return (samples, shards) written.

    Each shard is written under its own context manager and is only created
    once there is at least one sample to put in it.
    """
    written = 0
    shards = 0
    first_members = next(samples, None)
    while first_members is not None:
        shard_path = target.out_dir / f"shard-{shards:05d}.{target.ext}"
        with tarfile.open(shard_path, "w") as tf:
            for members in itertools.chain(
                [first_members], itertools.islice(samples, max(target.shard_size_samples - 1, 0))
            ):
                _add_tar_members(tf, members)
                written += 1
                pbar.update(1)
        shards += 1
        first_members = next(samples, None)
    return written, shards


def _transcode_split(
    split: str, mds_dir: Path, target: _ShardTarget, dctx: zstd.ZstdDecompressor
) -> tuple[int, int]:
    """Transcode one MDS split; return (samples, shards) written."""
    split_mds_dir = mds_dir / split if split != "all" else mds_dir
    with open(split_mds_dir / "index.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    shards_meta = meta.get("shards", [])
    expected = sum(s["samples"] for s in shards_meta)
    print(
        f"[MDS-TRANSCODE] Split '{split}': {len(shards_meta)} MDS shards,"
        f" {expected:,} samples expected."
    )
    samples = _iter_mds_tar_members(split_mds_dir, shards_meta, dctx)
    with tqdm(
        total=expected, desc=f"MDS -> WEBDATASET [{split}]", unit="img", dynamic_ncols=True
    ) as pbar:
        written, shards = _write_shards(samples, target, pbar)
    print(
        f"[MDS-TRANSCODE] Split '{split}' complete: {written:,} samples ->"
        f" {shards} shards."
    )
    return written, shards


def _verify_shards(shards_dir: Path, total_shards: int) -> bool:
    """Confirm the output shards exist before the legacy MDS folder may be purged."""
    has_valid_shards, total_shard_bytes = _fast_has_tar_shards(shards_dir)
    if not has_valid_shards or total_shard_bytes == 0:
        print(
            "[ERROR] Shard verification failed: output shards directory is empty. Aborting MDS"
            " cleanup."
        )
        return False
    print(
        f"[VERIFIED] Created {total_shards} WebDataset shards"
        f" ({total_shard_bytes / (1024**3):.1f} GB total)."
    )
    return True


def transcode_mds_to_webdataset(
    target_dir: Path,
    shard_size_samples: int = 1000,
    shard_ext: str = "data",
    purge_mds: bool = True,
) -> bool:
    """Transcode legacy MosaicML MDS shards (.mds.zstd) directly into WebDataset (.data) shards."""
    mds_dir = target_dir / "mds"
    if not mds_dir.exists():
        print(f"[ERROR] MDS directory does not exist: {mds_dir}")
        return False

    shards_dir = target_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    ext = shard_ext.lstrip(".")

    found_splits = _find_mds_splits(mds_dir)
    if found_splits is None:
        print(f"[ERROR] No valid MDS index.json discovered in {mds_dir}")
        return False

    total_samples = 0
    total_shards = 0
    dctx = zstd.ZstdDecompressor()
    for split in found_splits:
        split_out_dir = shards_dir / split if split != "all" else shards_dir
        split_out_dir.mkdir(parents=True, exist_ok=True)
        written, shards = _transcode_split(
            split, mds_dir, _ShardTarget(split_out_dir, ext, shard_size_samples), dctx
        )
        total_samples += written
        total_shards += shards

    update_dataset_info(
        target_dir,
        {
            "format": "webdataset",
            "canonical_format": "webdataset",
            "shard_extension": ext,
            "total_samples": total_samples,
            "total_shards": total_shards,
        },
        f"format to webdataset ({ext} shards)",
    )

    if not _verify_shards(shards_dir, total_shards):
        return False

    if purge_mds:
        print(f"[CLEANUP] Safely purging legacy MDS directory {mds_dir.name} to reclaim storage...")
        shutil.rmtree(mds_dir, ignore_errors=True)
        print("[SUCCESS] Legacy MDS directory purged.")
    return True
