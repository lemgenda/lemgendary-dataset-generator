"""Transcode legacy MosaicML MDS shards (.mds.zstd) into WebDataset tar shards."""

from __future__ import annotations

import io
import itertools
import json
import logging
import shutil
import struct
import tarfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zstandard as zstd
from tqdm import tqdm

from tools.modernize.common import SPLITS, update_dataset_info
from tools.modernize.queue_items import _fast_has_tar_shards
from utils.shard_audit import verify_shard_set

logger = logging.getLogger(__name__)

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


# Image columns written as their own tar members (column -> member suffix).
_IMAGE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("image", ""),
    ("target", ".target"),
    ("mask", ".mask"),
)
# Columns that only name the sample.
_KEY_COLUMNS = frozenset({"name"})
# struct formats of the MDS fixed-width numeric encodings.
_NUMERIC_FORMATS: dict[str, str] = {
    "int": "<q", "int8": "<b", "int16": "<h", "int32": "<i", "int64": "<q",
    "uint8": "<B", "uint16": "<H", "uint32": "<I", "uint64": "<Q",
    "float16": "<e", "float32": "<f", "float64": "<d",
}
_TEXT_ENCODINGS = frozenset({"str", ""})


def _decode_meta_value(encoding: str, raw: bytes) -> Any:
    """Decode one MDS metadata column payload by its column encoding.

    Raises ValueError for binary encodings (bytes, pil, pkl, ndarray, ...) that
    cannot be represented safely in a JSON sidecar.
    """
    if encoding in _TEXT_ENCODINGS:
        return raw.decode("utf-8")
    if encoding == "json":
        return json.loads(raw.decode("utf-8"))
    fmt = _NUMERIC_FORMATS.get(encoding)
    if fmt is None:
        raise ValueError(f"binary MDS encoding {encoding!r} is not JSON-representable")
    return struct.unpack(fmt, raw)[0]


def _mds_sample_to_tar_members(
    fields: Mapping[str, bytes | None], key: str, encodings: dict[str, str] | None = None
) -> TarMembers:
    """Map one decoded MDS sample to its WebDataset tar members (name, payload).

    Image columns become ``<key>[.target|.mask].<ext>`` members. Every other
    non-empty column (label, class, class_id, category, nima_scores, bbox,
    keypoints, source, split, task, ...) is decoded by its MDS encoding and
    stored in the ``<key>.json`` sidecar, so no metadata column is dropped.
    """
    encodings = encodings or {}
    members: TarMembers = []
    image_cols = {col for col, _ in _IMAGE_COLUMNS}
    for field_name, suffix in _IMAGE_COLUMNS:
        payload = fields.get(field_name)
        if payload:
            members.append((f"{key}{suffix}.{_detect_img_ext(payload)}", payload))

    meta_dict: dict[str, Any] = {}
    for col, raw in fields.items():
        if not raw or col in image_cols or col in _KEY_COLUMNS:
            continue
        try:
            meta_dict[col] = _decode_meta_value(encodings.get(col, "str"), raw)
        except (ValueError, UnicodeDecodeError, struct.error) as exc:
            logger.warning("Sample %s: column %s not copied to JSON: %s", key, col, exc)
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
        encodings = dict(zip(columns, s_info.get("column_encodings", []), strict=False))
        mds_file = split_mds_dir / s_info["zip_data"]["basename"]
        if not mds_file.exists():
            raise FileNotFoundError(f"MDS shard file not found: {mds_file}")

        for s_bytes in _iter_shard_samples(dctx.decompress(mds_file.read_bytes())):
            fields = _decode_sample(columns, s_bytes)
            raw_name = fields.get("name")
            key = raw_name.decode("utf-8") if raw_name else f"sample_{sample_counter:08d}"
            sample_counter += 1
            yield _mds_sample_to_tar_members(fields, key, encodings)


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
) -> tuple[int, int, int]:
    """Transcode one MDS split; return (samples written, shards, samples expected)."""
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
    return written, shards, expected


def _verify_shards(shards_dir: Path, total_shards: int) -> bool:
    """Confirm the output shards exist before the legacy MDS folder may be purged."""
    has_valid_shards, total_shard_bytes = _fast_has_tar_shards(shards_dir)
    if not has_valid_shards or total_shard_bytes == 0:
        print(
            "[ERROR] Shard verification failed: output shards directory is empty. Aborting MDS"
            " cleanup."
        )
        return False
    if not verify_shard_set(shards_dir, "MDS-TRANSCODE"):
        print("[ERROR] Shard verification failed: unsound shards. Aborting MDS cleanup.")
        return False
    print(
        f"[VERIFIED] Created {total_shards} WebDataset shards"
        f" ({total_shard_bytes / (1024**3):.1f} GB total)."
    )
    return True


def _transcode_splits(
    splits: list[str], mds_dir: Path, shards_dir: Path, base: _ShardTarget
) -> tuple[int, int, int]:
    """Transcode every split; return (samples written, shards, samples expected)."""
    totals = [0, 0, 0]
    dctx = zstd.ZstdDecompressor()
    for split in splits:
        split_out_dir = shards_dir / split if split != "all" else shards_dir
        split_out_dir.mkdir(parents=True, exist_ok=True)
        result = _transcode_split(
            split, mds_dir, _ShardTarget(split_out_dir, base.ext, base.shard_size_samples), dctx
        )
        totals = [t + r for t, r in zip(totals, result, strict=True)]
    return totals[0], totals[1], totals[2]


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

    total_samples, total_shards, total_expected = _transcode_splits(
        found_splits, mds_dir, shards_dir, _ShardTarget(shards_dir, ext, shard_size_samples)
    )

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
    if total_samples != total_expected:
        print(
            f"[ERROR] Sample count mismatch: wrote {total_samples:,} of {total_expected:,}"
            " MDS samples. Legacy MDS directory kept."
        )
        return False

    if purge_mds:
        _purge_mds_dir(mds_dir)
    return True


def _purge_mds_dir(mds_dir: Path) -> None:
    """Delete the legacy MDS folder, logging (not ignoring) any failure."""
    print(f"[CLEANUP] Safely purging legacy MDS directory {mds_dir.name} to reclaim storage...")
    try:
        shutil.rmtree(mds_dir)
    except OSError as exc:
        logger.error("Could not purge legacy MDS directory %s: %s", mds_dir, exc)
        print(f"[WARN] Could not purge {mds_dir.name}: {exc}")
        return
    print("[SUCCESS] Legacy MDS directory purged.")
