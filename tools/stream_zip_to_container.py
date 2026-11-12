"""
LemGendary Dataset Compiler — Streaming Zip-to-Container Engine.

Directly streams multi-gigabyte legacy dataset zip archives into modern
canonical containers (WebDataset tar shards, MDS) with on-the-fly
multi-threaded WebP transcoding, without ever extracting loose files to
disk. Eliminates Windows NTFS MFT lock contention, cluster slack, and
antivirus overhead.

This module is the stable facade and CLI entry point. The implementation
lives in:

    tools.zip_stream_common      — options, zip member layout, samples, cleanup
    tools.zip_stream_webdataset  — zip -> WebDataset tar shards
    tools.zip_stream_mds         — zip -> MDS shards (WebDataset fallback)
    utils.container_stream       — WebP transcode and shared container helpers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from formats.mds import LOOSE_MDS_COLUMNS
from tools import zip_stream_common as _common
from tools.zip_stream_common import SampleEncoding, ZipStreamOptions
from tools.zip_stream_mds import stream_zip_to_mds
from tools.zip_stream_webdataset import stream_zip_to_webdataset
from utils import container_stream as _container

# Legacy private names kept importable for existing callers and tests.
_IMAGE_EXTS = _common.IMAGE_EXTS
_extract_metadata_files = _common.extract_metadata_files
_partition_zip_members = _common.partition_zip_members
_post_process_zip = _common.post_process_zip
_transcode_to_webp = _container.transcode_to_webp

__all__ = [
    "LOOSE_MDS_COLUMNS",
    "_IMAGE_EXTS",
    "ZipStreamOptions",
    "_extract_metadata_files",
    "_partition_zip_members",
    "_post_process_zip",
    "_transcode_to_webp",
    "build_parser",
    "main",
    "stream_zip_to_mds",
    "stream_zip_to_webdataset",
]


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Stream legacy zip directly into canonical container format."
    )
    parser.add_argument("--zip", required=True, type=Path, help="Path to source .zip archive")
    parser.add_argument("--target", required=True, type=Path, help="Destination manifold directory")
    parser.add_argument("--format", default="webdataset", choices=["webdataset", "mds"],
                        help="Target container format (default: webdataset)")
    parser.add_argument("--shard-size", type=int, default=5000,
                        help="Number of samples per WebDataset shard (ignored for MDS)")
    parser.add_argument("--mds-shard-mb", type=int, default=512,
                        help="MDS shard size in MB (default: 512)")
    parser.add_argument(
        "--no-webp", action="store_true", help="Disable WebP transcoding (keep original format)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker threads for transcoding",
    )
    parser.add_argument(
        "--delete-zip",
        action="store_true",
        help="Delete source zip archive upon successful conversion",
    )
    parser.add_argument("--delete-legacy-dir", type=Path, default=None,
                        help="Optional legacy directory to delete after conversion")
    return parser


def main() -> int:
    """Stream a legacy zip archive into WebDataset or MDS shards (CLI entry point)."""
    args = build_parser().parse_args()
    options = ZipStreamOptions(
        shard_size_samples=args.shard_size,
        mds_shard_size_bytes=args.mds_shard_mb * 1024 * 1024,
        delete_zip=args.delete_zip,
        legacy_dir=args.delete_legacy_dir,
        max_workers=args.workers,
        encoding=SampleEncoding(transcode_webp=not args.no_webp),
    )

    if args.format == "webdataset":
        return 0 if stream_zip_to_webdataset(args.zip, args.target, options) else 1
    if args.format == "mds":
        return 0 if stream_zip_to_mds(args.zip, args.target, options) else 1

    print(f"[ERROR] Format {args.format} streaming not implemented.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
