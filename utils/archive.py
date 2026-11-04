"""
LemGendary Dataset Compiler — Smart Archive Utility.

Provides robust compression, verification, and chunked extraction
for datasets (.zip, .tar, .tar.gz, .tgz) with real-time progress tracking.

This module is the stable facade and CLI entry point. The implementation
lives in:

    utils.archive_io       — chunk constants, tar modes, progress, stream copy
    utils.archive_create   — create_archive
    utils.archive_extract  — smart_extract (7-Zip / tar / Python fallback)
"""

from __future__ import annotations

import argparse
import logging
import sys
import tarfile
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

from utils.archive_create import create_archive
from utils.archive_extract import (
    _find_7z,
    _find_tar,
    _flatten_common_root,
    smart_extract,
)
from utils.archive_io import CHUNK_SIZE, LARGE_FILE_THRESHOLD, is_tar_path, tar_mode

logger = logging.getLogger(__name__)

__all__ = [
    "CHUNK_SIZE",
    "LARGE_FILE_THRESHOLD",
    "_find_7z",
    "_find_tar",
    "_flatten_common_root",
    "create_archive",
    "main",
    "smart_extract",
    "verify_archive",
    "verify_zip",
]


def verify_archive(archive_path: str | Path) -> bool:
    """Check if archive is fully readable and uncorrupted."""
    path_str = str(archive_path)
    try:
        if is_tar_path(path_str):
            with tarfile.open(path_str, tar_mode(path_str, "r")) as tf:
                for member in tf.getmembers():
                    _ = member
            return True
        with zipfile.ZipFile(path_str, "r") as zf:
            bad_file = zf.testzip()
            if bad_file:
                print(f"[ERROR] Corrupted file in zip: {bad_file}")
                return False
        return True
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as e:
        print(f"[ERROR] Invalid archive file {archive_path}: {e}")
        return False


# Backward compatibility alias
verify_zip = verify_archive


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LemGendary Smart Archive Manager")
    parser.add_argument(
        "--action",
        type=str,
        choices=["verify", "extract", "archive"],
        required=True,
        help="Action to perform",
    )
    parser.add_argument(
        "--zip", "--archive", dest="archive_path", type=str, help="Path to the archive file"
    )
    parser.add_argument(
        "--dest", type=str, help="Path to extract destination or output archive file"
    )
    parser.add_argument("--source", type=str, help="Path to source directory for archiving")
    parser.add_argument(
        "--format",
        type=str,
        default="zip",
        choices=["zip", "tar", "tar.gz", "tgz"],
        help="Archive format for creation",
    )
    parser.add_argument(
        "--keep-archive", action="store_true", help="Do not delete archive after extraction"
    )
    return parser


def _cli_verify(args: argparse.Namespace) -> int:
    if not args.archive_path:
        print("[ERROR] --zip/--archive is required for verify action.")
        return 1
    return 0 if verify_archive(args.archive_path) else 1


def _cli_extract(args: argparse.Namespace) -> int:
    if not args.archive_path or not args.dest:
        print("[ERROR] --zip/--archive and --dest are required for extract action.")
        return 1
    success = smart_extract(args.archive_path, args.dest, delete_after=not args.keep_archive)
    return 0 if success else 1


def _cli_archive(args: argparse.Namespace) -> int:
    source_dir = args.source
    dest_archive = args.dest or args.archive_path
    if not source_dir or not dest_archive:
        print("[ERROR] --source and --dest are required for archive action.")
        return 1
    return 0 if create_archive(source_dir, dest_archive, archive_format=args.format) else 1


_CLI_ACTIONS: dict[str, Callable[[argparse.Namespace], int]] = {
    "verify": _cli_verify,
    "extract": _cli_extract,
    "archive": _cli_archive,
}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for archive manager."""
    args = _build_parser().parse_args(argv)
    handler = _CLI_ACTIONS.get(args.action)
    return handler(args) if handler is not None else 1


if __name__ == "__main__":
    sys.exit(main())
