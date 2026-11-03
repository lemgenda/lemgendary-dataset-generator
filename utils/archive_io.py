"""
LemGendary Dataset Compiler — Archive I/O primitives.

Chunk-size constants, tar mode detection, byte-level progress bars and the
chunked stream copy shared by archive creation and extraction.
"""

from __future__ import annotations

from typing import IO, Any

from utils.progress import create_progress_bar

__all__ = [
    "CHUNK_SIZE",
    "LARGE_FILE_THRESHOLD",
    "TAR_SUFFIXES",
    "byte_progress_bar",
    "copy_stream",
    "is_tar_path",
    "tar_mode",
]

CHUNK_SIZE = 1024 * 1024  # 1MB chunks for smooth real-time progress tracking
LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10MB

TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz")
_GZIP_TAR_SUFFIXES = (".tar.gz", ".tgz")


def is_tar_path(path_str: str) -> bool:
    """True when the path names a .tar, .tar.gz or .tgz archive."""
    return path_str.endswith(TAR_SUFFIXES)


def tar_mode(path_str: str, action: str) -> str:
    """Return the tarfile mode (`r:gz`/`r:` or `w:gz`/`w:`) for `action` 'r' or 'w'."""
    compression = "gz" if path_str.endswith(_GZIP_TAR_SUFFIXES) else ""
    return f"{action}:{compression}"


def byte_progress_bar(total_bytes: int, desc: str) -> Any:
    """Return a byte-unit progress bar (binary-scaled) for archive operations."""
    return create_progress_bar(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc,
    )


def copy_stream(source: IO[bytes], target: IO[bytes], pbar: Any) -> None:
    """Copy `source` to `target` in CHUNK_SIZE blocks, advancing `pbar` per block."""
    while True:
        chunk = source.read(CHUNK_SIZE)
        if not chunk:
            break
        target.write(chunk)
        pbar.update(len(chunk))
