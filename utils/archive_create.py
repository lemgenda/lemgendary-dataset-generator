"""
LemGendary Dataset Compiler — Archive creation.

Builds .zip / .tar / .tar.gz / .tgz archives from a directory with a
uniform real-time byte-level progress bar.
"""

from __future__ import annotations

import logging
import os
import tarfile
import zipfile
from pathlib import Path

from utils.archive_io import (
    LARGE_FILE_THRESHOLD,
    TarWriteMode,
    byte_progress_bar,
    copy_stream,
    tar_mode,
)

logger = logging.getLogger(__name__)

__all__ = ["create_archive"]

_TAR_FORMATS = ("tar", "tar.gz", "tgz")

# (absolute file path, archive member name, size in bytes)
_FileEntry = tuple[Path, str, int]
# (total bytes, progress-bar description)
_BarSpec = tuple[int, str]


def _arcname(
    full_path: Path, source_path: Path, root_dir: str | Path | None, base_dir: str | None
) -> str:
    """Return the archive member name of `full_path` for the root/base layout."""
    if root_dir is not None:
        # With or without base_dir, root_dir anchors the member names.
        return full_path.relative_to(Path(root_dir).resolve()).as_posix()
    if base_dir:
        return (Path(base_dir) / full_path.relative_to(source_path)).as_posix()
    # Default: archive entries retain manifold directory name
    return full_path.relative_to(source_path.parent).as_posix()


def _collect_entries(
    source_path: Path, root_dir: str | Path | None, base_dir: str | None
) -> tuple[list[_FileEntry], int]:
    """Walk `source_path`; return its file entries and their total byte size."""
    file_entries: list[_FileEntry] = []
    total_bytes = 0
    for root, _, files in os.walk(source_path):
        for f in files:
            full_path = Path(root) / f
            try:
                size = full_path.stat().st_size
                arcname = _arcname(full_path, source_path, root_dir, base_dir)
                file_entries.append((full_path, arcname, size))
                total_bytes += size
            except OSError as exc:
                logger.warning("Skipping unreadable file %s while archiving: %s", full_path, exc)
    return file_entries, total_bytes


def _write_tar(
    target_output: Path, mode: TarWriteMode, entries: list[_FileEntry], progress: _BarSpec
) -> None:
    with tarfile.open(target_output, mode) as tf, byte_progress_bar(*progress) as pbar:
        for full_path, arcname, size in entries:
            tar_info = tf.gettarinfo(str(full_path), arcname=arcname)
            if size > LARGE_FILE_THRESHOLD:
                with open(full_path, "rb") as f_in:
                    tf.addfile(tar_info, f_in)
            else:
                tf.add(str(full_path), arcname=arcname, recursive=False)
            pbar.update(size)


def _write_zip(target_output: Path, entries: list[_FileEntry], progress: _BarSpec) -> None:
    with (
        zipfile.ZipFile(
            target_output, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as zf,
        byte_progress_bar(*progress) as pbar,
    ):
        for full_path, arcname, size in entries:
            if size > LARGE_FILE_THRESHOLD:
                with (
                    open(full_path, "rb") as f_in,
                    zf.open(arcname, "w", force_zip64=True) as zf_out,
                ):
                    copy_stream(f_in, zf_out, pbar)
            else:
                zf.write(full_path, arcname)
                pbar.update(size)


def _remove_partial(target_output: Path) -> None:
    if target_output.exists():
        try:
            os.remove(target_output)
        except OSError as err:
            print(f"[WARNING] Could not remove partial archive {target_output}: {err}")


def create_archive(
    source_dir: str | Path,
    output_path: str | Path,
    archive_format: str = "zip",
    root_dir: str | Path | None = None,
    base_dir: str | None = None,
) -> bool:
    """Create a zip or tar archive with uniform real-time byte-level progress bar."""
    source_path = Path(source_dir).resolve()
    target_output = Path(output_path).resolve()
    target_output.parent.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        print(f"[ERROR] Source directory not found: {source_path}")
        return False

    file_entries, total_bytes = _collect_entries(source_path, root_dir, base_dir)
    if not file_entries:
        print(f"[WARN] No files found to archive in {source_path}")
        return False

    archive_name = target_output.name
    print(
        f"Archiving {len(file_entries)} files ({total_bytes / (1024**2):.2f} MB) -> {archive_name}"
    )

    fmt = archive_format.lower()
    try:
        progress: _BarSpec = (total_bytes, f"ARCHIVING: {archive_name}")
        if fmt in _TAR_FORMATS:
            _write_tar(target_output, tar_mode(f".{fmt}", "w"), file_entries, progress)
        else:
            _write_zip(target_output, file_entries, progress)
        print(f"[SUCCESS] Archive created successfully: {target_output}")
        return True
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as e:
        print(f"[ERROR] Failed to create archive {target_output}: {e}")
        _remove_partial(target_output)
        return False
