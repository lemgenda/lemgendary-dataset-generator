"""
Local transfer and staging-cleanup utilities for the sync core.

Byte-metered file/tree copies with uniform tqdm progress, plus the
Windows-safe purge of staging directories and dangling temp archives left
behind by interrupted uploads.
"""

import gc
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, BinaryIO

from tqdm import tqdm

from utils.naming import MANIFOLD_PREFIX

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1MB chunks
_MAX_ARCHIVE_AGE_SECONDS = 48 * 3600
# Binary (KiB/MiB) byte-metered bar styling shared by every finalizing copy
_BYTE_BAR_STYLE: dict[str, Any] = {
    "unit": "B", "unit_scale": True, "unit_divisor": 1024, "colour": "cyan",
}


def _finalizing_bar(total: int, label: str) -> Any:
    """Uniform byte-level progress bar used by every finalizing copy."""
    return tqdm(total=total, desc=f"FINALIZING: {label}", **_BYTE_BAR_STYLE)


def _stream_copy(fsrc: BinaryIO, fdst: BinaryIO, pbar: Any) -> None:
    """Copy one open file to another in CHUNK_SIZE pieces, advancing ``pbar``."""
    while True:
        buf = fsrc.read(CHUNK_SIZE)
        if not buf:
            break
        fdst.write(buf)
        pbar.update(len(buf))


def copy_with_progress(src, dst):
    """Buffered copy for a single file with uniform tqdm progress tracking."""
    file_size = os.path.getsize(src)
    with (
        open(src, "rb") as fsrc,
        open(dst, "wb") as fdst,
        _finalizing_bar(file_size, os.path.basename(src)) as pbar,
    ):
        _stream_copy(fsrc, fdst, pbar)


def _plan_tree_copy(src_path: Path, dst_path: Path) -> tuple[list[tuple[Path, Path]], int]:
    """List (source, destination) file pairs under ``src_path`` and their total size."""
    file_list: list[tuple[Path, Path]] = []
    total_bytes = 0
    for root, _, files in os.walk(src_path):
        for f in files:
            p = Path(root) / f
            try:
                size = p.stat().st_size
                file_list.append((p, dst_path / p.relative_to(src_path)))
                total_bytes += size
            except OSError as exc:
                logger.debug("Skipping unreadable file %s during copy planning: %s", p, exc)
    return file_list, total_bytes


def copy_tree_with_progress(src_dir, dst_dir):
    """Recursively copy directory tree with uniform real-time byte-level tqdm progress tracking."""
    src_path = Path(src_dir).resolve()
    dst_path = Path(dst_dir).resolve()
    dst_path.mkdir(parents=True, exist_ok=True)

    file_list, total_bytes = _plan_tree_copy(src_path, dst_path)
    with _finalizing_bar(total_bytes, src_path.name) as pbar:
        for src_file, dst_file in file_list:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            with open(src_file, "rb") as fsrc, open(dst_file, "wb") as fdst:
                _stream_copy(fsrc, fdst, pbar)


def _retry_remove(remove: Any, max_retries: int) -> bool:
    """Call ``remove()`` until it succeeds, collecting garbage and backing off on OSError."""
    for attempt in range(max_retries):
        try:
            remove()
            return True
        except OSError:
            gc.collect()
            time.sleep(0.5 * (attempt + 1))
    return False


def robust_staging_cleanup(
    staging_dir: Path, target_zip: Path | None = None, max_retries: int = 5
) -> bool:
    """Robustly purge staging archive and directory on Windows with GC and retry backoff."""
    gc.collect()
    time.sleep(0.5)

    if target_zip and target_zip.exists():
        deleted = _retry_remove(lambda: target_zip.unlink(missing_ok=True), max_retries)
        if not deleted and target_zip.exists():
            print(
                f"[WARN] Staging archive {target_zip.name} remained locked. Directory purge will"
                " attempt removal."
            )

    if staging_dir.exists():
        deleted_dir = _retry_remove(
            lambda: shutil.rmtree(staging_dir, ignore_errors=False), max_retries
        )
        if not deleted_dir and staging_dir.exists():
            print(
                f"[WARN] Staging directory {staging_dir.name} retained file locks. Scheduled for"
                " background sweep."
            )
            return False
    return True


def _cleanup_search_dirs(base_dir) -> list[Path]:
    """Directories that may hold staging folders or temp archives, in scan order."""
    dirs_to_check = []
    if base_dir:
        dirs_to_check.append(Path(base_dir))
    dirs_to_check.append(Path(__file__).resolve().parent.parent.parent / "LemGendaryDatasets")
    dirs_to_check.append(Path.cwd())
    temp_env = os.environ.get("TEMP") or os.environ.get("TMP")
    if temp_env:
        dirs_to_check.append(Path(temp_env))
    return dirs_to_check


def _staging_is_recent(staging: Path, now: float) -> bool:
    """True when the staging folder holds a zip modified within the retention window."""
    zips = list(staging.glob("*.zip"))
    if not zips:
        return False
    newest_mtime = max(z.stat().st_mtime for z in zips)
    return (now - newest_mtime) < _MAX_ARCHIVE_AGE_SECONDS


def _purge_staging_dirs(d: Path, manifold_name, now: float, force: bool) -> tuple[int, int]:
    """Remove stale ``.staging_*`` folders in ``d``; returns (count, bytes) removed."""
    cleaned_count = 0
    cleaned_bytes = 0
    staging_pattern = f".staging_{manifold_name}" if manifold_name else ".staging_*"
    for p in d.glob(staging_pattern):
        if not p.is_dir() or (not force and _staging_is_recent(p, now)):
            continue
        try:
            sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            shutil.rmtree(p, ignore_errors=True)
            cleaned_count += 1
            cleaned_bytes += sz
        except OSError as exc:
            logger.debug("Failed removing staging directory %s: %s", p, exc)
    return cleaned_count, cleaned_bytes


def _is_manifold_zip(path: Path, manifold_name: str | None) -> bool:
    """True for ``<manifold_name>.zip``, or any ``LemGendized*.zip`` when no name is given.

    The system Temp folder holds other programs' zips; only manifold archives
    are ours to delete.
    """
    name = path.name.lower()
    if not name.endswith(".zip"):
        return False
    if manifold_name:
        return name == f"{manifold_name}.zip".lower()
    return name.startswith(MANIFOLD_PREFIX.lower())


def _purge_temp_zips(d: Path, manifold_name, now: float, force: bool) -> tuple[int, int]:
    """Remove stale manifold zips from a Temp directory; returns (count, bytes) removed."""
    cleaned_count = 0
    cleaned_bytes = 0
    for zf in d.glob("*.zip"):
        if not _is_manifold_zip(zf, manifold_name):
            continue
        try:
            if not force and (now - zf.stat().st_mtime) < _MAX_ARCHIVE_AGE_SECONDS:
                continue
            sz = zf.stat().st_size
            zf.unlink(missing_ok=True)
            cleaned_count += 1
            cleaned_bytes += sz
        except OSError as exc:
            logger.debug("Failed unlinking temp zip %s: %s", zf, exc)
    return cleaned_count, cleaned_bytes


def cleanup_temp_archives(manifold_name=None, base_dir=None, force=False):
    """Scan and purge dangling staging directories or temporary archives."""
    cleaned_count = 0
    cleaned_bytes = 0
    now = time.time()

    for d in _cleanup_search_dirs(base_dir):
        if not d.exists():
            continue
        count, size = _purge_staging_dirs(d, manifold_name, now, force)
        cleaned_count += count
        cleaned_bytes += size
        if "Temp" in str(d):
            count, size = _purge_temp_zips(d, manifold_name, now, force)
            cleaned_count += count
            cleaned_bytes += size

    return cleaned_count, cleaned_bytes
