"""
LemGendary Dataset Compiler — Archive extraction.

Extracts .zip / .tar / .tar.gz / .tgz archives using native 7-Zip or tar
engines when available, falling back to a resumable pure-Python extractor
(only missing or empty files are written) with a byte-level progress bar.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from utils.archive_io import byte_progress_bar, copy_stream, is_tar_path, tar_mode
from utils.member_paths import STRUCTURAL_DIRS

logger = logging.getLogger(__name__)

__all__ = ["_find_7z", "_find_tar", "_flatten_common_root", "smart_extract"]

# Program-files roots searched for a 7-Zip install (Windows environment variables).
_PROGRAM_FILES_VARS = ("ProgramFiles", "ProgramFiles(x86)")

# (archive member, member name, uncompressed size)
_MemberSpec = tuple[Any, str, int]
# (archive member, destination file, uncompressed size)
_PlannedFile = tuple[Any, Path, int]


def _find_7z() -> str | None:
    """Locate 7-Zip executable on Windows or system PATH."""
    for var in _PROGRAM_FILES_VARS:
        base = os.environ.get(var)
        if base:
            candidate = os.path.join(base, "7-Zip", "7z.exe")
            if os.path.isfile(candidate):
                return candidate
    return shutil.which("7z") or shutil.which("7z.exe")


def _find_tar() -> str | None:
    """Locate native tar executable."""
    return shutil.which("tar") or shutil.which("tar.exe")




def _is_wrapper_dir(cand: Path, dest_path: Path) -> bool:
    """True when `cand` is a wrapper folder whose children belong in `dest_path`.

    A structural folder (``shards``, ``images``, ...) is never a wrapper, even
    when it is the only folder of a ``LemGendized*`` manifold.
    """
    if cand.name.lower() in STRUCTURAL_DIRS:
        return False
    return (
        cand.name.lower() == dest_path.name.lower()
        or cand.name.startswith("LemGendized")
        or dest_path.name.startswith("LemGendized")
    )


def _merge_dir_into(child: Path, dest_child: Path) -> None:
    """Move `child`'s entries missing from `dest_child` into it, then drop `child`."""
    for subchild in list(child.iterdir()):
        target_sub = dest_child / subchild.name
        if not target_sub.exists():
            shutil.move(str(subchild), str(target_sub))
    try:
        child.rmdir()
    except OSError as exc:
        logger.debug("Leaving non-empty merged dir %s: %s", child, exc)


def _move_up(child: Path, dest_path: Path) -> None:
    """Move one wrapper child into `dest_path`, merging or replacing collisions."""
    dest_child = dest_path / child.name
    if not dest_child.exists():
        shutil.move(str(child), str(dest_child))
        return
    if dest_child.is_dir() and child.is_dir():
        _merge_dir_into(child, dest_child)
        return
    try:
        os.remove(dest_child)
    except OSError as exc:
        logger.warning("Could not remove %s before replacing it: %s", dest_child, exc)
    shutil.move(str(child), str(dest_child))


def _flatten_common_root(dest_path: Path) -> None:
    """Move up the children of a single top-level wrapper folder (e.g. a LemGendized* folder)."""
    try:
        entries = [e for e in dest_path.iterdir() if e.name not in (".git", ".tmp")]
        subdirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        if len(subdirs) != 1 or len(files) > 5 or not _is_wrapper_dir(subdirs[0], dest_path):
            return
        cand = subdirs[0]
        for child in list(cand.iterdir()):
            _move_up(child, dest_path)
        try:
            cand.rmdir()
        except OSError as exc:
            logger.debug("Leaving wrapper dir %s in place: %s", cand, exc)
    except OSError as exc:
        logger.warning("Directory flattening of %s incomplete: %s", dest_path, exc)
        print(f"[DEBUG] Directory flattening note: {exc}")


def _delete_source(source_archive: Path, delete_after: bool) -> None:
    if not delete_after:
        return
    print(f"Extraction successful. Deleting source archive: {source_archive.name}")
    try:
        os.remove(source_archive)
    except OSError as ex:
        print(f"[WARN] Could not remove source archive: {ex}")


def _run_hidden(cmd: list[str]) -> int:
    """Run `cmd` without a console window on Windows; return its exit code."""
    cflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if sys.platform == "win32" else 0
    return subprocess.run(cmd, creationflags=cflags, check=False).returncode


def _stream_7z_progress(stdout: Any, archive_name: str) -> int:
    """Parse `-bsp1` 7-Zip stdout percentage tokens and emit progress lines."""
    last_pct = -1
    if stdout is None:
        return last_pct
    for line in stdout:
        for token in line.split():
            if token.endswith("%") and token[:-1].isdigit():
                pct = int(token[:-1])
                if pct != last_pct:
                    last_pct = pct
                    print(f"EXTRACTING: {archive_name} ({pct}%)", flush=True)
    return last_pct


def _try_7z(seven_z: str, source_archive: Path, dest_path: Path) -> bool:
    print(f"[EXTRACT] High-throughput native extraction via 7-Zip: {source_archive.name}")
    cmd = [seven_z, "x", str(source_archive), f"-o{dest_path}", "-aos", "-bsp1", "-mmt=on", "-y"]
    cflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", creationflags=cflags,
        ) as proc:
            last_pct = _stream_7z_progress(proc.stdout, source_archive.name)
            returncode = proc.wait()
        if returncode == 0:
            if last_pct < 100:
                print(f"EXTRACTING: {source_archive.name} (100%)", flush=True)
            return True
        print(f"[WARN] 7-Zip exited with return code {returncode}. Attempting fallback...")
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("7-Zip extraction of %s failed; falling back: %s", source_archive, exc)
        print(f"[WARN] 7-Zip extraction encountered error: {exc}. Attempting fallback...")
    return False


def _try_tar(native_tar: str, source_archive: Path, dest_path: Path) -> bool:
    print(f"[EXTRACT] Native extraction via tar: {source_archive.name}")
    cmd = [native_tar, "-xf", str(source_archive), "-C", str(dest_path)]
    try:
        return _run_hidden(cmd) == 0
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Native tar extraction of %s failed; falling back: %s", source_archive, exc)
        print(f"[WARN] Native tar extraction encountered error: {exc}. Attempting fallback...")
    return False


def _native_extract(source_archive: Path, dest_path: Path, is_tar: bool) -> bool:
    """Try 7-Zip, then native tar (tar archives only); flatten and return True on success."""
    seven_z = _find_7z()
    done = bool(seven_z) and _try_7z(seven_z, source_archive, dest_path)
    if not done:
        native_tar = _find_tar()
        done = is_tar and bool(native_tar) and _try_tar(native_tar, source_archive, dest_path)
    if done:
        _flatten_common_root(dest_path)
    return done


def _common_root(names: list[str]) -> str | None:
    """Return the single top-level folder shared by every name, if any."""
    if not names or not all(len(Path(name).parts) > 1 for name in names):
        return None
    first_parts = {Path(name).parts[0] for name in names}
    return first_parts.pop() if len(first_parts) == 1 else None


def _plan_extraction(
    specs: list[_MemberSpec], dest_path: Path
) -> tuple[list[_PlannedFile], int]:
    """Map members to destination files, keeping only missing or empty ones."""
    common_root = _common_root([name for _, name, _ in specs])
    strip_root = bool(
        common_root
        and (
            dest_path.name.lower() == common_root.lower()
            or dest_path.name.startswith("LemGendized")
        )
    )
    planned: list[_PlannedFile] = []
    total_bytes = 0
    for member, name, size in specs:
        rel_name = Path(name)
        if strip_root and common_root:
            rel_name = rel_name.relative_to(common_root)
        target_file = dest_path / rel_name
        if not target_file.exists() or target_file.stat().st_size == 0:
            planned.append((member, target_file, size))
            total_bytes += size
    return planned, total_bytes


def _extract_planned(
    specs: list[_MemberSpec],
    dest_path: Path,
    open_member: Callable[[Any], IO[bytes] | None],
    desc: str,
) -> None:
    """Write every missing member to disk with a byte-level progress bar."""
    planned, total_bytes = _plan_extraction(specs, dest_path)
    print(
        f"Found {len(planned)} missing files ({total_bytes / (1024**2):.2f} MB) out"
        f" of {len(specs)} total."
    )
    if not planned:
        print("All files already extracted.")
        return
    with byte_progress_bar(total_bytes, desc) as pbar:
        for member, target_file, _size in planned:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            source = open_member(member)
            if source is None:
                continue
            with source, open(target_file, "wb") as target:
                copy_stream(source, target, pbar)


def _python_extract(source_archive: Path, dest_path: Path, is_tar: bool) -> None:
    """Pure-Python resumable extraction of a tar or zip archive."""
    desc = f"EXTRACTING: {source_archive.name}"
    archive_str = str(source_archive)
    if is_tar:
        with tarfile.open(archive_str, tar_mode(archive_str, "r")) as tf:
            specs: list[_MemberSpec] = [
                (m, m.name, m.size) for m in tf.getmembers() if m.isfile()
            ]
            _extract_planned(specs, dest_path, tf.extractfile, desc)
        return
    with zipfile.ZipFile(source_archive, "r") as zf:
        specs = [(m, m.filename, m.file_size) for m in zf.infolist() if not m.is_dir()]
        _extract_planned(specs, dest_path, zf.open, desc)


def smart_extract(
    archive_path: str | Path, dest_dir: str | Path, delete_after: bool = True
) -> bool:
    """Extract archive using native 7-Zip/tar engines when available with Python fallback."""
    source_archive = Path(archive_path).resolve()
    dest_path = Path(dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    if not source_archive.exists():
        print(f"[ERROR] Archive not found: {source_archive}")
        return False

    is_tar = is_tar_path(str(source_archive))
    if _native_extract(source_archive, dest_path, is_tar):
        _delete_source(source_archive, delete_after)
        return True

    print(f"Opening archive: {source_archive.name}")
    try:
        _python_extract(source_archive, dest_path, is_tar)
        _delete_source(source_archive, delete_after)
        return True
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as e:
        print(f"[ERROR] Extraction failed for {source_archive}: {e}")
        return False
