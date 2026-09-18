"""
LemGendary Dataset Compiler — Smart Archive Utility.

Provides robust compression, verification, and chunked extraction
for datasets (.zip, .tar, .tar.gz, .tgz) with real-time progress tracking.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tarfile
from typing import Sequence
import zipfile

from tqdm import tqdm

CHUNK_SIZE = 1024 * 1024  # 1MB chunks for smooth real-time progress tracking
LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10MB


def verify_archive(archive_path: str | Path) -> bool:
    """Check if archive is fully readable and uncorrupted."""
    path_str = str(archive_path)
    try:
        if path_str.endswith((".tar", ".tar.gz", ".tgz")):
            mode = "r:gz" if path_str.endswith((".tar.gz", ".tgz")) else "r:"
            with tarfile.open(path_str, mode) as tf:
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

    # Collect files and calculate exact total uncompressed bytes
    file_entries: list[tuple[Path, str, int]] = []
    total_bytes = 0

    for root, _, files in os.walk(source_path):
        for f in files:
            full_path = Path(root) / f
            try:
                size = full_path.stat().st_size
                if root_dir is not None and base_dir is not None:
                    arcname = full_path.relative_to(Path(root_dir).resolve()).as_posix()
                elif base_dir:
                    rel_to_source = full_path.relative_to(source_path)
                    arcname = (Path(base_dir) / rel_to_source).as_posix()
                elif root_dir is not None:
                    arcname = full_path.relative_to(Path(root_dir).resolve()).as_posix()
                else:
                    # Default: archive entries retain manifold directory name
                    arcname = full_path.relative_to(source_path.parent).as_posix()

                file_entries.append((full_path, arcname, size))
                total_bytes += size
            except OSError:
                continue

    if not file_entries:
        print(f"[WARN] No files found to archive in {source_path}")
        return False

    archive_name = target_output.name
    print(f"Archiving {len(file_entries)} files ({total_bytes / (1024**2):.2f} MB) -> {archive_name}")

    try:
        pbar_kwargs = {
            "total": total_bytes,
            "unit": "B",
            "unit_scale": True,
            "unit_divisor": 1024,
            "desc": f"ARCHIVING: {archive_name}",
            "colour": "cyan",
            "file": sys.stdout,
            "dynamic_ncols": True,
            "mininterval": 0.25,
        }
        if archive_format.lower() in ["tar", "tar.gz", "tgz"]:
            mode = "w:gz" if archive_format.lower() in ["tar.gz", "tgz"] else "w:"
            with tarfile.open(target_output, mode) as tf:
                with tqdm(**pbar_kwargs) as pbar:
                    for full_path, arcname, size in file_entries:
                        tar_info = tf.gettarinfo(str(full_path), arcname=arcname)
                        if size > LARGE_FILE_THRESHOLD:
                            with open(full_path, "rb") as f_in:
                                tf.addfile(tar_info, f_in)
                                pbar.update(size)
                        else:
                            tf.add(str(full_path), arcname=arcname, recursive=False)
                            pbar.update(size)
        else:
            with zipfile.ZipFile(target_output, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
                with tqdm(**pbar_kwargs) as pbar:
                    for full_path, arcname, size in file_entries:
                        if size > LARGE_FILE_THRESHOLD:
                            with open(full_path, "rb") as f_in:
                                with zf.open(arcname, "w", force_zip64=True) as zf_out:
                                    while True:
                                        chunk = f_in.read(CHUNK_SIZE)
                                        if not chunk:
                                            break
                                        zf_out.write(chunk)
                                        pbar.update(len(chunk))
                        else:
                            zf.write(full_path, arcname)
                            pbar.update(size)

        print(f"[SUCCESS] Archive created successfully: {target_output}")
        return True
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as e:
        print(f"[ERROR] Failed to create archive {target_output}: {e}")
        if target_output.exists():
            try:
                os.remove(target_output)
            except OSError as err:
                print(f"[WARNING] Could not remove partial archive {target_output}: {err}")
        return False


def smart_extract(archive_path: str | Path, dest_dir: str | Path, delete_after: bool = True) -> bool:
    """Extract only missing files from archive with byte-level real-time progress bar."""
    source_archive = Path(archive_path).resolve()
    dest_path = Path(dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    if not source_archive.exists():
        print(f"[ERROR] Archive not found: {source_archive}")
        return False

    archive_str = str(source_archive)
    is_tar = archive_str.endswith((".tar", ".tar.gz", ".tgz"))

    print(f"Opening archive: {source_archive.name}")
    try:
        if is_tar:
            mode = "r:gz" if archive_str.endswith((".tar.gz", ".tgz")) else "r:"
            with tarfile.open(archive_str, mode) as tf:
                members = [m for m in tf.getmembers() if m.isfile()]
                total_files = len(members)

                # Detect if archive contains a common top-level directory covering all files
                all_names = [m.name for m in members]
                has_common_root = len(all_names) > 0 and all(len(Path(name).parts) > 1 for name in all_names)
                common_root = None
                if has_common_root:
                    first_parts = {Path(name).parts[0] for name in all_names}
                    if len(first_parts) == 1:
                        common_root = first_parts.pop()

                strip_root = bool(common_root and (dest_path.name.lower() == common_root.lower() or dest_path.name.startswith("LemGendized")))

                to_extract: list[tuple[tarfile.TarInfo, Path]] = []
                total_bytes = 0

                for member in members:
                    rel_name = Path(member.name).relative_to(common_root) if (strip_root and common_root) else Path(member.name)
                    target_file = dest_path / rel_name
                    if not target_file.exists() or target_file.stat().st_size == 0:
                        to_extract.append((member, target_file))
                        total_bytes += member.size

                print(f"Found {len(to_extract)} missing files ({total_bytes / (1024**2):.2f} MB) out of {total_files} total.")

                if to_extract:
                    pbar_kwargs = {
                        "total": total_bytes,
                        "unit": "B",
                        "unit_scale": True,
                        "unit_divisor": 1024,
                        "desc": f"EXTRACTING: {source_archive.name}",
                        "colour": "green",
                        "file": sys.stdout,
                        "dynamic_ncols": True,
                        "mininterval": 0.25,
                    }
                    with tqdm(**pbar_kwargs) as pbar:
                        for member, target_file in to_extract:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            source = tf.extractfile(member)
                            if source is not None:
                                with open(target_file, "wb") as target:
                                    while True:
                                        chunk = source.read(CHUNK_SIZE)
                                        if not chunk:
                                            break
                                        target.write(chunk)
                                        pbar.update(len(chunk))
                                source.close()
                else:
                    print("All files already extracted.")
        else:
            with zipfile.ZipFile(source_archive, "r") as zf:
                members = zf.infolist()
                file_members = [m for m in members if not m.is_dir()]
                total_files = len(file_members)

                # Detect if archive contains a common top-level directory covering all files
                all_names = [m.filename for m in file_members]
                has_common_root = len(all_names) > 0 and all(len(Path(name).parts) > 1 for name in all_names)
                common_root = None
                if has_common_root:
                    first_parts = {Path(name).parts[0] for name in all_names}
                    if len(first_parts) == 1:
                        common_root = first_parts.pop()

                strip_root = bool(common_root and (dest_path.name.lower() == common_root.lower() or dest_path.name.startswith("LemGendized")))

                to_extract_zip: list[tuple[zipfile.ZipInfo, Path]] = []
                total_bytes = 0

                for member_zip in file_members:
                    rel_name = Path(member_zip.filename).relative_to(common_root) if (strip_root and common_root) else Path(member_zip.filename)
                    target_file = dest_path / rel_name
                    if not target_file.exists() or target_file.stat().st_size == 0:
                        to_extract_zip.append((member_zip, target_file))
                        total_bytes += member_zip.file_size

                print(f"Found {len(to_extract_zip)} missing files ({total_bytes / (1024**2):.2f} MB) out of {total_files} total.")

                if to_extract_zip:
                    pbar_kwargs = {
                        "total": total_bytes,
                        "unit": "B",
                        "unit_scale": True,
                        "unit_divisor": 1024,
                        "desc": f"EXTRACTING: {source_archive.name}",
                        "colour": "green",
                        "file": sys.stdout,
                        "dynamic_ncols": True,
                        "mininterval": 0.25,
                    }
                    with tqdm(**pbar_kwargs) as pbar:
                        for member_zip, target_file in to_extract_zip:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            if member_zip.file_size > LARGE_FILE_THRESHOLD:
                                with zf.open(member_zip) as source, open(target_file, "wb") as target:
                                    while True:
                                        chunk = source.read(CHUNK_SIZE)
                                        if not chunk:
                                            break
                                        target.write(chunk)
                                        pbar.update(len(chunk))
                            else:
                                with zf.open(member_zip) as source, open(target_file, "wb") as target:
                                    while True:
                                        chunk = source.read(CHUNK_SIZE)
                                        if not chunk:
                                            break
                                        target.write(chunk)
                                        pbar.update(len(chunk))
                else:
                    print("All files already extracted.")

        if delete_after:
            print(f"Extraction successful. Deleting source archive: {source_archive.name}")
            try:
                os.remove(source_archive)
            except OSError as ex:
                print(f"[WARN] Could not remove source archive: {ex}")
        return True
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as e:
        print(f"[ERROR] Extraction failed for {source_archive}: {e}")
        return False


# Backward compatibility alias
verify_zip = verify_archive


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for archive manager."""
    parser = argparse.ArgumentParser(description="LemGendary Smart Archive Manager")
    parser.add_argument("--action", type=str, choices=["verify", "extract", "archive"], required=True, help="Action to perform")
    parser.add_argument("--zip", "--archive", dest="archive_path", type=str, help="Path to the archive file")
    parser.add_argument("--dest", type=str, help="Path to extract destination or output archive file")
    parser.add_argument("--source", type=str, help="Path to source directory for archiving")
    parser.add_argument("--format", type=str, default="zip", choices=["zip", "tar", "tar.gz", "tgz"], help="Archive format for creation")
    parser.add_argument("--keep-archive", action="store_true", help="Do not delete archive after extraction")

    args = parser.parse_args(argv)

    if args.action == "verify":
        if not args.archive_path:
            print("[ERROR] --zip/--archive is required for verify action.")
            return 1
        success = verify_archive(args.archive_path)
        return 0 if success else 1

    if args.action == "extract":
        if not args.archive_path or not args.dest:
            print("[ERROR] --zip/--archive and --dest are required for extract action.")
            return 1
        success = smart_extract(args.archive_path, args.dest, delete_after=not args.keep_archive)
        return 0 if success else 1

    if args.action == "archive":
        source_dir = args.source
        dest_archive = args.dest or args.archive_path
        if not source_dir or not dest_archive:
            print("[ERROR] --source and --dest are required for archive action.")
            return 1
        success = create_archive(source_dir, dest_archive, archive_format=args.format)
        return 0 if success else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
