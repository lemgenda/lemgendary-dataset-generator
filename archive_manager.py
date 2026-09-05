import os
import sys
import zipfile
import tarfile
import argparse
from pathlib import Path
from tqdm import tqdm

CHUNK_SIZE = 1024 * 1024  # 1MB chunks for smooth real-time progress tracking
LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10MB

def verify_archive(archive_path):
    """Check if archive is fully readable and uncorrupted."""
    path_str = str(archive_path)
    try:
        if path_str.endswith((".tar", ".tar.gz", ".tgz")):
            mode = "r:gz" if path_str.endswith((".tar.gz", ".tgz")) else "r:"
            with tarfile.open(path_str, mode) as tf:
                for member in tf.getmembers():
                    pass
            return True
        else:
            with zipfile.ZipFile(path_str, "r") as zf:
                bad_file = zf.testzip()
                if bad_file:
                    print(f"[ERROR] Corrupted file in zip: {bad_file}")
                    return False
            return True
    except Exception as e:
        print(f"[ERROR] Invalid archive file {archive_path}: {e}")
        return False

def create_archive(source_dir, output_path, format="zip", root_dir=None, base_dir=None):
    """Create a zip or tar archive with uniform real-time byte-level progress bar."""
    source_path = Path(source_dir).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        print(f"[ERROR] Source directory not found: {source_path}")
        return False

    # Collect files and calculate exact total uncompressed bytes
    file_entries = []
    total_bytes = 0

    if root_dir is not None:
        root_path = Path(root_dir).resolve()
    else:
        root_path = source_path.parent

    for root, _, files in os.walk(source_path):
        for f in files:
            full_path = Path(root) / f
            try:
                size = full_path.stat().st_size
                if root_dir is not None and base_dir is not None:
                    # shutil.make_archive compatibility: root_dir=OUT_PARENT, base_dir=manifold_name
                    arcname = str(full_path.relative_to(Path(root_dir).resolve())).replace("\\", "/")
                elif base_dir:
                    rel_to_source = full_path.relative_to(source_path)
                    arcname = str(Path(base_dir) / rel_to_source).replace("\\", "/")
                elif root_dir is not None:
                    arcname = str(full_path.relative_to(Path(root_dir).resolve())).replace("\\", "/")
                else:
                    # Default: archive contents relative to source directory
                    arcname = str(full_path.relative_to(source_path)).replace("\\", "/")
                file_entries.append((full_path, arcname, size))
                total_bytes += size
            except OSError:
                continue

    if not file_entries:
        print(f"[WARN] No files found to archive in {source_path}")
        return False

    archive_name = output_path.name
    print(f"Archiving {len(file_entries)} files ({total_bytes / (1024**2):.2f} MB) -> {archive_name}")

    try:
        if format.lower() in ["tar", "tar.gz", "tgz"]:
            mode = "w:gz" if format.lower() in ["tar.gz", "tgz"] else "w:"
            with tarfile.open(output_path, mode) as tf:
                with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc=f"ARCHIVING: {archive_name}", colour="cyan") as pbar:
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
            with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
                with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc=f"ARCHIVING: {archive_name}", colour="cyan") as pbar:
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

        print(f"[SUCCESS] Archive created successfully: {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create archive {output_path}: {e}")
        if output_path.exists():
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False

def smart_extract(archive_path, dest_dir, delete_after=True):
    """Extract only missing files from archive with byte-level real-time progress bar."""
    archive_path = Path(archive_path).resolve()
    dest_path = Path(dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        print(f"[ERROR] Archive not found: {archive_path}")
        return False

    archive_str = str(archive_path)
    is_tar = archive_str.endswith((".tar", ".tar.gz", ".tgz"))

    print(f"Opening archive: {archive_path.name}")
    try:
        if is_tar:
            mode = "r:gz" if archive_str.endswith((".tar.gz", ".tgz")) else "r:"
            with tarfile.open(archive_str, mode) as tf:
                members = [m for m in tf.getmembers() if m.isfile()]
                total_files = len(members)
                to_extract = []
                total_bytes = 0

                for member in members:
                    target_file = dest_path / member.name
                    if not target_file.exists() or target_file.stat().st_size == 0:
                        to_extract.append(member)
                        total_bytes += member.size

                print(f"Found {len(to_extract)} missing files ({total_bytes / (1024**2):.2f} MB) out of {total_files} total.")

                if to_extract:
                    with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc=f"EXTRACTING: {archive_path.name}", colour="green") as pbar:
                        for member in to_extract:
                            target_file = dest_path / member.name
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
            with zipfile.ZipFile(archive_path, "r") as zf:
                members = zf.infolist()
                file_members = [m for m in members if not m.is_dir()]
                total_files = len(file_members)
                to_extract = []
                total_bytes = 0

                for member in file_members:
                    target_file = dest_path / member.filename
                    if not target_file.exists() or target_file.stat().st_size == 0:
                        to_extract.append(member)
                        total_bytes += member.file_size

                print(f"Found {len(to_extract)} missing files ({total_bytes / (1024**2):.2f} MB) out of {total_files} total.")

                if to_extract:
                    with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc=f"EXTRACTING: {archive_path.name}", colour="green") as pbar:
                        for member in to_extract:
                            if member.file_size > LARGE_FILE_THRESHOLD:
                                target_file = dest_path / member.filename
                                target_file.parent.mkdir(parents=True, exist_ok=True)
                                with zf.open(member) as source, open(target_file, "wb") as target:
                                    while True:
                                        chunk = source.read(CHUNK_SIZE)
                                        if not chunk:
                                            break
                                        target.write(chunk)
                                        pbar.update(len(chunk))
                            else:
                                zf.extract(member, dest_path)
                                pbar.update(member.file_size)
                else:
                    print("All files already extracted.")

        if delete_after:
            print(f"Extraction successful. Deleting source archive: {archive_path.name}")
            try:
                os.remove(archive_path)
            except OSError as ex:
                print(f"[WARN] Could not remove source archive: {ex}")
        return True
    except Exception as e:
        print(f"[ERROR] Extraction failed for {archive_path}: {e}")
        return False

# Backward compatibility alias
verify_zip = verify_archive

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LemGendary Smart Archive Manager")
    parser.add_argument("--action", type=str, choices=["verify", "extract", "archive"], required=True, help="Action to perform")
    parser.add_argument("--zip", "--archive", dest="archive_path", type=str, help="Path to the archive file")
    parser.add_argument("--dest", type=str, help="Path to extract destination or output archive file")
    parser.add_argument("--source", type=str, help="Path to source directory for archiving")
    parser.add_argument("--format", type=str, default="zip", choices=["zip", "tar", "tar.gz", "tgz"], help="Archive format for creation")
    parser.add_argument("--keep-archive", action="store_true", help="Do not delete archive after extraction")

    args = parser.parse_args()

    if args.action == "verify":
        if not args.archive_path:
            print("[ERROR] --zip/--archive is required for verify action.")
            sys.exit(1)
        success = verify_archive(args.archive_path)
        sys.exit(0 if success else 1)

    elif args.action == "extract":
        if not args.archive_path or not args.dest:
            print("[ERROR] --zip/--archive and --dest are required for extract action.")
            sys.exit(1)
        success = smart_extract(args.archive_path, args.dest, delete_after=not args.keep_archive)
        sys.exit(0 if success else 1)

    elif args.action == "archive":
        source_dir = args.source
        dest_archive = args.dest or args.archive_path
        if not source_dir or not dest_archive:
            print("[ERROR] --source and --dest are required for archive action.")
            sys.exit(1)
        success = create_archive(source_dir, dest_archive, format=args.format)
        sys.exit(0 if success else 1)

