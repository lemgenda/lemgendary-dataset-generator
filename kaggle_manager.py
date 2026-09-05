import argparse
import logging
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import TypedDict
import kagglehub
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*outdated.*")
logging.getLogger("kagglehub").setLevel(logging.ERROR)

CHUNK_SIZE = 1024 * 1024  # 1MB chunks


class DatasetVersionInfo(TypedDict):
    current_version: int
    latest_version: int
    versions: dict[int, str]

def setup_kaggle_auth(default_user="lemtreursi"):
    """Ensure Kaggle credentials from environment or .kaggle_token are initialized."""
    token_file = Path(".kaggle_token")
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token
            if "KAGGLE_USERNAME" not in os.environ and default_user:
                os.environ["KAGGLE_USERNAME"] = default_user

    # Suppress verbose outdated version warnings from kaggle / kagglehub libraries
    try:
        import kagglehub.clients
        kagglehub.clients.already_printed_version_warning = True
    except Exception:
        pass

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        KaggleApi.already_printed_version_warning = True
    except Exception:
        pass

def cleanup_temp_archives(manifold_name=None, base_dir=None, force=False):
    """Scan and purge dangling staging directories or temporary archives unless preserved for resumption."""
    dirs_to_check = []
    if base_dir:
        dirs_to_check.append(Path(base_dir))
    dirs_to_check.append(Path(r"c:\Development\python\model-training\LemGendaryDatasets"))
    dirs_to_check.append(Path.cwd())
    dirs_to_check.append(Path(os.environ.get("TEMP", r"C:\Users\lemtr\AppData\Local\Temp")))

    cleaned_count = 0
    cleaned_bytes = 0
    now = time.time()
    max_age_seconds = 48 * 3600  # 48 hours preservation window

    for d in dirs_to_check:
        if not d.exists():
            continue
        # Staging directories
        staging_pattern = f".staging_{manifold_name}" if manifold_name else ".staging_*"
        for p in d.glob(staging_pattern):
            if not p.is_dir():
                continue
            if not force:
                zips = list(p.glob("*.zip"))
                if zips:
                    newest_mtime = max(z.stat().st_mtime for z in zips)
                    if (now - newest_mtime) < max_age_seconds:
                        continue
            try:
                sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p, ignore_errors=True)
                cleaned_count += 1
                cleaned_bytes += sz
            except Exception:
                pass

        # Lingering temporary archives in temp or old staging
        if "Temp" in str(d):
            zip_pattern = f"{manifold_name}.zip" if manifold_name else "*.zip"
            for zf in d.glob(zip_pattern):
                try:
                    if not force and (now - zf.stat().st_mtime) < max_age_seconds:
                        continue
                    sz = zf.stat().st_size
                    zf.unlink(missing_ok=True)
                    cleaned_count += 1
                    cleaned_bytes += sz
                except Exception:
                    pass

    return cleaned_count, cleaned_bytes

def get_dataset_version_info(repo_id: str) -> DatasetVersionInfo:
    """Retrieve current version number, latest version number, and per-version status."""
    clean_handle = repo_id.replace("kaggle://", "")
    owner = clean_handle.split("/")[0] if "/" in clean_handle else "lemtreursi"
    slug = clean_handle.split("/")[1] if "/" in clean_handle else clean_handle
    setup_kaggle_auth(default_user=owner)

    try:
        from kagglehub.clients import build_kaggle_client
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        with build_kaggle_client() as client:
            req = ApiGetDatasetRequest()
            req.owner_slug = owner
            req.dataset_slug = slug
            resp = client.datasets.dataset_api_client.get_dataset(req)
            curr = int(resp.current_version_number or 0)
            raw_versions = resp.versions or []
            valid_versions = [v for v in raw_versions if v is not None and v.version_number is not None]
            max_v = max([int(v.version_number) for v in valid_versions], default=curr)
            versions_dict: dict[int, str] = {
                int(v.version_number): str(v.status or "")
                for v in valid_versions
            }
            return {
                "current_version": curr,
                "latest_version": max_v,
                "versions": versions_dict
            }
    except Exception:
        # Fallback to KaggleApi CLI format
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            res = api.dataset_status(clean_handle)
            if isinstance(res, dict):
                curr = int(res.get("current_version_number") or 0)
                status = str(res.get("status") or "")
            elif isinstance(res, str):
                curr = 0
                status = res
            else:
                curr = 0
                status = str(res or "")
            return {"current_version": curr, "latest_version": curr, "versions": {curr: status}}
        except Exception:
            return {"current_version": 0, "latest_version": 0, "versions": {}}

def get_dataset_status(repo_id):
    """Query dataset status using Kaggle API with CLI fallback."""
    clean_handle = repo_id.replace("kaggle://", "")
    owner = clean_handle.split("/")[0] if "/" in clean_handle else "lemtreursi"
    setup_kaggle_auth(default_user=owner)

    # 1. Try Python API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api.dataset_status(clean_handle)
    except Exception:
        pass

    # 2. Fallback to CLI
    try:
        import subprocess
        res = subprocess.run(
            ["kaggle", "datasets", "status", clean_handle],
            capture_output=True,
            text=True,
            timeout=15
        )
        out = res.stdout.strip().lower()
        if "ready" in out:
            return "ready"
        elif "creating" in out or "pending" in out or "queued" in out:
            return "creating"
        elif "error" in out or "failed" in out:
            return "error"
        elif res.returncode == 0 and out:
            return out.split()[-1]
    except Exception:
        pass

    return None

def track_kaggle_dataset_status(
    repo_id: str,
    target_version: int | None = None,
    expected_files: int = 0,
    timeout: int = 1800,
    poll_interval: int = 4
) -> bool:
    """Monitor Kaggle server-side extraction/processing with real-time tqdm progress tracking."""
    clean_handle = repo_id.replace("kaggle://", "")
    owner = clean_handle.split("/")[0] if "/" in clean_handle else "lemtreursi"
    slug = clean_handle.split("/")[1] if "/" in clean_handle else clean_handle
    setup_kaggle_auth(default_user=owner)

    if target_version is None:
        info = get_dataset_version_info(clean_handle)
        target_version = info["latest_version"] or 1
        versions = info["versions"]
        if target_version in versions and versions[target_version].lower() == "ready":
            print(f"[KAGGLE STATUS] Dataset '{clean_handle}' version {target_version} is READY.")
            return True

    target_ver: int = target_version
    print(f"\n[KAGGLE] Monitoring server-side extraction for '{clean_handle}' (Target Version: {target_ver})...")
    start_time = time.time()

    pbar = tqdm(
        total=100,
        desc=f"KAGGLE EXTRACTION (v{target_ver})",
        unit="%",
        colour="cyan",
        dynamic_ncols=True,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed} [{postfix}]"
    )

    last_pct = 0.0

    while time.time() - start_time < timeout:
        elapsed = time.time() - start_time
        info = get_dataset_version_info(clean_handle)
        versions = info["versions"]

        if target_ver in versions:
            status_val = versions[target_ver].lower()
            if status_val == "ready":
                pbar.n = 100
                pbar.set_postfix_str("COMPLETE")
                pbar.refresh()
                pbar.close()
                print(f"[SUCCESS] Kaggle server-side extraction complete! Dataset '{clean_handle}' version {target_ver} is ready.")
                return True
            elif status_val in ["error", "failed"]:
                pbar.set_postfix_str("FAILED")
                pbar.refresh()
                pbar.close()
                print(f"\n[ERROR] Kaggle reports extraction/processing failed for version {target_ver}.")
                return False
            else:
                # In progress: check file summary if available
                file_pct = None
                if expected_files > 0:
                    try:
                        from kagglehub.clients import build_kaggle_client
                        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetFilesSummaryRequest
                        with build_kaggle_client() as client:
                            sreq = ApiGetDatasetFilesSummaryRequest()
                            sreq.owner_slug = owner
                            sreq.dataset_slug = slug
                            sreq.dataset_version_number = target_ver
                            sresp = client.datasets.dataset_api_client.get_dataset_files_summary(sreq)
                            cnt = sresp.file_summary_info.total_file_count if sresp.file_summary_info else 0
                            if cnt > 0:
                                file_pct = min(98.0, (cnt / expected_files) * 100.0)
                                pbar.set_postfix_str(f"{cnt}/{expected_files} files ({status_val})")
                    except Exception:
                        pass

                if file_pct is not None:
                    pbar.n = max(last_pct, file_pct)
                else:
                    # Asymptotic progress curve approaching 95% over cloud extraction duration
                    est_pct = 15.0 + 80.0 * (1.0 - math.exp(-elapsed / 300.0))
                    pbar.n = min(95.0, max(last_pct, est_pct))
                    pbar.set_postfix_str(f"cloud extracting ({status_val})...")

                last_pct = pbar.n
                pbar.refresh()
        else:
            # Target version initializing on Kaggle backend
            init_pct = min(15.0, 3.0 + (elapsed / 2.0))
            pbar.n = max(last_pct, init_pct)
            last_pct = pbar.n
            pbar.set_postfix_str("initializing cloud payload...")
            pbar.refresh()

        time.sleep(poll_interval)

    pbar.close()
    print(f"\n[WARN] Monitoring timed out after {timeout}s. Check Kaggle web dashboard for final status.")
    return False

def copy_with_progress(src, dst):
    """Buffered copy for a single file with uniform tqdm progress tracking."""
    file_size = os.path.getsize(src)
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc=f"FINALIZING: {os.path.basename(src)}", colour="cyan") as pbar:
            while True:
                buf = fsrc.read(CHUNK_SIZE)
                if not buf:
                    break
                fdst.write(buf)
                pbar.update(len(buf))

def copy_tree_with_progress(src_dir, dst_dir):
    """Recursively copy directory tree with uniform real-time byte-level tqdm progress tracking."""
    src_path = Path(src_dir).resolve()
    dst_path = Path(dst_dir).resolve()
    dst_path.mkdir(parents=True, exist_ok=True)

    file_list = []
    total_bytes = 0
    for root, _, files in os.walk(src_path):
        for f in files:
            p = Path(root) / f
            try:
                size = p.stat().st_size
                rel = p.relative_to(src_path)
                file_list.append((p, dst_path / rel, size))
                total_bytes += size
            except OSError:
                continue

    folder_name = src_path.name
    with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc=f"FINALIZING: {folder_name}", colour="cyan") as pbar:
        for src_file, dst_file, size in file_list:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            with open(src_file, "rb") as fsrc, open(dst_file, "wb") as fdst:
                while True:
                    buf = fsrc.read(CHUNK_SIZE)
                    if not buf:
                        break
                    fdst.write(buf)
                    pbar.update(len(buf))

def main():
    parser = argparse.ArgumentParser(description="LemGendary Kaggle Manager")
    parser.add_argument("--repo_id", required=True, help="Kaggle dataset handle or URL")
    parser.add_argument("--output_dir", default="", help="Target local directory")
    parser.add_argument("--is_competition", action="store_true", help="Download from competition")
    parser.add_argument("--action", default="download", choices=["download", "upload", "status"], help="Action to perform")
    parser.add_argument("--no-wait", action="store_true", help="Skip monitoring server-side extraction after upload")
    args = parser.parse_args()

    if args.action in ["download", "upload"] and not args.output_dir:
        parser.error(f"--output_dir is required when --action is '{args.action}'.")

    clean_repo_id = args.repo_id.replace("kaggle://", "")
    owner = clean_repo_id.split("/")[0] if "/" in clean_repo_id else "lemtreursi"
    setup_kaggle_auth(default_user=owner)

    if args.action == "download":
        print(f"PULLING: {clean_repo_id}")
        target_dir = Path(args.output_dir).resolve()

        # Determine root datasets directory and manifold name
        if target_dir.name.startswith("LemGendized"):
            root_datasets_dir = target_dir.parent
            manifold_name = target_dir.name
        else:
            root_datasets_dir = target_dir
            manifold_name = clean_repo_id.split("/")[-1]

        root_datasets_dir.mkdir(parents=True, exist_ok=True)
        slug = clean_repo_id.split("/")[-1]

        # 1. Resumption check: check if an unextracted or partially extracted archive already exists
        archive_candidates = [
            root_datasets_dir / f"{manifold_name}.zip",
            root_datasets_dir / f"{slug}.zip",
            target_dir / f"{manifold_name}.zip",
            target_dir / f"{slug}.zip",
        ]

        existing_archive = None
        from archive_manager import verify_archive, smart_extract

        for cand in archive_candidates:
            if cand.exists() and cand.is_file() and cand.stat().st_size > 0:
                print(f"[GET] Inspecting existing archive: {cand.name} ({cand.stat().st_size / (1024**3):.2f} GB)...")
                if verify_archive(cand):
                    existing_archive = cand
                    print(f"[GET] Archive verification passed. Resuming extraction directly from local archive...")
                    break
                else:
                    print(f"[WARN] Existing archive {cand.name} is incomplete or corrupted. Re-downloading...")
                    try:
                        cand.unlink()
                    except OSError:
                        pass

        archive_to_extract = existing_archive

        if not archive_to_extract:
            if args.is_competition:
                import subprocess
                subprocess.run(["kaggle", "competitions", "download", "-c", clean_repo_id, "-p", str(root_datasets_dir)], check=True)
                for cand in archive_candidates:
                    if cand.exists():
                        archive_to_extract = cand
                        break
            else:
                # Direct streaming download as ZIP to root_datasets_dir via Kaggle API
                downloaded = False
                try:
                    from kaggle.api.kaggle_api_extended import KaggleApi
                    api = KaggleApi()
                    api.authenticate()
                    print(f"[GET] Streaming dataset archive directly to {root_datasets_dir} via Kaggle API...")
                    api.dataset_download_files(clean_repo_id, path=str(root_datasets_dir), unzip=False, quiet=False)
                    downloaded = True
                except Exception as ex:
                    print(f"[WARN] Kaggle API direct download encountered an issue: {ex}. Falling back to Kaggle CLI...")
                    try:
                        import subprocess
                        subprocess.run(["kaggle", "datasets", "download", clean_repo_id, "-p", str(root_datasets_dir)], check=True)
                        downloaded = True
                    except Exception as cli_ex:
                        print(f"[WARN] Kaggle CLI download also failed: {cli_ex}. Falling back to kagglehub...")
                        path = kagglehub.dataset_download(clean_repo_id)
                        if os.path.isfile(path):
                            archive_to_extract = Path(path)
                        else:
                            print(f"[GET] Dataset cached by kagglehub at {path}. Finalizing to {target_dir}...")
                            items = os.listdir(path)
                            if len(items) == 1 and items[0] == target_dir.name:
                                copy_tree_with_progress(os.path.join(path, items[0]), str(target_dir))
                            else:
                                copy_tree_with_progress(path, str(target_dir))
                            shutil.rmtree(path, ignore_errors=True)

                if downloaded:
                    for cand in archive_candidates:
                        if cand.exists() and cand.is_file() and cand.stat().st_size > 0:
                            archive_to_extract = cand
                            break
                    if not archive_to_extract:
                        recent_zips = sorted(root_datasets_dir.glob("*.zip"), key=lambda f: f.stat().st_mtime, reverse=True)
                        if recent_zips:
                            archive_to_extract = recent_zips[0]

        if archive_to_extract and archive_to_extract.exists():
            print(f"[EXTRACT] Unpacking manifold archive: {archive_to_extract.name}")
            success = smart_extract(archive_to_extract, str(root_datasets_dir), delete_after=True)
            if not success:
                print(f"[ERROR] Extraction failed or interrupted. Archive preserved at {archive_to_extract} for resumption.")
                sys.exit(1)
            else:
                print(f"[SUCCESS] Manifold extracted successfully into {root_datasets_dir / manifold_name}")

    elif args.action == "upload":
        print(f"PUSHING: {clean_repo_id}")
        src_path = Path(args.output_dir).resolve()
        if not src_path.exists():
            print(f"[ERROR] Target directory does not exist: {src_path}")
            sys.exit(1)

        # Detect pre-upload version state to identify target version
        version_info = get_dataset_version_info(clean_repo_id)
        target_version: int = (version_info["latest_version"] or 0) + 1

        file_count = 0
        newest_src_mtime = 0.0
        for root, _, files in os.walk(src_path):
            file_count += len(files)
            for f in files:
                try:
                    mt = (Path(root) / f).stat().st_mtime
                    if mt > newest_src_mtime:
                        newest_src_mtime = mt
                except OSError:
                    pass

        staging_dir = None
        if file_count > 50:
            manifold_name = src_path.name
            staging_dir = src_path.parent / f".staging_{manifold_name}"
            staging_dir.mkdir(parents=True, exist_ok=True)
            target_zip = staging_dir / f"{manifold_name}.zip"

            # Check if valid staging archive already exists for resumption
            from archive_manager import verify_archive, create_archive
            can_reuse = False
            if target_zip.exists() and target_zip.is_file() and target_zip.stat().st_size > 0:
                print(f"[SYNC] Existing staging archive detected ({target_zip.stat().st_size / (1024**3):.2f} GB). Verifying...")
                if target_zip.stat().st_mtime >= newest_src_mtime and verify_archive(target_zip):
                    can_reuse = True
                    print(f"[SYNC] Existing staging archive is valid and up to date. Skipping compression and resuming upload...")
                else:
                    print(f"[SYNC] Existing archive is outdated or invalid. Re-creating...")
                    try:
                        target_zip.unlink()
                    except OSError:
                        pass

            if not can_reuse:
                print(f"[SYNC] Detected {file_count} files (>50 threshold). Archiving manifold with progress tracking...")
                success = create_archive(src_path, target_zip, format="zip")
                if not success or not target_zip.exists():
                    raise RuntimeError(f"Archive creation failed for {src_path}")

            zip_size_gb = target_zip.stat().st_size / (1024**3)
            print(f"[SYNC] Staging archive ready: {target_zip.name} ({zip_size_gb:.2f} GB)")

            upload_success = False
            try:
                print(f"[SYNC] Uploading archive to Kaggle via KaggleHub API...")
                kagglehub.dataset_upload(clean_repo_id, str(staging_dir))
                upload_success = True
            except Exception as e:
                print(f"[ERROR] Upload failed: {e}")
                print(f"[SYNC] Staging archive preserved at {target_zip} for instant resumption on next attempt.")
                sys.exit(1)
            finally:
                if upload_success:
                    print(f"[SYNC] Upload succeeded. Cleaning up local staging archive...")
                    if target_zip.exists():
                        try:
                            target_zip.unlink()
                        except OSError:
                            pass
                    if staging_dir.exists():
                        try:
                            shutil.rmtree(staging_dir, ignore_errors=True)
                        except OSError:
                            pass
        else:
            print(f"[SYNC] Uploading {file_count} files directly to Kaggle via KaggleHub API...")
            kagglehub.dataset_upload(clean_repo_id, str(src_path))

        print(f"[SUCCESS] Dataset upload payload transferred to Kaggle.")

        if not args.no_wait:
            success = track_kaggle_dataset_status(
                clean_repo_id,
                target_version=target_version,
                expected_files=file_count
            )
            if not success:
                sys.exit(1)

    elif args.action == "status":
        success = track_kaggle_dataset_status(clean_repo_id)
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()


