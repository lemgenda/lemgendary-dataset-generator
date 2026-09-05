import argparse
import logging
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
import kagglehub
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*outdated.*")
logging.getLogger("kagglehub").setLevel(logging.ERROR)

CHUNK_SIZE = 1024 * 1024  # 1MB chunks

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

def cleanup_temp_archives(manifold_name=None, base_dir=None):
    """Scan and purge dangling staging directories or temporary archives."""
    dirs_to_check = []
    if base_dir:
        dirs_to_check.append(Path(base_dir))
    dirs_to_check.append(Path(r"c:\Development\python\model-training\LemGendaryDatasets"))
    dirs_to_check.append(Path.cwd())
    dirs_to_check.append(Path(os.environ.get("TEMP", r"C:\Users\lemtr\AppData\Local\Temp")))

    cleaned_count = 0
    cleaned_bytes = 0

    for d in dirs_to_check:
        if not d.exists():
            continue
        # Staging directories
        staging_pattern = f".staging_{manifold_name}" if manifold_name else ".staging_*"
        for p in d.glob(staging_pattern):
            try:
                sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p, ignore_errors=True)
                cleaned_count += 1
                cleaned_bytes += sz
            except Exception:
                pass

        # Lingering temporary archives in staging or temp
        if d.name.startswith(".staging_") or "Temp" in str(d):
            zip_pattern = f"{manifold_name}.zip" if manifold_name else "*.zip"
            for zf in d.glob(zip_pattern):
                try:
                    sz = zf.stat().st_size
                    zf.unlink(missing_ok=True)
                    cleaned_count += 1
                    cleaned_bytes += sz
                except Exception:
                    pass

    return cleaned_count, cleaned_bytes

def get_dataset_version_info(repo_id):
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
            curr = resp.current_version_number or 0
            versions = resp.versions or []
            max_v = max([v.version_number for v in versions], default=curr)
            return {
                "current_version": curr,
                "latest_version": max_v,
                "versions": {v.version_number: v.status for v in versions}
            }
    except Exception:
        # Fallback to KaggleApi CLI format
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            res = api.dataset_status(clean_handle)
            if isinstance(res, dict):
                curr = res.get("current_version_number") or 0
                status = str(res.get("status") or "")
            else:
                curr = 0
                status = str(res)
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

def track_kaggle_dataset_status(repo_id, target_version=None, expected_files=0, timeout=1800, poll_interval=4):
    """Monitor Kaggle server-side extraction/processing with real-time tqdm progress tracking."""
    clean_handle = repo_id.replace("kaggle://", "")
    owner = clean_handle.split("/")[0] if "/" in clean_handle else "lemtreursi"
    slug = clean_handle.split("/")[1] if "/" in clean_handle else clean_handle
    setup_kaggle_auth(default_user=owner)

    if target_version is None:
        info = get_dataset_version_info(clean_handle)
        target_version = info.get("latest_version") or 1
        versions = info.get("versions", {})
        if target_version in versions and str(versions[target_version]).lower() == "ready":
            print(f"[KAGGLE STATUS] Dataset '{clean_handle}' version {target_version} is READY.")
            return True

    print(f"\n[KAGGLE] Monitoring server-side extraction for '{clean_handle}' (Target Version: {target_version})...")
    start_time = time.time()

    pbar = tqdm(
        total=100,
        desc=f"KAGGLE EXTRACTION (v{target_version})",
        unit="%",
        colour="cyan",
        dynamic_ncols=True,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed} [{postfix}]"
    )

    last_pct = 0.0

    while time.time() - start_time < timeout:
        elapsed = time.time() - start_time
        info = get_dataset_version_info(clean_handle)
        versions = info.get("versions", {})

        if target_version in versions:
            status_val = str(versions[target_version]).lower()
            if status_val == "ready":
                pbar.n = 100
                pbar.set_postfix_str("COMPLETE")
                pbar.refresh()
                pbar.close()
                print(f"[SUCCESS] Kaggle server-side extraction complete! Dataset '{clean_handle}' version {target_version} is ready.")
                return True
            elif status_val in ["error", "failed"]:
                pbar.set_postfix_str("FAILED")
                pbar.refresh()
                pbar.close()
                print(f"\n[ERROR] Kaggle reports extraction/processing failed for version {target_version}.")
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
                            sreq.dataset_version_number = target_version
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
        if args.is_competition:
            import subprocess
            subprocess.run(["kaggle", "competitions", "download", "-c", clean_repo_id, "-p", args.output_dir], check=True)
        else:
            path = kagglehub.dataset_download(clean_repo_id)
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)

            print("Downloaded to cache. Finalizing manifold...")
            if os.path.isfile(path):
                target = os.path.join(args.output_dir, os.path.basename(path))
                copy_with_progress(path, target)
                try:
                    os.remove(path)
                except OSError:
                    pass
            else:
                items = os.listdir(path)
                for item in items:
                    s = os.path.join(path, item)
                    d = os.path.join(args.output_dir, item)
                    if os.path.isdir(s):
                        copy_tree_with_progress(s, d)
                    else:
                        copy_with_progress(s, d)

                try:
                    shutil.rmtree(path)
                except OSError:
                    pass

        # Clean up kagglehub download cache to reclaim disk space
        hub_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / clean_repo_id.replace("/", os.sep)
        if hub_cache.exists():
            shutil.rmtree(hub_cache, ignore_errors=True)

        # Extract any downloaded archives with real-time progress tracking
        from archive_manager import smart_extract
        zip_files = list(Path(args.output_dir).glob("*.zip"))
        for zf in zip_files:
            print(f"[EXTRACT] Unpacking manifold archive: {zf.name}")
            smart_extract(zf, args.output_dir, delete_after=True)

    elif args.action == "upload":
        print(f"PUSHING: {clean_repo_id}")
        src_path = Path(args.output_dir).resolve()
        if not src_path.exists():
            print(f"[ERROR] Target directory does not exist: {src_path}")
            sys.exit(1)

        # Pre-cleanup any dangling staging archives from interrupted runs
        cleanup_temp_archives(manifold_name=src_path.name, base_dir=src_path.parent)

        # Detect pre-upload version state to identify target version
        version_info = get_dataset_version_info(clean_repo_id)
        target_version = (version_info.get("latest_version") or 0) + 1

        file_count = 0
        for _, _, files in os.walk(src_path):
            file_count += len(files)

        staging_dir = None
        if file_count > 50:
            print(f"[SYNC] Detected {file_count} files (>50 threshold). Archiving manifold with progress tracking...")
            manifold_name = src_path.name
            staging_dir = src_path.parent / f".staging_{manifold_name}"
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            staging_dir.mkdir(parents=True, exist_ok=True)

            target_zip = staging_dir / f"{manifold_name}.zip"
            try:
                from archive_manager import create_archive
                success = create_archive(src_path, target_zip, format="zip")
                if not success or not target_zip.exists():
                    raise RuntimeError(f"Archive creation failed for {src_path}")

                zip_size_gb = target_zip.stat().st_size / (1024**3)
                print(f"[SYNC] Archive created: {target_zip.name} ({zip_size_gb:.2f} GB)")
                print(f"[SYNC] Uploading archive to Kaggle via KaggleHub API...")
                kagglehub.dataset_upload(clean_repo_id, str(staging_dir))
            finally:
                if target_zip.exists():
                    try:
                        target_zip.unlink()
                    except OSError:
                        pass
                if staging_dir and staging_dir.exists():
                    print(f"[SYNC] Cleaning up local staging archive...")
                    try:
                        shutil.rmtree(staging_dir, ignore_errors=True)
                    except OSError as ex:
                        print(f"[WARN] Could not remove staging directory: {ex}")
                cleanup_temp_archives(manifold_name=src_path.name, base_dir=src_path.parent)
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


