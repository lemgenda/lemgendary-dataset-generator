"""
LemGendary Dataset Synchronization & Cloud Transfer Core
========================================================
Centralized Kaggle authentication, archive staging, status tracking,
and resilient transfer utilities shared across dataset managers.
"""

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import TypedDict
import kagglehub
from tqdm import tqdm

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1MB chunks


def copy_with_progress(src, dst):
    """Buffered copy for a single file with uniform tqdm progress tracking."""
    file_size = os.path.getsize(src)
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"FINALIZING: {os.path.basename(src)}",
            colour="cyan"
        ) as pbar:
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
    with tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"FINALIZING: {folder_name}",
        colour="cyan"
    ) as pbar:
        for src_file, dst_file, _ in file_list:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            with open(src_file, "rb") as fsrc, open(dst_file, "wb") as fdst:
                while True:
                    buf = fsrc.read(CHUNK_SIZE)
                    if not buf:
                        break
                    fdst.write(buf)
                    pbar.update(len(buf))


class DatasetVersionInfo(TypedDict):
    current_version: int
    latest_version: int
    versions: dict[int, str]


def setup_kaggle_auth(default_user="lemtreursi"):
    """Ensure Kaggle credentials from environment or .kaggle_token are initialized."""
    _root = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "core" else Path(__file__).resolve().parent
    token_file = _root / ".kaggle_token"
    if not token_file.exists():
        token_file = Path(__file__).parent / ".kaggle_token"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token
            if "KAGGLE_USERNAME" not in os.environ and default_user:
                os.environ["KAGGLE_USERNAME"] = default_user

    try:
        import kagglehub.clients
        kagglehub.clients.already_printed_version_warning = True
    except Exception as exc:
        logger.debug("Kagglehub warning flag suppression skipped: %s", exc)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        KaggleApi.already_printed_version_warning = True
    except Exception as exc:
        logger.debug("KaggleApi warning flag suppression skipped: %s", exc)


def robust_staging_cleanup(staging_dir: Path, target_zip: Path | None = None, max_retries: int = 5) -> bool:
    """Robustly purge staging archive and directory on Windows with GC and retry backoff."""
    import gc
    gc.collect()
    time.sleep(0.5)

    if target_zip and target_zip.exists():
        deleted = False
        for attempt in range(max_retries):
            try:
                target_zip.unlink(missing_ok=True)
                deleted = True
                break
            except OSError:
                gc.collect()
                time.sleep(0.5 * (attempt + 1))
        if not deleted and target_zip.exists():
            print(f"[WARN] Staging archive {target_zip.name} remained locked. Directory purge will attempt removal.")

    if staging_dir.exists():
        deleted_dir = False
        for attempt in range(max_retries):
            try:
                shutil.rmtree(staging_dir, ignore_errors=False)
                deleted_dir = True
                break
            except OSError:
                gc.collect()
                time.sleep(0.5 * (attempt + 1))
        if not deleted_dir and staging_dir.exists():
            print(f"[WARN] Staging directory {staging_dir.name} retained file locks. Scheduled for background sweep.")
            return False
    return True


def cleanup_temp_archives(manifold_name=None, base_dir=None, force=False):
    """Scan and purge dangling staging directories or temporary archives."""
    dirs_to_check = []
    if base_dir:
        dirs_to_check.append(Path(base_dir))
    dirs_to_check.append(Path(r"c:\Development\python\model-training\LemGendaryDatasets"))
    dirs_to_check.append(Path.cwd())
    dirs_to_check.append(Path(os.environ.get("TEMP", r"C:\Users\lemtr\AppData\Local\Temp")))

    cleaned_count = 0
    cleaned_bytes = 0
    now = time.time()
    max_age_seconds = 48 * 3600

    for d in dirs_to_check:
        if not d.exists():
            continue
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
            except Exception as exc:
                logger.debug("Failed removing staging directory %s: %s", p, exc)

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
                except Exception as exc:
                    logger.debug("Failed unlinking temp zip %s: %s", zf, exc)

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
            curr = resp.current_version_number or 0
            raw_versions = resp.versions or []
            valid_versions = [v for v in raw_versions if v is not None and v.version_number is not None]
            max_v = max((v.version_number for v in valid_versions), default=curr)
            versions_dict: dict[int, str] = {
                v.version_number: v.status or ""
                for v in valid_versions
                if v.version_number is not None
            }
            return {
                "current_version": curr,
                "latest_version": max_v,
                "versions": versions_dict,
            }
    except Exception:
        return {
            "current_version": 0,
            "latest_version": 0,
            "versions": {},
        }


def get_dataset_status(repo_id: str) -> str | None:
    """Query current server-side status of a dataset."""
    clean_handle = repo_id.replace("kaggle://", "")
    owner = clean_handle.split("/")[0] if "/" in clean_handle else "lemtreursi"
    setup_kaggle_auth(default_user=owner)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api.dataset_status(clean_handle)
    except Exception as exc:
        logger.debug("Kaggle API dataset_status failed: %s", exc)

    try:
        import subprocess
        res = subprocess.run(
            ["kaggle", "datasets", "status", clean_handle],
            capture_output=True,
            text=True,
            timeout=15,
            check=False
        )
        out = res.stdout.strip().lower()
        if "ready" in out:
            return "ready"
        if "creating" in out or "pending" in out or "queued" in out:
            return "creating"
        if "error" in out or "failed" in out:
            return "error"
        if res.returncode == 0 and out:
            return out.split()[-1]
    except Exception as exc:
        logger.debug("Kaggle CLI status check failed: %s", exc)

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
            if status_val in ["error", "failed"]:
                pbar.set_postfix_str("FAILED")
                pbar.refresh()
                pbar.close()
                print(f"\n[ERROR] Kaggle reports extraction/processing failed for version {target_ver}.")
                return False

            file_pct = None
            if expected_files > 0:
                try:
                    from kagglehub.clients import build_kaggle_client
                    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetFilesSummaryRequest
                    with build_kaggle_client() as client:
                        sreq = ApiGetDatasetFilesSummaryRequest()
                        sreq.owner_slug = owner
                        sreq.dataset_slug = clean_handle.split("/")[1] if "/" in clean_handle else clean_handle
                        sresp = client.datasets.dataset_api_client.get_dataset_files_summary(sreq)
                        summary_files = getattr(sresp, "files", None) or getattr(sresp, "dataset_files", None) or []
                        cur_files = len(summary_files)
                        if cur_files > 0:
                            file_pct = min(99.0, (cur_files / expected_files) * 100.0)
                except Exception as exc:
                    logger.debug("Failed querying Kaggle dataset files summary: %s", exc)

            if file_pct is not None and file_pct > last_pct:
                delta = file_pct - last_pct
                pbar.update(delta)
                last_pct = file_pct
                pbar.set_postfix_str(f"{status_val.upper()} ({int(file_pct)}% files)")
            else:
                simulated_pct = min(95.0, (time.time() - start_time) / 120.0 * 80.0)
                if simulated_pct > last_pct:
                    pbar.update(simulated_pct - last_pct)
                    last_pct = simulated_pct
                pbar.set_postfix_str(status_val.upper())

        time.sleep(poll_interval)

    pbar.close()
    print(f"\n[TIMEOUT] Monitoring timed out after {timeout} seconds. Dataset may still be processing on Kaggle.")
    return False


def _verify_or_create_staging_zip(
    src_path: Path,
    target_zip: Path,
    root_datasets_dir: Path,
    file_count: int,
    newest_src_mtime: float
) -> None:
    try:
        from utils.archive import verify_archive, create_archive
    except ImportError:
        from archive_manager import verify_archive, create_archive

    can_reuse = False
    if target_zip.exists() and target_zip.is_file() and target_zip.stat().st_size > 0:
        print(f"[SYNC] Existing staging archive detected ({target_zip.stat().st_size / (1024**3):.2f} GB). Verifying...")
        if target_zip.stat().st_mtime >= newest_src_mtime and verify_archive(target_zip):
            can_reuse = True
            print("[SYNC] Existing staging archive is valid and up to date. Skipping compression and resuming upload...")
        else:
            print("[SYNC] Existing archive is outdated or invalid. Re-creating...")
            try:
                target_zip.unlink()
            except OSError as exc:
                print(f"[WARN] Failed unlinking outdated archive {target_zip}: {exc}")

    if not can_reuse:
        print(f"[SYNC] Archiving manifold '{src_path.name}' ({file_count} files)...")
        success = create_archive(src_path, target_zip, archive_format="zip", root_dir=root_datasets_dir)
        if not success or not target_zip.exists():
            raise RuntimeError(f"Archive creation failed for {src_path}")


def perform_dataset_upload(src_path: Path, clean_repo_id: str, no_wait: bool = False) -> bool:
    """Stage, archive, upload, update metadata, and track a dataset upload to Kaggle."""
    src_path = Path(src_path).resolve()
    if not src_path.exists():
        print(f"[ERROR] Target directory does not exist: {src_path}")
        return False

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
            except OSError as exc:
                logger.debug("Failed checking mtime for %s: %s", f, exc)

    manifold_name = src_path.name
    root_datasets_dir = src_path.parent
    staging_dir = root_datasets_dir / f".staging_{manifold_name}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    target_zip = staging_dir / f"{manifold_name}.zip"

    _verify_or_create_staging_zip(src_path, target_zip, root_datasets_dir, file_count, newest_src_mtime)

    zip_size_gb = target_zip.stat().st_size / (1024**3)
    print(f"[SYNC] Staging archive ready: {target_zip.name} ({zip_size_gb:.2f} GB)")

    meta_src = src_path / "dataset-metadata.json"
    upload_success = False
    try:
        print("[SYNC] Uploading archive to Kaggle via KaggleHub API...")
        kagglehub.dataset_upload(clean_repo_id, str(staging_dir))
        upload_success = True
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        print(f"[SYNC] Staging archive preserved at {target_zip} for instant resumption on next attempt.")
        return False
    finally:
        if upload_success:
            print("[SYNC] Upload succeeded. Robustly cleaning up local staging archive...")
            robust_staging_cleanup(staging_dir, target_zip)

    print("[SUCCESS] Dataset upload payload transferred to Kaggle.")

    if meta_src.exists():
        try:
            print("[SYNC] Pushing Kaggle dataset metadata & column descriptors...")
            push_kaggle_dataset_metadata(clean_repo_id, meta_src)
        except Exception as meta_ex:
            print(f"[WARN] Kaggle metadata update notice: {meta_ex}")

    if not no_wait:
        success = track_kaggle_dataset_status(
            clean_repo_id,
            target_version=target_version,
            expected_files=file_count
        )
        if not success:
            return False

    return True


def push_kaggle_dataset_metadata(repo_id: str, metadata_path: Path) -> bool:
    """Pushes complete Kaggle metadata including single license and column descriptors."""
    clean_handle = repo_id.replace("kaggle://", "")
    owner = clean_handle.split("/")[0] if "/" in clean_handle else "lemtreursi"
    slug = clean_handle.split("/")[1] if "/" in clean_handle else clean_handle
    setup_kaggle_auth(default_user=owner)

    meta_file = metadata_path if metadata_path.is_file() else metadata_path / "dataset-metadata.json"
    if not meta_file.exists():
        print(f"[ERROR] Metadata file not found at {meta_file}")
        return False

    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiUpdateDatasetMetadataRequest
    from kagglesdk.datasets.types.dataset_types import (
        DatasetSettings,
        DatasetSettingsFile,
        DatasetSettingsFileColumn,
        SettingsLicense,
    )

    api = KaggleApi()
    api.authenticate()

    req = ApiUpdateDatasetMetadataRequest()
    req.owner_slug = owner
    req.dataset_slug = slug
    req.settings = DatasetSettings()
    req.settings.title = meta.get("title", slug)
    req.settings.subtitle = meta.get("subtitle", "")
    req.settings.description = meta.get("description", "")

    # Kaggle strictly requires exactly one license
    lic = SettingsLicense()
    lic.name = "CC0-1.0"
    raw_lic = meta.get("licenses", [])
    if raw_lic and isinstance(raw_lic, list) and isinstance(raw_lic[0], dict) and "name" in raw_lic[0]:
        lic.name = raw_lic[0]["name"]
    req.settings.licenses = [lic]

    if meta.get("keywords"):
        req.settings.keywords = [str(k) for k in meta["keywords"]]

    # Build data array with column descriptions
    files_data = []
    if "data" in meta and isinstance(meta["data"], list):
        for entry in meta["data"]:
            df = DatasetSettingsFile()
            df.name = entry.get("name", "")
            df.description = entry.get("description", "")
            cols = []
            for c in entry.get("columns", []):
                col = DatasetSettingsFileColumn()
                col.name = c.get("name", "")
                col.description = c.get("description", "")
                col.type = c.get("type", "string")
                cols.append(col)
            df.columns = cols
            files_data.append(df)
    elif "resources" in meta and isinstance(meta["resources"], list):
        for r in meta["resources"]:
            df = DatasetSettingsFile()
            df.name = r.get("path", "")
            df.description = r.get("description", "")
            cols = []
            for fld in r.get("schema", {}).get("fields", []):
                col = DatasetSettingsFileColumn()
                col.name = fld.get("name", "")
                col.description = fld.get("description", "")
                col.type = fld.get("type", "string")
                cols.append(col)
            df.columns = cols
            files_data.append(df)

    if files_data:
        req.settings.data = files_data

    client = api.build_kaggle_client()
    resp = client.datasets.dataset_api_client.update_dataset_metadata(req)
    if resp and getattr(resp, "errors", None):
        print(f"[ERROR] Kaggle metadata update rejected: {resp.errors}")
        return False

    print(f"[SUCCESS] Kaggle dataset metadata & column descriptors synced successfully ({len(files_data)} file definitions registered).")
    return True


def _fetch_remote_archive(
    clean_repo_id: str,
    root_datasets_dir: Path,
    target_dir: Path,
    archive_candidates: list[Path],
    is_competition: bool
) -> Path | None:
    """Download archive or directory payload from Kaggle."""
    import subprocess

    if is_competition:
        subprocess.run(["kaggle", "competitions", "download", "-c", clean_repo_id, "-p", str(root_datasets_dir)], check=True)
        for cand in archive_candidates:
            if cand.exists():
                return cand
        return None

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
            subprocess.run(["kaggle", "datasets", "download", clean_repo_id, "-p", str(root_datasets_dir)], check=True)
            downloaded = True
        except Exception as cli_ex:
            print(f"[WARN] Kaggle CLI download also failed: {cli_ex}. Falling back to kagglehub...")
            path = kagglehub.dataset_download(clean_repo_id)
            if os.path.isfile(path):
                return Path(path)
            print(f"[GET] Dataset cached by kagglehub at {path}. Finalizing to {target_dir}...")
            items = os.listdir(path)
            sub_src = os.path.join(path, items[0]) if (len(items) == 1 and items[0] == target_dir.name) else path
            copy_tree_with_progress(sub_src, str(target_dir))
            shutil.rmtree(path, ignore_errors=True)

    if downloaded:
        for cand in archive_candidates:
            if cand.exists() and cand.is_file() and cand.stat().st_size > 0:
                return cand
        recent_zips = sorted(root_datasets_dir.glob("*.zip"), key=lambda f: f.stat().st_mtime, reverse=True)
        if recent_zips:
            return recent_zips[0]

    return None


def perform_dataset_download(
    clean_repo_id: str,
    target_dir: Path,
    root_datasets_dir: Path,
    manifold_name: str,
    slug: str,
    is_competition: bool = False
) -> bool:
    """Download and extract a manifold from Kaggle with full resumption and verification."""
    root_datasets_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    archive_candidates = [
        root_datasets_dir / f"{manifold_name}.zip",
        root_datasets_dir / f"{slug}.zip",
        root_datasets_dir / f"{manifold_name.lower()}.zip",
        root_datasets_dir / f"{slug.lower()}.zip",
        target_dir / f"{manifold_name}.zip",
        target_dir / f"{slug}.zip",
    ]

    existing_archive = None
    try:
        from utils.archive import verify_archive, smart_extract
    except ImportError:
        from archive_manager import verify_archive, smart_extract

    for cand in archive_candidates:
        if cand.exists() and cand.is_file() and cand.stat().st_size > 0:
            print(f"[GET] Inspecting existing archive: {cand.name} ({cand.stat().st_size / (1024**3):.2f} GB)...")
            if verify_archive(cand):
                existing_archive = cand
                print("[GET] Archive verification passed. Resuming extraction directly from local archive...")
                break
            print(f"[WARN] Existing archive {cand.name} is incomplete or corrupted. Re-downloading...")
            try:
                cand.unlink()
            except OSError as exc:
                print(f"[WARN] Failed unlinking corrupt archive {cand}: {exc}")

    archive_to_extract = existing_archive or _fetch_remote_archive(
        clean_repo_id,
        root_datasets_dir,
        target_dir,
        archive_candidates,
        is_competition
    )

    if archive_to_extract and archive_to_extract.exists():
        print(f"[EXTRACT] Unpacking manifold archive: {archive_to_extract.name}")
        success = smart_extract(archive_to_extract, str(target_dir), delete_after=True)
        if not success:
            print(f"[ERROR] Extraction failed or interrupted. Archive preserved at {archive_to_extract} for resumption.")
            return False
        print(f"[SUCCESS] Manifold extracted successfully into {target_dir}")
        return True

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[SUCCESS] Manifold data present in {target_dir}")
        return True

    print(f"[ERROR] Could not locate or download archive for {clean_repo_id}")
    return False
