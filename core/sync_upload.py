"""
Kaggle dataset upload pipeline for the sync core.

Stages a manifold into a resumable zip, uploads it through KaggleHub, pushes
metadata, and optionally waits for Kaggle's server-side extraction.
"""

import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import kagglehub

from core.kaggle_errors import KAGGLE_CLIENT_ERRORS, KAGGLE_REMOTE_ERRORS
from core.sync_auth import is_kaggle_large_manifold
from core.sync_metadata import push_kaggle_dataset_metadata
from core.sync_remote import ensure_kaggle_dataset_exists, get_dataset_version_info
from core.sync_tracking import track_kaggle_dataset_status
from core.sync_transfer import robust_staging_cleanup
from utils.archive import create_archive, verify_archive

logger = logging.getLogger(__name__)

_GCS_BAR_FORMAT = (
    "{desc} - {percentage:3.0f}% [{n_fmt}/{total_fmt}, {elapsed}<{remaining}, {rate_fmt}]"
)


def _verify_or_create_staging_zip(
    src_path: Path,
    target_zip: Path,
    root_datasets_dir: Path,
    file_count: int,
    newest_src_mtime: float
) -> None:
    can_reuse = False
    if target_zip.exists() and target_zip.is_file() and target_zip.stat().st_size > 0:
        print(
            f"[SYNC] Existing staging archive detected ({target_zip.stat().st_size / (1024**3):.2f}"
            " GB). Verifying..."
        )
        if target_zip.stat().st_mtime >= newest_src_mtime and verify_archive(target_zip):
            can_reuse = True
            print(
                "[SYNC] Existing staging archive is valid and up to date. Skipping compression and"
                " resuming upload..."
            )
        else:
            print("[SYNC] Existing archive is outdated or invalid. Re-creating...")
            try:
                target_zip.unlink()
            except OSError as exc:
                print(f"[WARN] Failed unlinking outdated archive {target_zip}: {exc}")

    if not can_reuse:
        print(f"[SYNC] Archiving manifold '{src_path.name}' ({file_count} files)...")
        success = create_archive(
            src_path, target_zip, archive_format="zip", root_dir=root_datasets_dir
        )
        if not success or not target_zip.exists():
            raise RuntimeError(f"Archive creation failed for {src_path}")


def _read_dataset_title(src_path: Path) -> str:
    """Title from dataset-metadata.json, falling back to the folder name."""
    meta_src = src_path / "dataset-metadata.json"
    if meta_src.exists():
        try:
            meta_content = json.loads(meta_src.read_text(encoding="utf-8"))
            return meta_content.get("title", src_path.name)
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Could not read dataset title from %s: %s", meta_src, exc)
    return src_path.name


def _scan_source_tree(src_path: Path) -> tuple[int, float]:
    """File count and newest modification time under ``src_path``."""
    file_count = 0
    newest_src_mtime = 0.0
    for root, _, files in os.walk(src_path):
        file_count += len(files)
        for f in files:
            try:
                newest_src_mtime = max(newest_src_mtime, os.path.getmtime(os.path.join(root, f)))
            except OSError as exc:
                logger.debug("Failed checking mtime for %s: %s", f, exc)
    return file_count, newest_src_mtime


def _patch_gcs_upload_tqdm() -> None:
    """Make kagglehub's GCS upload bar pipe-friendly (fixed width, explicit format)."""
    try:
        gcs_mod = importlib.import_module("kagglehub.gcs_upload")
        orig_tqdm = getattr(gcs_mod, "tqdm", None)
        if orig_tqdm is None:
            return

        def _safe_tqdm(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("dynamic_ncols", False)
            kwargs.setdefault("bar_format", _GCS_BAR_FORMAT)
            return orig_tqdm(*args, **kwargs)

        gcs_mod.tqdm = _safe_tqdm
    except (ImportError, AttributeError) as patch_exc:
        logger.debug("Tqdm patch notice: %s", patch_exc)


def _upload_staging_archive(clean_repo_id: str, staging_dir: Path, target_zip: Path) -> bool:
    """Upload the staging folder; clean it on success, keep it for resumption on failure."""
    upload_success = False
    try:
        print("[SYNC] Uploading archive to Kaggle via KaggleHub API...")
        _patch_gcs_upload_tqdm()
        kagglehub.dataset_upload(clean_repo_id, str(staging_dir))
        upload_success = True
    except KAGGLE_REMOTE_ERRORS as e:
        logger.error("Kaggle upload of %s failed: %s", clean_repo_id, e)
        print(f"[ERROR] Upload failed: {e}")
        print(
            f"[SYNC] Staging archive preserved at {target_zip} for instant resumption on next"
            " attempt."
        )
        return False
    finally:
        if upload_success:
            print("[SYNC] Upload succeeded. Robustly cleaning up local staging archive...")
            robust_staging_cleanup(staging_dir, target_zip)
    return True


def _push_metadata_best_effort(clean_repo_id: str, meta_src: Path) -> None:
    """Push dataset metadata after an upload; failures are reported, not fatal."""
    if not meta_src.exists():
        return
    try:
        print("[SYNC] Pushing Kaggle dataset metadata & column descriptors...")
        push_kaggle_dataset_metadata(clean_repo_id, meta_src)
    except KAGGLE_CLIENT_ERRORS as meta_ex:
        logger.warning("Kaggle metadata update for %s failed: %s", clean_repo_id, meta_ex)
        print(f"[WARN] Kaggle metadata update notice: {meta_ex}")


def _extraction_timeout(zip_size_gb: float, file_count: int) -> int:
    """30 min + 3 min per GB + 2 min per 10k files, capped at 4 hours."""
    size_timeout_sec = int(zip_size_gb * 180)
    file_timeout_sec = (file_count // 10000) * 120
    return max(1800, min(14400, 1800 + size_timeout_sec + file_timeout_sec))


def _stage_archive(src_path: Path) -> tuple[Path, Path, int]:
    """Build (or reuse) ``.staging_<name>/<name>.zip``; returns (staging dir, zip, file count)."""
    file_count, newest_src_mtime = _scan_source_tree(src_path)
    manifold_name = src_path.name
    root_datasets_dir = src_path.parent
    staging_dir = root_datasets_dir / f".staging_{manifold_name}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    target_zip = staging_dir / f"{manifold_name}.zip"
    _verify_or_create_staging_zip(
        src_path, target_zip, root_datasets_dir, file_count, newest_src_mtime
    )
    return staging_dir, target_zip, file_count


def perform_dataset_upload(src_path: Path, clean_repo_id: str, no_wait: bool = False) -> bool:
    """Stage, archive, upload, update metadata, and track a dataset upload to Kaggle."""
    if is_kaggle_large_manifold(clean_repo_id):
        print(f"[SAFEGUARD] Refusing to upload to legacy *large manifold: {clean_repo_id}")
        return False

    src_path = Path(src_path).resolve()
    if not src_path.exists():
        print(f"[ERROR] Target directory does not exist: {src_path}")
        return False

    # Ensure the Kaggle dataset exists before uploading (prevents 403 on metadata update)
    if not ensure_kaggle_dataset_exists(clean_repo_id, title=_read_dataset_title(src_path)):
        print(
            f"[WARN] Could not pre-create Kaggle dataset '{clean_repo_id}'. Upload may still"
            " succeed if dataset exists."
        )

    target_version: int = (get_dataset_version_info(clean_repo_id)["latest_version"] or 0) + 1
    staging_dir, target_zip, file_count = _stage_archive(src_path)

    zip_size_gb = target_zip.stat().st_size / (1024**3)
    print(f"[SYNC] Staging archive ready: {target_zip.name} ({zip_size_gb:.2f} GB)")

    if not _upload_staging_archive(clean_repo_id, staging_dir, target_zip):
        return False
    print("[SUCCESS] Dataset upload payload transferred to Kaggle.")

    _push_metadata_best_effort(clean_repo_id, src_path / "dataset-metadata.json")

    if no_wait:
        return True
    return track_kaggle_dataset_status(
        clean_repo_id,
        target_version=target_version,
        expected_files=file_count,
        timeout=_extraction_timeout(zip_size_gb, file_count),
    )
