"""
Kaggle dataset upload pipeline for the sync core.

Stages a manifold into a reusable zip, verifies its deterministic SHA-256
manifest, skips upload when the remote dataset already matches (idempotent
push), uploads through KaggleHub with exponential-backoff retries on
transient errors, pushes metadata, and optionally waits for Kaggle's
server-side extraction.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import kagglehub

from core.cli_args import __version__
from core.gcs_upload_chunks import install_large_chunk_upload
from core.kaggle_errors import KAGGLE_CLIENT_ERRORS, KAGGLE_REMOTE_ERRORS
from core.sync_auth import is_kaggle_large_manifold
from core.sync_manifest import (
    SYNC_MANIFEST_FILENAME,
    SYNC_STATE_FILENAME,
    SyncManifest,
    SyncReport,
    compute_sync_manifest,
    is_push_idempotent_noop,
    verify_sync_manifest,
    write_sync_state,
)
from core.sync_metadata import push_kaggle_dataset_metadata
from core.sync_metadata_settings import canonical_license_name
from core.sync_remote import ensure_kaggle_dataset_exists, get_dataset_version_info
from core.sync_tracking import track_kaggle_dataset_status
from core.sync_transfer import robust_staging_cleanup
from core.sync_version_notes import build_version_notes
from utils.archive import create_archive, verify_archive

logger = logging.getLogger(__name__)

_GCS_BAR_FORMAT = (
    "{desc} - {percentage:3.0f}% [{n_fmt}/{total_fmt}, {elapsed}<{remaining}, {rate_fmt}]"
)
_MTIME_IGNORED_FILES = frozenset({SYNC_MANIFEST_FILENAME, SYNC_STATE_FILENAME})


def _verify_or_create_staging_zip(
    src_path: Path,
    target_zip: Path,
    root_datasets_dir: Path,
    file_count: int,
    newest_src_mtime: float,
) -> bool:
    """Verify and reuse an existing staging archive, or create a new one; returns True if reused."""
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
    return can_reuse


def _read_dataset_metadata(src_path: Path) -> dict[str, Any]:
    """The manifold's dataset-metadata.json, or an empty mapping when unreadable."""
    meta_src = src_path / "dataset-metadata.json"
    if meta_src.exists():
        try:
            meta_content = json.loads(meta_src.read_text(encoding="utf-8"))
            if isinstance(meta_content, dict):
                return meta_content
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Could not read dataset metadata from %s: %s", meta_src, exc)
    return {}


def _scan_source_tree(src_path: Path) -> tuple[int, float]:
    """File count and newest source modification time under ``src_path``."""
    file_count = 0
    newest_src_mtime = 0.0
    for root, _, files in os.walk(src_path):
        for f in files:
            if f == SYNC_STATE_FILENAME:
                continue
            file_count += 1
            if f in _MTIME_IGNORED_FILES:
                continue
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

        vars(gcs_mod)["tqdm"] = _safe_tqdm
    except (ImportError, AttributeError) as patch_exc:
        logger.debug("Tqdm patch notice: %s", patch_exc)


def _upload_staging_archive(
    clean_repo_id: str,
    staging_dir: Path,
    target_zip: Path,
    version_notes: str,
    *,
    retry_opts: dict[str, Any] | None = None,
) -> tuple[bool, int]:
    """Upload the staging folder as a new version labelled ``version_notes``.

    Retries transient remote/network failures up to ``max_retries`` times with
    exponential backoff while preserving ``target_zip``. The staging folder is
    cleaned on success; on permanent failure the zip is kept so the next
    attempt resumes directly from the verified archive without re-compressing.
    """
    ropts = retry_opts or {}
    max_retries = int(ropts.get("max_retries", 3))
    retry_backoff_s = float(ropts.get("retry_backoff_s", 1.0))
    sleep: Callable[[float], None] = ropts.get("sleep", time.sleep)
    upload_success = False
    retries_used = 0
    attempts = max(1, max_retries)
    try:
        print("[SYNC] Uploading archive to Kaggle via KaggleHub API...")
        print(f"[SYNC] Version notes: {version_notes}")
        _patch_gcs_upload_tqdm()
        install_large_chunk_upload()

        for attempt in range(attempts):
            try:
                kagglehub.dataset_upload(
                    clean_repo_id, str(staging_dir), version_notes=version_notes
                )
                upload_success = True
                break
            except KAGGLE_REMOTE_ERRORS as exc:
                if attempt + 1 < attempts:
                    retries_used += 1
                    wait_s = retry_backoff_s * (2**attempt)
                    logger.warning(
                        "Kaggle upload attempt %d/%d of %s interrupted (%s); "
                        "retrying in %.2fs with staging archive intact...",
                        attempt + 1,
                        attempts,
                        clean_repo_id,
                        exc,
                        wait_s,
                    )
                    print(
                        f"[WARN] Upload attempt {attempt + 1}/{attempts} interrupted ({exc}); "
                        f"retrying in {wait_s:.2f}s..."
                    )
                    sleep(wait_s)
                else:
                    logger.error("Kaggle upload of %s failed: %s", clean_repo_id, exc)
                    print(f"[ERROR] Upload failed: {exc}")
                    print(
                        f"[SYNC] Staging archive kept at {target_zip}: the next attempt reuses it"
                        " without re-compressing."
                    )
                    return False, retries_used
    finally:
        if upload_success:
            print("[SYNC] Upload succeeded. Robustly cleaning up local staging archive...")
            robust_staging_cleanup(staging_dir, target_zip)
    return True, retries_used


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


def _stage_archive(src_path: Path) -> tuple[Path, Path, int, bool]:
    """Build (or reuse) ``.staging_<name>/<name>.zip``; returns (dir, zip, file count, reused)."""
    file_count, newest_src_mtime = _scan_source_tree(src_path)
    manifold_name = src_path.name
    root_datasets_dir = src_path.parent
    staging_dir = root_datasets_dir / f".staging_{manifold_name}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    target_zip = staging_dir / f"{manifold_name}.zip"
    reused = _verify_or_create_staging_zip(
        src_path, target_zip, root_datasets_dir, file_count, newest_src_mtime
    )
    return staging_dir, target_zip, file_count, reused


def _populate_report(
    report: SyncReport | None,
    *,
    status: str,
    start_time: float = 0.0,
    **fields: Any,
) -> None:
    if report is None:
        return
    report.update(
        status=status,
        elapsed_seconds=round(max(0.0, time.monotonic() - start_time), 3),
        **fields,
    )


def _prepare_upload_manifest(
    src_path: Path,
    clean_repo_id: str,
    t0: float,
    report_out: SyncReport | None,
) -> SyncManifest | None:
    """Compute and verify sync_manifest.json and ensure remote dataset exists."""
    manifest = compute_sync_manifest(
        src_path, compiler_version=__version__, write_file=True
    )
    verified, manifest_sha, verify_errors = verify_sync_manifest(
        src_path, expected_sha256=manifest.manifest_sha256
    )
    if not verified:
        logger.error(
            "Pre-push manifest checksum verification failed for %s: %s",
            src_path,
            verify_errors,
        )
        print(f"[ERROR] Pre-push manifest checksum verification failed: {verify_errors}")
        _populate_report(
            report_out,
            status="failed",
            manifest_sha256=manifest_sha,
            checksum_verified=False,
            start_time=t0,
            errors=verify_errors,
        )
        return None

    meta = _read_dataset_metadata(src_path)
    if not ensure_kaggle_dataset_exists(
        clean_repo_id,
        title=meta.get("title", src_path.name),
        license_name=canonical_license_name(meta),
    ):
        print(
            f"[WARN] Could not pre-create Kaggle dataset '{clean_repo_id}'. Upload may still"
            " succeed if dataset exists."
        )
    return manifest


def _execute_staged_upload(
    src_path: Path,
    clean_repo_id: str,
    manifest: SyncManifest,
    target_version: int,
    options: dict[str, Any],
) -> bool:
    """Stage archive, upload with retry, push metadata, and optionally track extraction."""
    t0 = float(options.get("t0", time.monotonic()))
    staging_dir, target_zip, file_count, reused = _stage_archive(src_path)
    zip_bytes = target_zip.stat().st_size if target_zip.exists() else 0
    print(f"[SYNC] Staging archive ready: {target_zip.name} ({zip_bytes / (1024**3):.2f} GB)")

    upload_ok, retries_used = _upload_staging_archive(
        clean_repo_id,
        staging_dir,
        target_zip,
        build_version_notes(src_path, options.get("notes")),
        retry_opts=options,
    )
    if not upload_ok:
        _populate_report(
            options.get("report_out"),
            status="failed",
            version=target_version,
            manifest_sha256=manifest.manifest_sha256,
            checksum_verified=True,
            resumed=reused,
            retries=retries_used,
            files_count=file_count,
            bytes_transferred=0,
            start_time=t0,
            errors=[f"Upload failed for {clean_repo_id} after {retries_used + 1} attempt(s)"],
        )
        return False
    print("[SUCCESS] Dataset upload payload transferred to Kaggle.")

    write_sync_state(src_path, clean_repo_id, manifest.manifest_sha256, target_version)
    _push_metadata_best_effort(clean_repo_id, src_path / "dataset-metadata.json")

    tracked = True
    if not options.get("no_wait", False):
        tracked = track_kaggle_dataset_status(
            clean_repo_id,
            target_version=target_version,
            expected_files=file_count,
            timeout=_extraction_timeout(zip_bytes / (1024**3), file_count),
        )
    _populate_report(
        options.get("report_out"),
        status="passed" if tracked else "failed",
        version=target_version,
        manifest_sha256=manifest.manifest_sha256,
        checksum_verified=True,
        resumed=reused,
        retries=retries_used,
        files_count=file_count,
        bytes_transferred=zip_bytes,
        start_time=t0,
        errors=[] if tracked else [f"Extraction tracking timed out or failed for {clean_repo_id}"],
    )
    return tracked


def perform_dataset_upload(
    src_path: Path,
    clean_repo_id: str,
    no_wait: bool = False,
    notes: str | None = None,
    **options: Any,
) -> bool:
    """Stage, archive, upload, update metadata, and track a dataset upload to Kaggle.

    ``notes`` describes the version; it follows the generated version notes.
    When ``force=False`` and the local manifest checksum matches the already-synced
    remote version, the upload is skipped as an idempotent no-op.
    """
    t0 = time.monotonic()
    report_out: SyncReport | None = options.get("report_out")
    if is_kaggle_large_manifold(clean_repo_id):
        msg = f"Refusing to upload to legacy *large manifold: {clean_repo_id}"
        print(f"[SAFEGUARD] {msg}")
        _populate_report(report_out, status="failed", start_time=t0, errors=[msg])
        return False

    src_path = Path(src_path).resolve()
    if not src_path.exists():
        msg = f"Target directory does not exist: {src_path}"
        print(f"[ERROR] {msg}")
        _populate_report(report_out, status="failed", start_time=t0, errors=[msg])
        return False

    manifest = _prepare_upload_manifest(src_path, clean_repo_id, t0, report_out)
    if manifest is None:
        return False

    version_info = get_dataset_version_info(clean_repo_id)
    latest_version = int(version_info.get("latest_version") or 0)

    if not options.get("force", False) and is_push_idempotent_noop(
        src_path, clean_repo_id, manifest.manifest_sha256, version_info
    ):
        logger.info(
            "Idempotent push no-op for %s (%s): manifest_sha256=%s already matches v%d",
            src_path.name,
            clean_repo_id,
            manifest.manifest_sha256,
            latest_version,
        )
        print(
            f"[SYNC] Idempotent no-op: remote dataset '{clean_repo_id}' (v{latest_version}) "
            f"already matches manifest_sha256={manifest.manifest_sha256[:12]}..."
        )
        _populate_report(
            report_out,
            status="noop",
            version=latest_version,
            manifest_sha256=manifest.manifest_sha256,
            checksum_verified=True,
            idempotent_noop=True,
            files_count=manifest.file_count,
            bytes_transferred=0,
            start_time=t0,
        )
        return True

    merged_opts = {**options, "no_wait": no_wait, "notes": notes, "t0": t0}
    return _execute_staged_upload(
        src_path, clean_repo_id, manifest, latest_version + 1, merged_opts
    )
