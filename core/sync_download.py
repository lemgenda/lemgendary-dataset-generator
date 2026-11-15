"""
Kaggle manifold download pipeline for the sync core.

Reuses a verified local archive when one exists, otherwise downloads via the
Kaggle API, then the Kaggle CLI, then kagglehub; legacy loose-image archives
are streamed straight into WebDataset shards, everything else is extracted.
When Kaggle has no whole-dataset archive (a very large dataset), the files
are downloaded one by one instead; ``only`` fetches just some folders or
files (:mod:`core.sync_partial_download`).
"""

import logging
import os
import shutil
import socket
import subprocess
import tarfile
import time
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import kagglehub
import yaml

from core.kaggle_errors import KAGGLE_CLIENT_ERRORS
from core.lazy_imports import load_kaggle_api
from core.sync_auth import kaggle_cli_creationflags
from core.sync_manifest import SyncReport, verify_sync_manifest
from core.sync_partial_download import (
    download_file_by_file,
    download_selected,
    is_missing_archive_error,
)
from core.sync_remote import parse_version_pin, resolve_pinned_version
from core.sync_transfer import copy_tree_with_progress
from tools.stream_zip_to_container import stream_zip_to_webdataset
from utils.archive import smart_extract, verify_archive
from utils.shard_audit import find_shards

logger = logging.getLogger(__name__)

_LOOSE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif")
# Members that mean the archive already holds a modern container: extract it as is.
_CONTAINER_SUFFIXES = (".tar", ".data", ".parquet", ".mds", ".mds.zstd")
_STREAM_ERRORS = (ImportError, OSError, ValueError, zipfile.BadZipFile, tarfile.TarError)


@dataclass(frozen=True)
class DownloadRequest:
    """Where a Kaggle manifold comes from and where it lands locally."""
    clean_repo_id: str
    target_dir: Path
    root_datasets_dir: Path
    manifold_name: str
    slug: str
    is_competition: bool = False
    label_json: str | None = None  # JSON sidecar strategy for text labels (utils.label_json)


class NoWholeArchiveError(RuntimeError):
    """Kaggle has no whole-dataset archive for this dataset (download answered 404)."""


def _archive_candidates(
    request: DownloadRequest, resolved_version: int | None = None
) -> list[Path]:
    """Archive paths a previous or fresh download may have produced, in priority order."""
    root, target = request.root_datasets_dir, request.target_dir
    name, slug = request.manifold_name, request.slug
    versioned: list[Path] = []
    if resolved_version is not None:
        versioned = [
            root / f"{name}_v{resolved_version}.zip",
            root / f"{slug}_v{resolved_version}.zip",
            target / f"{name}_v{resolved_version}.zip",
            target / f"{slug}_v{resolved_version}.zip",
        ]
    return [
        *versioned,
        root / f"{name}.zip",
        root / f"{slug}.zip",
        root / f"{name.lower()}.zip",
        root / f"{slug.lower()}.zip",
        target / f"{name}.zip",
        target / f"{slug}.zip",
    ]


def _is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _find_verified_local_archive(candidates: list[Path]) -> Path | None:
    """First candidate archive that passes verification; corrupt ones are deleted."""
    for cand in candidates:
        if not _is_nonempty_file(cand):
            continue
        print(
            f"[GET] Inspecting existing archive: {cand.name}"
            f" ({cand.stat().st_size / (1024**3):.2f} GB)..."
        )
        if verify_archive(cand):
            print(
                "[GET] Archive verification passed. Resuming extraction directly from local"
                " archive..."
            )
            return cand
        print(f"[WARN] Existing archive {cand.name} is incomplete or corrupted. Re-downloading...")
        try:
            cand.unlink()
        except OSError as exc:
            print(f"[WARN] Failed unlinking corrupt archive {cand}: {exc}")
    return None


def _download_competition(request: DownloadRequest, candidates: list[Path]) -> Path | None:
    """Download competition files with the Kaggle CLI and return the archive it wrote."""
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            request.clean_repo_id,
            "-p",
            str(request.root_datasets_dir),
        ],
        check=True,
        creationflags=kaggle_cli_creationflags(),
    )
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _download_via_api_or_cli(
    request: DownloadRequest, resolved_version: int | None = None
) -> bool:
    """Download the dataset zip via the Kaggle API, falling back to the Kaggle CLI.

    Raises :class:`NoWholeArchiveError` when Kaggle has no archive to download.
    """
    root = request.root_datasets_dir
    if resolved_version is None:
        try:
            socket.setdefaulttimeout(60.0)
            api = load_kaggle_api()()
            api.authenticate()
            print(f"[GET] Streaming dataset archive directly to {root} via Kaggle API...")
            api.dataset_download_files(
                request.clean_repo_id, path=str(root), unzip=False, quiet=False
            )
            return True
        except KAGGLE_CLIENT_ERRORS as ex:
            if is_missing_archive_error(ex):
                raise NoWholeArchiveError(str(ex)) from ex
            print(
                f"[WARN] Kaggle API direct download encountered an issue: {ex}. Falling back to"
                " Kaggle CLI..."
            )
    try:
        cli_cmd = ["kaggle", "datasets", "download", request.clean_repo_id, "-p", str(root)]
        if resolved_version is not None:
            cli_cmd.extend(["-v", str(resolved_version)])
        subprocess.run(cli_cmd, check=True)
        return True
    except (OSError, subprocess.CalledProcessError) as cli_ex:
        print(f"[WARN] Kaggle CLI download also failed: {cli_ex}. Falling back to kagglehub...")
        return False


def _log_cache_cleanup_error(_func: object, failed_path: str, exc: BaseException) -> None:
    """Report a kagglehub cache entry that could not be removed after the copy."""
    logger.warning("Could not remove kagglehub cache entry %s: %s", failed_path, exc)


@dataclass(frozen=True)
class _Fetched:
    """Outcome of a remote fetch: an archive to unpack, or a tree copied into the target."""

    archive: Path | None = None
    tree_copied: bool = False


def _download_via_kagglehub(
    request: DownloadRequest, resolved_version: int | None = None
) -> _Fetched:
    """Last-resort kagglehub download: return a single file, or copy the cached tree."""
    target_dir = request.target_dir
    handle = (
        f"{request.clean_repo_id}/versions/{resolved_version}"
        if resolved_version is not None
        else request.clean_repo_id
    )
    path = kagglehub.dataset_download(handle)
    if os.path.isfile(path):
        return _Fetched(archive=Path(path))
    print(f"[GET] Dataset cached by kagglehub at {path}. Finalizing to {target_dir}...")
    items = os.listdir(path)
    sub_src = (
        os.path.join(path, items[0])
        if (len(items) == 1 and items[0] == target_dir.name)
        else path
    )
    copy_tree_with_progress(sub_src, str(target_dir))
    shutil.rmtree(path, onexc=_log_cache_cleanup_error)
    return _Fetched(tree_copied=True)


def _locate_downloaded_archive(request: DownloadRequest, candidates: list[Path]) -> Path | None:
    """The archive this dataset's download wrote (one of ``candidates``), else None.

    Kaggle names the download ``<slug>.zip``. Any other zip in the datasets
    root belongs to another download and is never picked up.
    """
    for cand in candidates:
        if _is_nonempty_file(cand):
            return cand
    print(
        f"[ERROR] Download of {request.clean_repo_id} finished but none of the expected"
        f" archives exists: {', '.join(c.name for c in candidates)}"
    )
    return None


def _fetch_remote_archive(
    request: DownloadRequest,
    candidates: list[Path],
    resolved_version: int | None = None,
) -> _Fetched:
    """Download archive or directory payload from Kaggle."""
    if request.is_competition:
        return _Fetched(archive=_download_competition(request, candidates))
    try:
        if resolved_version is None:
            downloaded = _download_via_api_or_cli(request)
        else:
            downloaded = _download_via_api_or_cli(request, resolved_version=resolved_version)
    except NoWholeArchiveError:
        return _Fetched(
            tree_copied=download_file_by_file(request.clean_repo_id, request.target_dir)
        )
    if not downloaded:
        if resolved_version is None:
            return _download_via_kagglehub(request)
        return _download_via_kagglehub(request, resolved_version=resolved_version)
    return _Fetched(archive=_locate_downloaded_archive(request, candidates))


def _registry_marks_webdataset(target_dir: Path, slug: str) -> bool:
    """True when unified_data.yaml declares this manifold's canonical format as webdataset."""
    try:
        yd_path = Path(__file__).resolve().parent.parent / "unified_data.yaml"
        if not yd_path.exists():
            return False
        with open(yd_path, encoding="utf-8") as yf:
            ydata = yaml.safe_load(yf) or {}
        for entry in ydata.get("datasets", {}).values():
            if entry.get("canonical_format") != "webdataset":
                continue
            m_folder = entry.get("modernized_folder", "")
            k_ref = entry.get("kaggle_ref", "")
            if (m_folder and m_folder.lower() in target_dir.name.lower()) or (
                k_ref and slug.lower() in k_ref.lower()
            ):
                return True
    except (OSError, yaml.YAMLError) as exc:
        logger.debug("Could not read unified_data.yaml during download inspection: %s", exc)
    return False


def _has_loose_images(members: list[str]) -> bool:
    """True when sampled members look like a legacy loose image tree."""
    return any(
        m.lower().endswith(_LOOSE_IMAGE_SUFFIXES)
        or "/images/" in m.lower()
        or "/targets/" in m.lower()
        for m in members
    )


def _is_container_member(name: str) -> bool:
    """True for shard, Parquet or MDS members, and JSONL splits under ``data/`` (LLM layout).

    A root-level ``.jsonl`` (``rejected_records.jsonl``) is metadata, not data.
    """
    if name.endswith(_CONTAINER_SUFFIXES):
        return True
    return name.endswith(".jsonl") and (name.startswith("data/") or "/data/" in name)


def _should_stream_to_webdataset(archive: Path, target_dir: Path, slug: str) -> bool:
    """Legacy vision zips are streamed straight into WebDataset shards.

    Not when the zip already holds a container (WebDataset ``.tar``/``.data``
    shards, Parquet, JSONL or MDS): re-streaming would convert those again.
    Not when ``target_dir`` already has shards either: the new shards would
    be mixed with (and partly overwrite) the existing set. Both cases fall
    back to plain extraction.
    """
    if not archive.name.endswith(".zip"):
        return False
    if find_shards(target_dir / "shards"):
        print(f"[STREAM] {target_dir.name} already has shards; extracting without streaming.")
        return False
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            members = [m.filename.replace("\\", "/").lower() for m in zf.infolist()]
        if any(_is_container_member(m) for m in members):
            return False
        # Also cross-reference registry canonical format
        return _has_loose_images(members) or _registry_marks_webdataset(target_dir, slug)
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as check_exc:
        logger.warning(
            "Archive inspection of %s failed; using standard extraction: %s",
            archive,
            check_exc,
        )
        return False


def _stream_to_webdataset(
    archive: Path, target_dir: Path, label_json: str | None = None
) -> bool | None:
    """Convert the zip into WebDataset shards; None means fall back to plain extraction."""
    print(f"[STREAM] Discovered legacy image matrix in {archive.name}.")
    print(
        "[STREAM] Converting directly into modern WebDataset (.tar) shards via in-memory"
        " streaming..."
    )
    try:
        success = stream_zip_to_webdataset(
            zip_path=archive,
            target_dir=target_dir,
            shard_size_samples=5000,
            delete_zip=True,
            transcode_webp=True,
            label_json=label_json,
        )
    except _STREAM_ERRORS as stream_err:
        print(
            f"[WARN] Direct stream-to-container encountered error: {stream_err}. Falling"
            " back to standard extraction..."
        )
        return None
    if not success:
        print(f"[ERROR] Streaming conversion failed. Archive preserved at {archive}.")
        return False
    print(f"[SUCCESS] Manifold modernized directly into WebDataset shards at {target_dir}")
    return True


def _unpack_archive(archive: Path, request: DownloadRequest) -> bool:
    """Stream or extract ``archive`` into the request's target directory."""
    target_dir = request.target_dir
    if _should_stream_to_webdataset(archive, target_dir, request.slug):
        outcome = _stream_to_webdataset(archive, target_dir, request.label_json)
        if outcome is not None:
            return outcome

    print(f"[EXTRACT] Unpacking manifold archive: {archive.name}")
    if not smart_extract(archive, str(target_dir), delete_after=True):
        print(
            "[ERROR] Extraction failed or interrupted. Archive preserved at"
            f" {archive} for resumption."
        )
        return False
    print(f"[SUCCESS] Manifold extracted successfully into {target_dir}")
    return True


def _populate_pull_report(
    report: SyncReport | None,
    *,
    request: DownloadRequest,
    status: str,
    elapsed: float,
    **fields: object,
) -> None:
    """Fill ``report`` in-place if requested by caller."""
    if report is None:
        return
    report.update(
        action="pull",
        repo_id=request.clean_repo_id,
        manifold=request.manifold_name,
        status=status,
        elapsed_seconds=round(elapsed, 4),
        **fields,
    )


def _run_selected_download(
    clean_req: DownloadRequest,
    only: tuple[str, ...],
    resolved_version: int | None,
) -> tuple[bool, str, list[str]]:
    """Download only ``only`` prefixes and verify their checksums in sync_manifest.json."""
    if resolved_version is None:
        selected_ok = download_selected(clean_req.clean_repo_id, clean_req.target_dir, only)
    else:
        selected_ok = download_selected(
            clean_req.clean_repo_id, clean_req.target_dir, only, version=resolved_version
        )
    if not selected_ok:
        return False, "", ["Selected member download failed"]
    manifest_ok, manifest_sha, manifest_errs = verify_sync_manifest(
        clean_req.target_dir, only=only
    )
    if not manifest_ok:
        for err in manifest_errs:
            print(f"[ERROR] Manifest verification failed: {err}")
    return manifest_ok, manifest_sha, manifest_errs


def _extract_full_download(
    clean_req: DownloadRequest, resolved_version: int | None
) -> tuple[bool, bool]:
    """Fetch and extract the full archive or file tree; returns (extracted_ok, resumed)."""
    candidates = _archive_candidates(clean_req, resolved_version=resolved_version)
    local_archive = (
        _find_verified_local_archive(candidates) if resolved_version is None else None
    )
    fetched = (
        _Fetched(archive=local_archive)
        if local_archive is not None
        else _fetch_remote_archive(clean_req, candidates, resolved_version=resolved_version)
    )
    extracted_ok = False
    if fetched.archive is not None and fetched.archive.is_file():
        extracted_ok = _unpack_archive(fetched.archive, clean_req)
    elif fetched.tree_copied and any(clean_req.target_dir.iterdir()):
        print(f"[SUCCESS] Manifold data present in {clean_req.target_dir}")
        extracted_ok = True
    else:
        print(f"[ERROR] Could not locate or download archive for {clean_req.clean_repo_id}")
    return extracted_ok, local_archive is not None


def _resolve_request_version(
    request: DownloadRequest,
    version: int | str | None,
    start_t: float,
    report_out: SyncReport | None,
) -> tuple[DownloadRequest, int | None, int | str | None] | None:
    """Resolve and validate the requested version pin; returns None when invalid."""
    base_repo_id, effective_pin = parse_version_pin(request.clean_repo_id, version)
    clean_req = (
        replace(request, clean_repo_id=base_repo_id)
        if base_repo_id != request.clean_repo_id
        else request
    )
    try:
        resolved_version = resolve_pinned_version(base_repo_id, effective_pin)
    except ValueError as pin_err:
        msg = f"Invalid version pin for {base_repo_id}: {pin_err}"
        print(f"[ERROR] {msg}")
        logger.error(msg)
        _populate_pull_report(
            report_out,
            request=clean_req,
            status="failed",
            elapsed=time.monotonic() - start_t,
            version=None,
            pinned_version=effective_pin,
            manifest_sha256="",
            checksum_verified=False,
            resumed=False,
            errors=[msg],
        )
        return None
    if resolved_version is not None:
        print(f"[GET] Pinned to dataset version v{resolved_version} for {base_repo_id}")
    return clean_req, resolved_version, effective_pin


def _verify_and_report_full_pull(
    clean_req: DownloadRequest,
    resolved_version: int | None,
    effective_pin: int | str | None,
    start_t: float,
    report_out: SyncReport | None,
) -> bool:
    """Extract full archive and verify sync_manifest.json."""
    extracted_ok, resumed = _extract_full_download(clean_req, resolved_version)
    if not extracted_ok:
        _populate_pull_report(
            report_out,
            request=clean_req,
            status="failed",
            elapsed=time.monotonic() - start_t,
            version=resolved_version,
            pinned_version=effective_pin,
            manifest_sha256="",
            checksum_verified=False,
            resumed=resumed,
            errors=[f"Download or extraction failed for {clean_req.clean_repo_id}"],
        )
        return False

    manifest_ok, manifest_sha, manifest_errs = verify_sync_manifest(clean_req.target_dir)
    if not manifest_ok:
        for err in manifest_errs:
            print(f"[ERROR] Post-pull manifest checksum verification failed: {err}")
            logger.error("Post-pull manifest checksum verification failed: %s", err)
    _populate_pull_report(
        report_out,
        request=clean_req,
        status="passed" if manifest_ok else "failed",
        elapsed=time.monotonic() - start_t,
        version=resolved_version,
        pinned_version=effective_pin,
        manifest_sha256=manifest_sha,
        checksum_verified=manifest_ok,
        resumed=resumed,
        errors=manifest_errs,
    )
    return manifest_ok


def perform_dataset_download(
    request: DownloadRequest,
    only: tuple[str, ...] = (),
    *,
    version: int | str | None = None,
    report_out: SyncReport | None = None,
) -> bool:
    """Download and extract a manifold from Kaggle with full resumption and verification.

    ``only`` (folders ending in ``/`` or files, relative to the dataset's
    wrapper folder) fetches just those members, never the whole archive.
    """
    start_t = time.monotonic()
    resolved_info = _resolve_request_version(request, version, start_t, report_out)
    if resolved_info is None:
        return False
    clean_req, resolved_version, effective_pin = resolved_info

    clean_req.root_datasets_dir.mkdir(parents=True, exist_ok=True)
    clean_req.target_dir.mkdir(parents=True, exist_ok=True)
    if only:
        sel_ok, sel_sha, sel_errs = _run_selected_download(clean_req, only, resolved_version)
        _populate_pull_report(
            report_out,
            request=clean_req,
            status="passed" if sel_ok else "failed",
            elapsed=time.monotonic() - start_t,
            version=resolved_version,
            pinned_version=effective_pin,
            manifest_sha256=sel_sha,
            checksum_verified=sel_ok,
            resumed=False,
            errors=sel_errs,
        )
        return sel_ok

    return _verify_and_report_full_pull(
        clean_req, resolved_version, effective_pin, start_t, report_out
    )
