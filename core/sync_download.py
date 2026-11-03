"""
Kaggle manifold download pipeline for the sync core.

Reuses a verified local archive when one exists, otherwise downloads via the
Kaggle API, then the Kaggle CLI, then kagglehub; legacy loose-image archives
are streamed straight into WebDataset shards, everything else is extracted.
"""

import logging
import os
import shutil
import socket
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import kagglehub
import yaml

from core.kaggle_errors import KAGGLE_CLIENT_ERRORS
from core.lazy_imports import load_kaggle_api
from core.sync_auth import kaggle_cli_creationflags
from core.sync_transfer import copy_tree_with_progress
from tools.stream_zip_to_container import stream_zip_to_webdataset
from utils.archive import smart_extract, verify_archive

logger = logging.getLogger(__name__)

_LOOSE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif")
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


def _archive_candidates(request: DownloadRequest) -> list[Path]:
    """Archive paths a previous or fresh download may have produced, in priority order."""
    root, target = request.root_datasets_dir, request.target_dir
    name, slug = request.manifold_name, request.slug
    return [
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


def _download_via_api_or_cli(request: DownloadRequest) -> bool:
    """Download the dataset zip via the Kaggle API, falling back to the Kaggle CLI."""
    root = request.root_datasets_dir
    try:
        socket.setdefaulttimeout(60.0)
        api = load_kaggle_api()()
        api.authenticate()
        print(f"[GET] Streaming dataset archive directly to {root} via Kaggle API...")
        api.dataset_download_files(request.clean_repo_id, path=str(root), unzip=False, quiet=False)
        return True
    except KAGGLE_CLIENT_ERRORS as ex:
        print(
            f"[WARN] Kaggle API direct download encountered an issue: {ex}. Falling back to Kaggle"
            " CLI..."
        )
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", request.clean_repo_id, "-p", str(root)],
            check=True,
        )
        return True
    except (OSError, subprocess.CalledProcessError) as cli_ex:
        print(f"[WARN] Kaggle CLI download also failed: {cli_ex}. Falling back to kagglehub...")
        return False


def _download_via_kagglehub(request: DownloadRequest) -> Path | None:
    """Last-resort kagglehub download: return a single file, or copy the cached tree."""
    target_dir = request.target_dir
    path = kagglehub.dataset_download(request.clean_repo_id)
    if os.path.isfile(path):
        return Path(path)
    print(f"[GET] Dataset cached by kagglehub at {path}. Finalizing to {target_dir}...")
    items = os.listdir(path)
    sub_src = (
        os.path.join(path, items[0])
        if (len(items) == 1 and items[0] == target_dir.name)
        else path
    )
    copy_tree_with_progress(sub_src, str(target_dir))
    shutil.rmtree(path, ignore_errors=True)
    return None


def _locate_downloaded_archive(root: Path, candidates: list[Path]) -> Path | None:
    """The expected archive, else the most recently modified zip in ``root``."""
    for cand in candidates:
        if _is_nonempty_file(cand):
            return cand
    recent_zips = sorted(root.glob("*.zip"), key=lambda f: f.stat().st_mtime, reverse=True)
    return recent_zips[0] if recent_zips else None


def _fetch_remote_archive(request: DownloadRequest, candidates: list[Path]) -> Path | None:
    """Download archive or directory payload from Kaggle."""
    if request.is_competition:
        return _download_competition(request, candidates)
    if not _download_via_api_or_cli(request):
        return _download_via_kagglehub(request)
    return _locate_downloaded_archive(request.root_datasets_dir, candidates)


def _registry_marks_webdataset(target_dir: Path, slug: str) -> bool:
    """True when unified_data.yaml declares this manifold's canonical format as webdataset."""
    try:
        yd_path = Path(__file__).resolve().parent.parent / "unified_data.yaml"
        if not yd_path.exists():
            return False
        with open(yd_path, "r", encoding="utf-8") as yf:
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


def _should_stream_to_webdataset(archive: Path, target_dir: Path, slug: str) -> bool:
    """Legacy vision zips without .tar shards are streamed straight into WebDataset shards."""
    if not archive.name.endswith(".zip"):
        return False
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            sample_members = [m.filename.replace("\\", "/") for m in zf.infolist()[:1000]]
            has_tar_shards = any(m.endswith(".tar") for m in sample_members)
            loose = _has_loose_images(sample_members)
            # Also cross-reference registry canonical format
            is_webdataset_target = _registry_marks_webdataset(target_dir, slug)
            return (loose or is_webdataset_target) and not has_tar_shards
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as check_exc:
        logger.warning(
            "Archive inspection of %s failed; using standard extraction: %s",
            archive,
            check_exc,
        )
        return False


def _stream_to_webdataset(archive: Path, target_dir: Path) -> bool | None:
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
        outcome = _stream_to_webdataset(archive, target_dir)
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


def perform_dataset_download(request: DownloadRequest) -> bool:
    """Download and extract a manifold from Kaggle with full resumption and verification."""
    request.root_datasets_dir.mkdir(parents=True, exist_ok=True)
    request.target_dir.mkdir(parents=True, exist_ok=True)

    candidates = _archive_candidates(request)
    archive = _find_verified_local_archive(candidates) or _fetch_remote_archive(
        request, candidates
    )
    if archive and archive.exists():
        return _unpack_archive(archive, request)

    target_dir = request.target_dir
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[SUCCESS] Manifold data present in {target_dir}")
        return True

    print(f"[ERROR] Could not locate or download archive for {request.clean_repo_id}")
    return False
