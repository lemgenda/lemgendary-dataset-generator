"""Hugging Face source: download a dataset repository or single file.

Run from the project root as ``python -m sources.hf``.
"""
import argparse
import logging
import os
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, hf_hub_url, list_repo_files, login, snapshot_download
# 2026 Resilience: Force ASCII progress bars globally to prevent Unicode Mojibake in PowerShell
# Removed TQDM_ASCII override to allow Unicode block characters
from tqdm import tqdm

from utils.archive import smart_extract

logger = logging.getLogger(__name__)

# Expected failures from HTTP transfer (requests/huggingface_hub errors are OSError
# subclasses), bad headers/arguments, and archive extraction.
_DOWNLOAD_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    RuntimeError,
    EOFError,
    tarfile.TarError,
    zipfile.BadZipFile,
)

_ARCHIVE_SUFFIXES = (".tgz", ".tar.gz", ".zip")
# IQA archives pulled on their own when a repository carries them.
_HUMAN_TECHNICAL_TARGETS = ("spaq.tgz", "live.tgz", "csiq.tgz", "tid2013.tgz")
_DL_BAR_FORMAT = "{desc}: {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


@dataclass(frozen=True)
class _HubRepo:
    """The repository being pulled and the credentials to pull it with."""

    repo_id: str
    repo_type: str
    token: str | None


def _read_token(token_file: str) -> str | None:
    """Return the token stored in ``token_file`` (logging in with it), or None."""
    token = None
    if os.path.exists(token_file):
        with open(token_file, "r", encoding="utf-8") as f:
            token = f.read().strip()
            if token:
                login(token=token)
                print(f"[AUTH] Logged into Hugging Face via {token_file}")
    return token


def _select_targets(repo: _HubRepo, target_file: str | None) -> list[str]:
    """Files to pull: the surgical target, the IQA archives, else all archives/parquets."""
    if target_file:
        return [target_file]
    # Get list of files to see if we should be surgical
    all_files = list_repo_files(repo_id=repo.repo_id, repo_type=repo.repo_type, token=repo.token)
    targets = [f for f in all_files if f.lower() in _HUMAN_TECHNICAL_TARGETS]
    if not targets:
        print(
            "[HF] No specific custom/IQA targets found. Falling back to full archives &"
            " parquets..."
        )
        targets = [f for f in all_files if f.endswith((*_ARCHIVE_SUFFIXES, ".parquet"))]
    return targets


def _dedicated_dir(output_dir: str, t_file: str) -> Path:
    """Archives extract into ``output_dir/<archive stem>``; other files land in output_dir."""
    if t_file.endswith(_ARCHIVE_SUFFIXES):
        # Use Path(t_file).name to strip any repo subdirectories (like data/)
        slug = Path(t_file).name.replace(".tgz", "").replace(".tar.gz", "").replace(".zip", "")
        # Extract inside the actual dataset folder (output_dir), not its parent
        return Path(output_dir) / slug
    return Path(output_dir)


def _http_download(repo: _HubRepo, t_file: str, path: Path) -> None:
    """Resumable download of ``t_file`` to ``path`` via requests + tqdm."""
    url = hf_hub_url(repo_id=repo.repo_id, filename=t_file, repo_type=repo.repo_type)
    headers = {"Authorization": f"Bearer {repo.token}"} if repo.token else {}

    existing_size = 0
    if path.exists():
        existing_size = path.stat().st_size
        headers["Range"] = f"bytes={existing_size}-"
        print(
            f"  [RESUME] Found partial file ({existing_size / (1024 * 1024):.2f}"
            " MB). Picking up..."
        )

    response = requests.get(url, headers=headers, stream=True, timeout=60)
    # If 206 Partial Content, we are resuming
    total_size = int(response.headers.get("content-length", 0))
    if response.status_code == 200 and existing_size > 0:
        print("  [WARN] Server ignored resume request. Overwriting from start.")
        existing_size = 0
    elif response.status_code == 206:
        total_size += existing_size

    mode = "ab" if existing_size > 0 else "wb"
    with (
        open(path, mode) as f,
        tqdm(
            desc=f"   [DL] {t_file}",
            total=total_size,
            initial=existing_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            bar_format=_DL_BAR_FORMAT,
        ) as pbar,
    ):
        for data in response.iter_content(chunk_size=1024 * 1024):
            pbar.update(f.write(data))


def _pull_file(repo: _HubRepo, t_file: str, output_dir: str) -> None:
    """Download (and auto-extract) one file, falling back to hf_hub_download."""
    dedicated_dir = _dedicated_dir(output_dir, t_file)
    dedicated_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [PULL] {t_file} -> {dedicated_dir}")

    # High-Visibility Download via requests + tqdm (Resumable)
    try:
        path = dedicated_dir / t_file
        _http_download(repo, t_file, path)
        # Auto-Extract to dedicated_dir
        if t_file.endswith(_ARCHIVE_SUFFIXES):
            smart_extract(path, dedicated_dir, delete_after=True)
    except _DOWNLOAD_ERRORS as e:
        logger.warning(
            "Direct HTTP download of %s failed; falling back to hf_hub_download: %s",
            t_file,
            e,
        )
        print(f"  [WARN] Requests failed, falling back to basic HF-Hub: {e}")
        hf_hub_download(
            repo_id=repo.repo_id,
            filename=t_file,
            repo_type=repo.repo_type,
            local_dir=dedicated_dir,
            token=repo.token,
        )


def _sweep_archives(output_dir: str) -> None:
    """2026 Resilience: extract any archive left behind (e.g. by a snapshot download)."""
    print(f"[SWEEP] Scanning {output_dir} for unextracted archives...")
    for root, _dirs, files in os.walk(output_dir):
        for f in files:
            f_path = Path(root) / f
            if not f.endswith(_ARCHIVE_SUFFIXES):
                continue
            try:
                smart_extract(f_path, root, delete_after=True)
            except _DOWNLOAD_ERRORS as ex:
                logger.error("Failed to extract %s: %s", f_path, ex)
                print(f"  [ERR] Failed to extract {f}: {ex}")


def _download(repo: _HubRepo, target_file: str | None, output_dir: str) -> None:
    """Pull the selected files (or the whole snapshot), then extract leftovers."""
    targets = _select_targets(repo, target_file)
    if not targets:
        print("[HF] No specific targets found. Performing full snapshot download...")
        snapshot_download(
            repo_id=repo.repo_id,
            repo_type=repo.repo_type,
            local_dir=output_dir,
            token=repo.token,
        )
    else:
        print(f"[HF] Processing {len(targets)} files.")
        for t_file in targets:
            _pull_file(repo, t_file, output_dir)
    _sweep_archives(output_dir)


def main():
    """Download a Hugging Face repository (CLI entry point)."""
    parser = argparse.ArgumentParser(description="LemGendary HF Downloader")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Repo ID")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--repo_type", type=str, default="model", help="Repository type: model, dataset, space"
    )
    parser.add_argument(
        "--token_file", type=str, default=".huggingface_token", help="Path to token file"
    )
    args = parser.parse_args()

    # Auth
    token = _read_token(args.token_file)

    # Download (Surgical Mode v2.0)
    print(f"[HF] Starting {args.repo_type} download for {args.repo_id}...")

    # Sub-Repo Targeting (Syntax: repo_id:filename)
    target_file = None
    repo_id = args.repo_id
    if ":" in repo_id:
        repo_id, target_file = repo_id.split(":", 1)
        print(f"[SURGICAL] Targeting specific file: {target_file}")

    try:
        _download(_HubRepo(repo_id, args.repo_type, token), target_file, args.output_dir)
        print(f"[SUCCESS] {args.repo_id} processed to {args.output_dir}")
    except (*_DOWNLOAD_ERRORS, ImportError) as e:
        logger.error("Failed to download %s: %s", args.repo_id, e)
        print(f"[ERROR] Failed to download {args.repo_id}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
