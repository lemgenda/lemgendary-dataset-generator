"""Download a Kaggle dataset file by file, for datasets without a whole-dataset archive.

Kaggle builds no archive for a very large dataset (the download answers 404
"No gcs url found"). This module lists every file (all pages of
``dataset_list_files``), downloads each one through its own signed URL and
checks its size against the listing:

- a file already on disk with the listed size is skipped, and an interrupted
  ``.part`` download continues with an HTTP range (resumable);
- Kaggle may serve a single file zipped: a one-member zip whose member has
  the listed size is unpacked in place of the file;
- ``prefixes`` (``--only``) limit the download to some folders or files,
  relative to the dataset's wrapper folder (:mod:`utils.member_paths`).
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from core.kaggle_http import KaggleHttpError, signed_url
from utils.member_paths import (
    common_wrapper,
    local_path,
    matches_prefixes,
    relative_to_wrapper,
)

__all__ = [
    "FileDownloadClients",
    "FileDownloadResult",
    "RemoteFile",
    "download_dataset_files",
    "list_dataset_files",
]

logger = logging.getLogger(__name__)

_PAGE_SIZE = 200
_PART_SUFFIX = ".part"
_RETRIES = 5
_TIMEOUT_S = 120
_CHUNK = 1 << 20
_PARTIAL = 206
_WORKERS = 4


@dataclass(frozen=True)
class RemoteFile:
    """One file of a Kaggle dataset as the listing reports it."""

    name: str
    size: int


@dataclass
class FileDownloadResult:
    """Files downloaded, already complete, and failed (with the reason)."""

    downloaded: int = 0
    skipped: int = 0
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when every selected file is on disk with its listed size."""
        return not self.failed


def list_dataset_files(api: Any, handle: str) -> list[RemoteFile]:
    """Every file of the dataset, read page by page until ``next_page_token`` is empty.

    Raises :class:`KaggleHttpError` when Kaggle repeats a page token (the
    listing would be incomplete).
    """
    files: list[RemoteFile] = []
    token: str | None = None
    seen: set[str] = set()
    while True:
        page = api.dataset_list_files(handle, page_token=token, page_size=_PAGE_SIZE)
        files.extend(RemoteFile(str(f.name), int(f.total_bytes))
                     for f in (getattr(page, "files", None) or []) if f is not None)
        token = getattr(page, "next_page_token", None) or None
        if token is None:
            return files
        if token in seen:
            raise KaggleHttpError(f"Kaggle repeated a page token while listing {handle}")
        seen.add(token)


@dataclass(frozen=True)
class _Fetcher:
    """How one file is fetched: the Kaggle session, the storage client and the pause."""

    handle: str
    session: requests.Session
    http: requests.Session
    sleep: Callable[[float], None]


def _unzip_single(part: Path, dst: Path, size: int) -> bool:
    """Unpack a one-member zip of ``size`` bytes into ``dst``; False when it is not one."""
    if not zipfile.is_zipfile(part):
        return False
    with zipfile.ZipFile(part) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].file_size != size:
            return False
        tmp = dst.with_name(dst.name + _PART_SUFFIX + ".unzip")
        with archive.open(members[0]) as src, open(tmp, "wb") as out:
            shutil.copyfileobj(src, out, _CHUNK)
    os.replace(tmp, dst)
    part.unlink()
    return True


def _download_once(fetcher: _Fetcher, item: RemoteFile, dst: Path) -> None:
    """One attempt: resume ``.part``, then verify the size (or unzip a zipped file)."""
    part = dst.with_name(dst.name + _PART_SUFFIX)
    have = part.stat().st_size if part.is_file() else 0
    if have == item.size:
        os.replace(part, dst)
        return
    url = signed_url(fetcher.session, fetcher.handle, item.name)
    if url is None:
        raise KaggleHttpError(f"{item.name}: Kaggle has no download for this file")
    headers = {"Range": f"bytes={have}-"} if have else {}
    with fetcher.http.get(url, headers=headers, stream=True, timeout=_TIMEOUT_S) as response:
        response.raise_for_status()
        append = bool(have) and response.status_code == _PARTIAL
        with open(part, "ab" if append else "wb") as out:
            out.writelines(response.iter_content(_CHUNK))
    got = part.stat().st_size
    if got == item.size:
        os.replace(part, dst)
        return
    if _unzip_single(part, dst, item.size):
        return
    if got > item.size:
        part.unlink()
    raise KaggleHttpError(f"{item.name}: {got} bytes on disk, the listing says {item.size}")


def _download_file(fetcher: _Fetcher, item: RemoteFile, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(_RETRIES):
        try:
            _download_once(fetcher, item, dst)
            return
        except (OSError, requests.RequestException, zipfile.BadZipFile) as exc:
            if attempt == _RETRIES - 1:
                raise KaggleHttpError(str(exc)) from exc
            logger.warning("Retrying %s (attempt %d): %s", item.name, attempt + 1, exc)
            fetcher.sleep(float(5 * (attempt + 1)))


def _selected(files: list[RemoteFile], target_dir: Path,
              prefixes: tuple[str, ...]) -> list[tuple[RemoteFile, Path]]:
    wrapper = common_wrapper(f.name for f in files)
    chosen = []
    for item in files:
        rel = relative_to_wrapper(item.name, wrapper)
        if matches_prefixes(rel, prefixes):
            chosen.append((item, local_path(target_dir, rel)))
    return chosen


@dataclass(frozen=True)
class FileDownloadClients:
    """The Kaggle API (listing), the authenticated session and the storage client."""

    api: Any
    session: requests.Session
    http: requests.Session | None = None
    sleep: Callable[[float], None] = time.sleep
    workers: int = _WORKERS


def download_dataset_files(handle: str, target_dir: Path, clients: FileDownloadClients,
                           prefixes: tuple[str, ...] = ()) -> FileDownloadResult:
    """Download the dataset's files (or those under ``prefixes``) into ``target_dir``."""
    result = FileDownloadResult()
    fetcher = _Fetcher(handle, clients.session, clients.http or requests.Session(), clients.sleep)
    pending: list[tuple[RemoteFile, Path]] = []
    for item, dst in _selected(list_dataset_files(clients.api, handle), target_dir, prefixes):
        if dst.is_file() and dst.stat().st_size == item.size:
            result.skipped += 1
        else:
            pending.append((item, dst))
    print(f"[GET] File-by-file download of {handle}: {len(pending)} files to fetch,"
          f" {result.skipped} already complete.")

    def _one(job: tuple[RemoteFile, Path]) -> tuple[str, str | None]:
        item, dst = job
        try:
            _download_file(fetcher, item, dst)
        except KaggleHttpError as exc:
            return item.name, str(exc)
        return item.name, None

    with ThreadPoolExecutor(max_workers=max(1, clients.workers)) as pool:
        for name, error in pool.map(_one, sorted(pending, key=lambda j: -j[0].size)):
            if error is None:
                result.downloaded += 1
            else:
                result.failed[name] = error
    return result
