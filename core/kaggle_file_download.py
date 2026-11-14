"""Download a Kaggle dataset file by file, for datasets without a whole-dataset archive.

Kaggle builds no archive for a very large dataset (the download answers 404
"No gcs url found"). This module lists every file (all pages of
``dataset_list_files``), downloads each one and checks its size against the
listing:

- each file comes through its own signed URL, else, when Kaggle answers 404
  for the per-file URL, through the Kaggle API's ``dataset_download_file``
  (:mod:`core.kaggle_file_routes`; after the first 404 the API serves every
  remaining file);

- a file already on disk with the listed size is skipped, and an interrupted
  download continues (an HTTP range on the ``.part``, or the API's own
  resume in its staging folder);
- Kaggle may serve a single file zipped: a one-member zip whose member has
  the listed size is unpacked in place of the file;
- ``prefixes`` (``--only``) limit the download to some folders or files,
  relative to the dataset's wrapper folder (:mod:`utils.member_paths`);
- an HTTP 429 (rate limit) on a listing page or a file waits (Retry-After,
  else a growing pause up to :data:`RATE_LIMIT_MAX_WAIT_S`) and tries again,
  up to :data:`RATE_LIMIT_RETRIES` times.
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

from core.kaggle_errors import KAGGLE_REMOTE_ERRORS
from core.kaggle_file_routes import (
    ApiRoute,
    FileRoute,
    RemoteFile,
    RouteChain,
    SignedUrlRoute,
)
from core.kaggle_http import (
    TOO_MANY_REQUESTS,
    KaggleHttpError,
    KaggleRateLimitError,
)
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
    "rate_limit_wait",
]

logger = logging.getLogger(__name__)

_PAGE_SIZE = 200
_PART_SUFFIX = ".part"
_RETRIES = 5
_CHUNK = 1 << 20
_WORKERS = 4
RATE_LIMIT_RETRIES = 12
_RATE_LIMIT_BASE_S = 30.0
RATE_LIMIT_MAX_WAIT_S = 600.0


def rate_limit_wait(exc: BaseException, attempt: int) -> float | None:
    """Seconds to wait before retry ``attempt`` when ``exc`` is an HTTP 429; else None."""
    if isinstance(exc, KaggleRateLimitError):
        retry_after = exc.retry_after
    elif getattr(getattr(exc, "response", None), "status_code", None) == TOO_MANY_REQUESTS:
        retry_after = None
    else:
        return None
    backoff = _RATE_LIMIT_BASE_S * (2 ** min(attempt, 5))
    return min(RATE_LIMIT_MAX_WAIT_S, retry_after if retry_after else backoff)


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


def _list_page(api: Any, handle: str, token: str | None,
               sleep: Callable[[float], None]) -> Any:
    """One listing page; an HTTP 429 waits and asks again (the last one propagates)."""
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return api.dataset_list_files(handle, page_token=token, page_size=_PAGE_SIZE)
        except KAGGLE_REMOTE_ERRORS as exc:
            wait = rate_limit_wait(exc, attempt)
            if wait is None or attempt == RATE_LIMIT_RETRIES - 1:
                raise
            print(f"[GET] Kaggle rate limit while listing {handle}; waiting {wait:.0f} s.")
            sleep(wait)
    raise KaggleHttpError(f"listing {handle} never got past the rate limit")


def list_dataset_files(api: Any, handle: str,
                       sleep: Callable[[float], None] = time.sleep) -> list[RemoteFile]:
    """Every file of the dataset, read page by page until ``next_page_token`` is empty.

    Raises :class:`KaggleHttpError` when Kaggle repeats a page token (the
    listing would be incomplete).
    """
    files: list[RemoteFile] = []
    token: str | None = None
    seen: set[str] = set()
    while True:
        page = _list_page(api, handle, token, sleep)
        files.extend(RemoteFile(str(f.name), int(f.total_bytes))
                     for f in (getattr(page, "files", None) or []) if f is not None)
        token = getattr(page, "next_page_token", None) or None
        if token is None:
            return files
        if token in seen:
            raise KaggleHttpError(f"Kaggle repeated a page token while listing {handle}")
        seen.add(token)


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


def _download_once(route: FileRoute, item: RemoteFile, dst: Path) -> None:
    """One attempt: fetch into ``.part``, then verify the size (or unzip a zipped file)."""
    part = dst.with_name(dst.name + _PART_SUFFIX)
    have = part.stat().st_size if part.is_file() else 0
    if have == item.size:
        os.replace(part, dst)
        return
    if not route.fetch(item, part):
        raise KaggleHttpError(f"{item.name}: Kaggle has no download for this file")
    got = part.stat().st_size
    if got == item.size:
        os.replace(part, dst)
        return
    if _unzip_single(part, dst, item.size):
        return
    if got > item.size:
        part.unlink()
    raise KaggleHttpError(f"{item.name}: {got} bytes on disk, the listing says {item.size}")


def _download_file(route: FileRoute, item: RemoteFile, dst: Path,
                   sleep: Callable[[float], None]) -> None:
    """Download one file; failures retry ``_RETRIES`` times, rate limits more often."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    failures = limited = 0
    while True:
        try:
            _download_once(route, item, dst)
            return
        except (*KAGGLE_REMOTE_ERRORS, zipfile.BadZipFile) as exc:
            wait = rate_limit_wait(exc, limited)
            if wait is not None:
                limited += 1
                if limited >= RATE_LIMIT_RETRIES:
                    raise KaggleHttpError(str(exc)) from exc
            else:
                failures += 1
                if failures >= _RETRIES:
                    raise KaggleHttpError(str(exc)) from exc
                wait = float(5 * failures)
            logger.warning("Retrying %s in %.0f s: %s", item.name, wait, exc)
            sleep(wait)


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
    """The Kaggle API (listing, per-file API route), the session and the storage client."""

    api: Any
    session: requests.Session
    http: requests.Session | None = None
    sleep: Callable[[float], None] = time.sleep
    workers: int = _WORKERS


def download_dataset_files(handle: str, target_dir: Path, clients: FileDownloadClients,
                           prefixes: tuple[str, ...] = ()) -> FileDownloadResult:
    """Download the dataset's files (or those under ``prefixes``) into ``target_dir``."""
    result = FileDownloadResult()
    route = RouteChain((SignedUrlRoute(handle, clients.session, clients.http or requests.Session()),
                        ApiRoute(handle, clients.api)))
    pending: list[tuple[RemoteFile, Path]] = []
    listing = list_dataset_files(clients.api, handle, clients.sleep)
    for item, dst in _selected(listing, target_dir, prefixes):
        if dst.is_file() and dst.stat().st_size == item.size:
            result.skipped += 1
        else:
            pending.append((item, dst))
    print(f"[GET] File-by-file download of {handle}: {len(pending)} files to fetch,"
          f" {result.skipped} already complete.")

    def _one(job: tuple[RemoteFile, Path]) -> tuple[str, str | None]:
        item, dst = job
        try:
            _download_file(route, item, dst, clients.sleep)
        except KaggleHttpError as exc:
            return item.name, str(exc)
        return item.name, None

    total_files = len(pending) + result.skipped
    with ThreadPoolExecutor(max_workers=max(1, clients.workers)) as pool:
        for name, error in pool.map(_one, sorted(pending, key=lambda j: -j[0].size)):
            if error is None:
                result.downloaded += 1
            else:
                result.failed[name] = error
            done = result.downloaded + result.skipped
            pct = (done / total_files * 100.0) if total_files else 100.0
            print(f"DOWNLOADING: {handle} [{done}/{total_files} files] ({pct:.1f}%)", flush=True)
    return result
