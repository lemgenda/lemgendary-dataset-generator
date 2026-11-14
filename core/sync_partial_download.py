"""Kaggle downloads that do not go through the whole-dataset archive.

- :func:`download_file_by_file` is the fallback when Kaggle has no
  whole-dataset archive (a very large dataset): every file is downloaded on
  its own and checked against the listing (:mod:`core.kaggle_file_download`);
- :func:`download_selected` fetches only some folders or files (``sync pull
  --only labels/ --only classes.txt``): from the whole-dataset zip through
  HTTP byte ranges (:mod:`utils.zip_ranges`) when Kaggle has one, else file
  by file.

Both resume: what is already on disk and complete is skipped.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import requests

from core.kaggle_errors import KAGGLE_CLIENT_ERRORS
from core.kaggle_file_download import FileDownloadClients, download_dataset_files
from core.kaggle_http import (
    HttpRangeReader,
    KaggleHttpError,
    SignedUrl,
    kaggle_session,
    signed_url,
)
from core.lazy_imports import load_kaggle_api
from utils.member_paths import UnsafeMemberPath
from utils.zip_ranges import (
    ExtractStats,
    ZipRangeError,
    extract_members,
    read_central_directory,
    select_members,
)

__all__ = ["download_file_by_file", "download_selected", "is_missing_archive_error"]

_NOT_FOUND = 404
_NO_ARCHIVE_TEXT = "no gcs url"
_SHOWN = 5
_ERRORS = (KaggleHttpError, ZipRangeError, UnsafeMemberPath, requests.RequestException,
           zlib.error, *KAGGLE_CLIENT_ERRORS)


def is_missing_archive_error(exc: BaseException) -> bool:
    """True when a whole-dataset download failed because Kaggle has no archive for it."""
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == _NOT_FOUND:
        return True
    return _NO_ARCHIVE_TEXT in str(exc).lower()


def _clients() -> FileDownloadClients:
    api = load_kaggle_api()()
    api.authenticate()
    return FileDownloadClients(api=api, session=kaggle_session())


def _report_failures(failed: dict[str, str] | list[str]) -> None:
    names = list(failed)
    print(f"[ERROR] {len(names)} files failed: {', '.join(names[:_SHOWN])}")


def _files(handle: str, target_dir: Path, prefixes: tuple[str, ...]) -> bool:
    result = download_dataset_files(handle, target_dir, _clients(), prefixes)
    print(f"[GET] {result.downloaded} files downloaded, {result.skipped} already complete.")
    if result.failed:
        _report_failures(result.failed)
    return result.ok and (result.downloaded + result.skipped) > 0


def download_file_by_file(handle: str, target_dir: Path, prefixes: tuple[str, ...] = ()) -> bool:
    """Download every file (or those under ``prefixes``) into ``target_dir``; True when all are."""
    print(
        f"[GET] Kaggle whole-dataset archive is unavailable or awaiting compression "
        f"for {handle}; downloading file by file."
    )
    try:
        return _files(handle, target_dir, prefixes)
    except _ERRORS as exc:
        print(f"[ERROR] File-by-file download of {handle} failed: {exc}")
        return False


def _progress(stats: ExtractStats) -> None:
    print(f"[GET] {stats.written} members written, {stats.bytes_read / 1e6:.1f} MB read",
          flush=True)


def _ranges(handle: str, url: str, session: requests.Session, target_dir: Path,
            prefixes: tuple[str, ...]) -> bool:
    signed = SignedUrl(lambda: signed_url(session, handle), initial=url)
    reader = HttpRangeReader(signed)
    members = select_members(read_central_directory(reader), prefixes)
    if not members:
        print(f"[ERROR] Nothing in {handle} matches {', '.join(prefixes)}")
        return False
    print(f"[GET] {len(members)} members under {', '.join(prefixes)}; fetching byte ranges.")
    stats = extract_members(reader, members, target_dir, _progress)
    print(f"[GET] {stats.written} written, {stats.skipped} already complete,"
          f" {stats.bytes_read / 1e6:.1f} MB read.")
    if stats.failed:
        _report_failures(stats.failed)
    return not stats.failed


def download_selected(handle: str, target_dir: Path, prefixes: tuple[str, ...]) -> bool:
    """Fetch only the members under ``prefixes`` into ``target_dir``; True when all arrived."""
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        session = kaggle_session()
        url = signed_url(session, handle)
        if url is None:
            return download_file_by_file(handle, target_dir, prefixes)
        return _ranges(handle, url, session, target_dir, prefixes)
    except _ERRORS as exc:
        print(f"[ERROR] Selected download of {handle} failed: {exc}")
        return False
