"""How one file of a Kaggle dataset reaches the disk in the file-by-file download.

A route fetches the bytes of one listed file into a ``.part`` path; the
caller (:mod:`core.kaggle_file_download`) checks the size, unpacks a zipped
file and moves it into place. A route returns False when Kaggle offers it no
download for the file (HTTP 404); any other failure raises.

- :class:`SignedUrlRoute`: Kaggle's per-file download URL
  (``/datasets/download/<owner>/<slug>/<file>``) redirects to signed
  storage; an interrupted ``.part`` continues with an HTTP range.
- :class:`ApiRoute`: the Kaggle API's ``dataset_download_file`` (the
  ``DownloadDataset`` RPC). It serves files whose per-file URL answers 404
  (seen for large datasets). The API writes into a staging folder beside the
  ``.part`` and resumes an interrupted file itself.
- :class:`RouteChain`: tries its routes in order. A route that had no
  download for one file is skipped for every later file of the run: Kaggle
  answers the same way for all files of a dataset, and each extra request
  counts against the rate limit.

HTTP 429 and other failures propagate to the caller's retry loop.
"""

from __future__ import annotations

import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import requests

from core.kaggle_http import KaggleHttpError, signed_url

__all__ = ["ApiRoute", "FileRoute", "RemoteFile", "RouteChain", "SignedUrlRoute"]

_TIMEOUT_S = 120
_CHUNK = 1 << 20
_PARTIAL = 206
_STAGING_SUFFIX = ".api"
_ZIP_SUFFIX = ".zip"


@dataclass(frozen=True)
class RemoteFile:
    """One file of a Kaggle dataset as the listing reports it."""

    name: str
    size: int


class FileRoute(Protocol):
    """Fetches one listed file into ``part``."""

    @property
    def label(self) -> str:
        """Short name of the route for progress messages."""
        raise NotImplementedError

    def fetch(self, item: RemoteFile, part: Path) -> bool:
        """Write ``item``'s bytes (maybe a one-member zip) to ``part``; False on HTTP 404."""
        raise NotImplementedError


@dataclass(frozen=True)
class SignedUrlRoute:
    """The per-file download URL of the Kaggle API, resumed with HTTP ranges."""

    handle: str
    session: requests.Session
    http: requests.Session
    label: str = "signed URL"

    def fetch(self, item: RemoteFile, part: Path) -> bool:
        """Resume or start ``part`` from the file's signed storage URL."""
        url = signed_url(self.session, self.handle, item.name)
        if url is None:
            return False
        have = part.stat().st_size if part.is_file() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        with self.http.get(url, headers=headers, stream=True, timeout=_TIMEOUT_S) as response:
            response.raise_for_status()
            append = bool(have) and response.status_code == _PARTIAL
            with open(part, "ab" if append else "wb") as out:
                out.writelines(response.iter_content(_CHUNK))
        return True


def _staged_file(staging: Path, name: str) -> Path | None:
    """Where the API put ``name`` under ``staging``: the file, or the zip it was wrapped in."""
    rel = PurePosixPath(name)
    target = staging.joinpath(*rel.parts)
    for candidate in (target, target.with_name(target.name + _ZIP_SUFFIX)):
        if candidate.is_file():
            return candidate
    return None


@dataclass(frozen=True)
class ApiRoute:
    """``api.dataset_download_file`` into a staging folder, then moved to ``part``."""

    handle: str
    api: Any
    label: str = "Kaggle API"

    def fetch(self, item: RemoteFile, part: Path) -> bool:
        """Download ``item`` through the API; the staging folder is removed afterwards."""
        staging = part.with_name(part.name + _STAGING_SUFFIX)
        self.api.dataset_download_file(self.handle, item.name, path=str(staging), quiet=True)
        staged = _staged_file(staging, item.name)
        if staged is None:
            raise KaggleHttpError(f"{item.name}: the Kaggle API wrote no file under {staging}")
        os.replace(staged, part)
        shutil.rmtree(staging)
        return True


@dataclass
class RouteChain:
    """Routes tried in order; one without a download is skipped from then on."""

    routes: tuple[FileRoute, ...]
    _skipped: set[int] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def label(self) -> str:
        """The labels of the routes still in use."""
        return " then ".join(r.label for i, r in enumerate(self.routes) if i not in self._skipped)

    def fetch(self, item: RemoteFile, part: Path) -> bool:
        """The first route that has a download for ``item``; False when none has."""
        for index, route in enumerate(self.routes):
            with self._lock:
                if index in self._skipped:
                    continue
            if route.fetch(item, part):
                return True
            with self._lock:
                if index not in self._skipped and index < len(self.routes) - 1:
                    self._skipped.add(index)
                    print(f"[GET] {route.label} has no download for {item.name};"
                          f" using {self.routes[index + 1].label} for the remaining files.")
        return False
