"""Authenticated HTTP access to Kaggle dataset downloads for the sync core.

Kaggle answers ``GET /api/v1/datasets/download/<owner>/<slug>[/<file>]``
with a redirect to a short-lived signed storage URL; a dataset too large
for a whole-dataset archive answers 404 ("No gcs url found") instead.

- :func:`kaggle_session` authenticates with the repo's credential module
  (an access token as a Bearer header, a legacy key as basic auth);
- :func:`signed_url` resolves the redirect (None when there is no archive);
- :class:`SignedUrl` re-resolves the URL before it expires or when storage
  rejects it;
- :class:`HttpRangeReader` reads byte ranges of the signed URL with retries
  (a :class:`utils.zip_ranges.RangeReader`).

The signed storage URL is fetched without Kaggle credentials: they are only
sent to kaggle.com.
"""

from __future__ import annotations

import time
import urllib.parse
from collections.abc import Callable

import requests

from core.kaggle_credentials import (
    KaggleCredential,
    environment_credential,
    stored_credential,
)

__all__ = [
    "API_BASE",
    "HttpRangeReader",
    "KaggleHttpError",
    "KaggleRateLimitError",
    "SignedUrl",
    "dataset_download_url",
    "kaggle_session",
    "signed_url",
]

API_BASE = "https://www.kaggle.com/api/v1"
_REDIRECTS = frozenset({301, 302, 303, 307, 308})
_NOT_FOUND = 404
TOO_MANY_REQUESTS = 429
_PARTIAL = 206
_EXPIRED = frozenset({400, 401, 403})
# Signed URLs live about an hour; refresh well before that.
SIGNED_URL_TTL_S = 1800.0
_TIMEOUT_S = 120
_RETRIES = 8


class KaggleHttpError(OSError):
    """A Kaggle download request failed (no credential, bad status, short read)."""


class KaggleRateLimitError(KaggleHttpError):
    """Kaggle answered HTTP 429; ``retry_after`` is its Retry-After in seconds, if any."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after(response: requests.Response) -> float | None:
    value = response.headers.get("Retry-After", "")
    return float(value) if value.strip().isdigit() else None


def kaggle_session(credential: KaggleCredential | None = None) -> requests.Session:
    """A session that authenticates to kaggle.com with ``credential``.

    Without one, the vault's stored credential is used, then the
    environment's; raises :class:`KaggleHttpError` when neither exists.
    """
    chosen = credential or stored_credential() or environment_credential()
    if chosen is None:
        raise KaggleHttpError("no Kaggle credential in the secrets vault or the environment")
    session = requests.Session()
    if chosen.is_access_token:
        session.headers["Authorization"] = f"Bearer {chosen.secret}"
    else:
        session.auth = (chosen.username or "", chosen.secret)
    return session


def dataset_download_url(handle: str, file_name: str | None = None) -> str:
    """The Kaggle API download URL of a dataset (``owner/slug``) or of one of its files."""
    url = f"{API_BASE}/datasets/download/{handle}"
    return f"{url}/{urllib.parse.quote(file_name, safe='')}" if file_name else url


def signed_url(session: requests.Session, handle: str, file_name: str | None = None) -> str | None:
    """The signed storage URL Kaggle redirects to; None when Kaggle has no such archive."""
    response = session.get(dataset_download_url(handle, file_name), allow_redirects=False,
                           timeout=_TIMEOUT_S)
    location = response.headers.get("Location")
    if response.status_code in _REDIRECTS and location:
        return location
    if response.status_code == _NOT_FOUND:
        return None
    if response.status_code == TOO_MANY_REQUESTS:
        raise KaggleRateLimitError(f"Kaggle rate-limited the download of {handle}",
                                   _retry_after(response))
    raise KaggleHttpError(
        f"Kaggle download of {handle}{'/' + file_name if file_name else ''} answered"
        f" HTTP {response.status_code}"
    )


class SignedUrl:
    """A signed URL that is resolved again when it is old or rejected."""

    def __init__(self, resolve: Callable[[], str | None], initial: str = "",
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._resolve = resolve
        self._clock = clock
        self._url = initial
        self._at = clock()

    def get(self) -> str:
        """The current URL, refreshed when it is older than ``SIGNED_URL_TTL_S``."""
        if not self._url or self._clock() - self._at > SIGNED_URL_TTL_S:
            self.refresh()
        return self._url

    def refresh(self) -> str:
        """Resolve a fresh URL; raises :class:`KaggleHttpError` when there is none."""
        url = self._resolve()
        if not url:
            raise KaggleHttpError("Kaggle no longer offers a download URL for this archive")
        self._url, self._at = url, self._clock()
        return url


class HttpRangeReader:
    """Byte ranges of a signed URL, with retries and URL refresh."""

    def __init__(self, url: SignedUrl, http: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._url = url
        self._http = http or requests.Session()
        self._sleep = sleep
        self._size: int | None = None

    def _get(self, start: int, end: int, check_length: bool = True) -> requests.Response:
        """A 206 answer for the range (with all its bytes when ``check_length``), with retries."""
        last = ""
        for attempt in range(_RETRIES):
            try:
                response = self._http.get(self._url.get(),
                                          headers={"Range": f"bytes={start}-{end}"},
                                          timeout=_TIMEOUT_S)
            except requests.RequestException as exc:
                last = str(exc)
            else:
                if response.status_code == _PARTIAL and (
                        not check_length or len(response.content) == end - start + 1):
                    return response
                if response.status_code in _EXPIRED:
                    self._url.refresh()
                last = f"HTTP {response.status_code}, {len(response.content)} bytes"
            self._sleep(float(attempt + 1))
        raise KaggleHttpError(f"range {start}-{end} failed after {_RETRIES} attempts ({last})")

    def size(self) -> int:
        """Total size, from the ``Content-Range`` of a one-byte request."""
        if self._size is None:
            header = self._get(0, 0, check_length=False).headers.get("Content-Range", "")
            total = header.rpartition("/")[2]
            if not total.isdigit():
                raise KaggleHttpError(f"no total size in Content-Range {header!r}")
            self._size = int(total)
        return self._size

    def read(self, start: int, end: int) -> bytes:
        """Bytes ``start`` to ``end`` inclusive; raises when they never arrive complete."""
        return self._get(start, end).content
