"""
Large-chunk reads for kagglehub's GCS blob upload.

kagglehub's ``_upload_blob`` streams the archive through
``tqdm.utils.CallbackIOWrapper``; urllib3 then reads it in 16 KB blocks and
one upload stalls at a few MB/s. ``LargeChunkReader`` is a drop-in for that
wrapper whose ``read`` returns ``UPLOAD_CHUNK_BYTES`` whatever size is asked,
still reports every block to the progress callback, and reads from the
current file position (kagglehub seeks to the resume offset before wrapping).
"""

import importlib
import logging
from collections.abc import Callable
from typing import IO, Any

logger = logging.getLogger(__name__)

UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024


class LargeChunkReader:
    """File proxy whose ``read`` returns large chunks and reports their size."""

    def __init__(self, callback: Callable[[int], Any], stream: IO[bytes], method: str = "read"):
        if method != "read":
            raise ValueError(f"LargeChunkReader only wraps read, not {method!r}")
        self._callback = callback
        self._stream = stream

    def read(self, size: int = -1) -> bytes:
        """Read ``UPLOAD_CHUNK_BYTES`` or more (all on a negative size)."""
        want = size if size < 0 else max(size, UPLOAD_CHUNK_BYTES)
        data = self._stream.read(want)
        self._callback(len(data))
        return data

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def install_large_chunk_upload() -> None:
    """Point ``kagglehub.gcs_upload.CallbackIOWrapper`` at ``LargeChunkReader``."""
    try:
        gcs_mod = importlib.import_module("kagglehub.gcs_upload")
    except ImportError as exc:
        logger.debug("Chunked upload patch notice: %s", exc)
        return
    if "CallbackIOWrapper" in vars(gcs_mod):
        vars(gcs_mod)["CallbackIOWrapper"] = LargeChunkReader
