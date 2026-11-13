"""Large-chunk reads for kagglehub's GCS upload (core/gcs_upload_chunks.py).

kagglehub's upload call is replaced by a fake; nothing reaches the network.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import kagglehub.gcs_upload as gcs_mod
from tqdm.utils import CallbackIOWrapper

from core import gcs_upload_chunks
from core.gcs_upload_chunks import LargeChunkReader, install_large_chunk_upload
from core.sync_upload import perform_dataset_upload

_CHUNK = 1024


def _drain(reader: LargeChunkReader, size: int) -> list[bytes]:
    blocks = []
    while block := reader.read(size):
        blocks.append(block)
    return blocks


class LargeChunkReaderTests(unittest.TestCase):
    """Reads return big chunks, report every byte, and honour the seek offset."""

    def setUp(self) -> None:
        patcher = patch.object(gcs_upload_chunks, "UPLOAD_CHUNK_BYTES", _CHUNK)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.payload = bytes(range(256)) * 10  # 2560 bytes

    def test_small_requests_get_a_full_chunk(self) -> None:
        """An 8 KB-style small read returns UPLOAD_CHUNK_BYTES, the last one the rest."""
        progress: list[int] = []
        blocks = _drain(LargeChunkReader(progress.append, io.BytesIO(self.payload), "read"), 16)
        self.assertEqual([len(b) for b in blocks], [_CHUNK, _CHUNK, 512])
        self.assertEqual(b"".join(blocks), self.payload)

    def test_callback_totals_match_the_bytes_read(self) -> None:
        """The progress callback sees every block, summing to the file size."""
        progress: list[int] = []
        _drain(LargeChunkReader(progress.append, io.BytesIO(self.payload), "read"), 16)
        self.assertEqual(sum(progress), len(self.payload))

    def test_eof_returns_empty_bytes(self) -> None:
        """At end of file read gives b'' and reports zero bytes."""
        progress: list[int] = []
        reader = LargeChunkReader(progress.append, io.BytesIO(b""), "read")
        self.assertEqual(reader.read(16), b"")
        self.assertEqual(progress, [0])

    def test_reads_start_at_the_seeked_offset(self) -> None:
        """A resumed upload seeks first; the reader continues from there."""
        stream = io.BytesIO(self.payload)
        stream.seek(1500)
        progress: list[int] = []
        blocks = _drain(LargeChunkReader(progress.append, stream, "read"), 16)
        self.assertEqual(b"".join(blocks), self.payload[1500:])
        self.assertEqual(sum(progress), len(self.payload) - 1500)

    def test_larger_requests_and_attributes_pass_through(self) -> None:
        """A request above the chunk is honoured; tell and other attributes reach the file."""
        stream = io.BytesIO(self.payload)
        reader = LargeChunkReader([].append, stream, "read")
        self.assertEqual(len(reader.read(2000)), 2000)
        self.assertEqual(reader.tell(), 2000)

    def test_only_read_is_wrapped(self) -> None:
        """Wrapping write is refused instead of silently skipping the callback."""
        with self.assertRaises(ValueError):
            LargeChunkReader([].append, io.BytesIO(), "write")


class InstallOnPushPathTests(unittest.TestCase):
    """sync push installs the reader into kagglehub.gcs_upload before uploading."""

    def test_install_replaces_callback_wrapper(self) -> None:
        """install_large_chunk_upload swaps the module variable."""
        with patch.object(gcs_mod, "CallbackIOWrapper", CallbackIOWrapper):
            install_large_chunk_upload()
            self.assertIs(vars(gcs_mod)["CallbackIOWrapper"], LargeChunkReader)

    def test_push_uploads_with_the_large_chunk_reader(self) -> None:
        """When kagglehub.dataset_upload runs, gcs_upload already uses LargeChunkReader."""
        seen: dict[str, Any] = {}

        def _fake_upload(*_args: Any, **_kwargs: Any) -> None:
            seen["wrapper"] = vars(gcs_mod)["CallbackIOWrapper"]

        with tempfile.TemporaryDirectory() as tmp:
            manifold = Path(tmp) / "LemGendizedDemo"
            manifold.mkdir()
            (manifold / "index.json").write_text(json.dumps({"train": []}), encoding="utf-8")
            with patch.object(gcs_mod, "CallbackIOWrapper", CallbackIOWrapper), \
                    patch.object(gcs_mod, "tqdm", gcs_mod.tqdm), \
                    patch("core.sync_upload.ensure_kaggle_dataset_exists", return_value=True), \
                    patch("core.sync_upload.get_dataset_version_info",
                          return_value={"latest_version": 1}), \
                    patch("core.sync_upload.kagglehub.dataset_upload", side_effect=_fake_upload), \
                    patch("core.sync_upload.push_kaggle_dataset_metadata"):
                self.assertTrue(perform_dataset_upload(manifold, "owner/demo", no_wait=True))
        self.assertIs(seen["wrapper"], LargeChunkReader)


if __name__ == "__main__":
    unittest.main()
