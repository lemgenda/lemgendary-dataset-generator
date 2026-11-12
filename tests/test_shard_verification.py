"""Post-write shard verification and verify_archive reject empty images and scattered groups."""

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
import zipfile
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path

from tests.tar_fixtures import write_tar
from tools.modernize.mds_transcode import _verify_shards as _mds_verify_shards
from tools.zip_stream_webdataset import _verify_shards
from utils.archive import verify_archive
from utils.shard_audit import audit_shard, is_empty_defect, verify_shard_set


class ShardVerificationTests(unittest.TestCase):
    """Converters refuse to report success on unsound shards."""

    def setUp(self) -> None:
        """Fresh temporary directory per test."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.shards = self.root / "shards"

    def _quiet(self, func: Callable[..., bool], *args: object) -> tuple[bool, str]:
        with redirect_stdout(io.StringIO()) as out:
            result = func(*args)
        return result, out.getvalue()

    def test_empty_txt_is_not_a_defect(self) -> None:
        """Empty txt is not a defect."""
        self.assertFalse(is_empty_defect("a.txt", 0))
        self.assertTrue(is_empty_defect("a.jpg", 0))
        self.assertFalse(is_empty_defect("a.jpg", 10))
        shard = write_tar(self.shards / "train" / "s.data",
                           [("a.webp", b"x"), ("a.txt", b"")])
        self.assertFalse(audit_shard(shard).needs_rewrite)

    def test_sound_shards_pass(self) -> None:
        """Sound shards pass."""
        write_tar(self.shards / "train" / "s.data", [("a.webp", b"x"), ("a.json", b"{}")])
        self.assertTrue(self._quiet(verify_shard_set, self.shards, "T")[0])
        self.assertTrue(self._quiet(_verify_shards, self.shards, "data")[0])

    def test_zero_byte_member_fails_stream_verify(self) -> None:
        """Zero byte member fails stream verify."""
        write_tar(self.shards / "train" / "s.data", [("a.webp", b""), ("a.json", b"{}")])
        ok, out = self._quiet(_verify_shards, self.shards, "data")
        self.assertFalse(ok)
        self.assertIn("zero-byte", out)

    def test_unpaired_sample_in_paired_set_fails(self) -> None:
        """Once one sample has a target, every sample needs one."""
        write_tar(self.shards / "train" / "s.data",
                  [("a.webp", b"x"), ("a.target.webp", b"y"), ("b.webp", b"z")])
        ok, out = self._quiet(_verify_shards, self.shards, "data")
        self.assertFalse(ok)
        self.assertIn("1 samples without a target (first: b)", out)

    def test_scattered_group_fails_mds_verify(self) -> None:
        """Scattered group fails mds verify."""
        write_tar(self.shards / "s.data",
                   [("a.webp", b"x"), ("b.webp", b"y"), ("a.json", b"{}")])
        ok, out = self._quiet(_mds_verify_shards, self.shards, 1)
        self.assertFalse(ok)
        self.assertIn("not contiguous", out)


class VerifyArchiveTests(unittest.TestCase):
    """verify_archive fails on zero-byte image/shard members, not on other empty files."""

    def setUp(self) -> None:
        """Fresh temporary directory per test."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _zip(self, members: list[tuple[str, bytes]]) -> Path:
        path = self.root / "a.zip"
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in members:
                zf.writestr(name, data)
        return path

    def test_tar_with_empty_image_fails(self) -> None:
        """Tar with empty image fails."""
        bad = write_tar(self.root / "a.tar", [("x/img.jpg", b""), ("x/ok.jpg", b"1")])
        with redirect_stdout(io.StringIO()):
            self.assertFalse(verify_archive(bad))

    def test_tar_with_empty_log_passes(self) -> None:
        """Tar with empty log passes."""
        good = write_tar(self.root / "b.tar",
                          [("rejected_records.jsonl", b""), ("img.jpg", b"1")])
        self.assertTrue(verify_archive(good))

    def test_zip_with_empty_shard_fails(self) -> None:
        """Zip with empty shard fails."""
        with redirect_stdout(io.StringIO()):
            self.assertFalse(verify_archive(self._zip([("shards/s.data", b"")])))
        self.assertTrue(verify_archive(self._zip([("labels/a.txt", b""), ("a.png", b"1")])))


if __name__ == "__main__":
    unittest.main()
