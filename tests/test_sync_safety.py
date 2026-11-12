"""Download safety: no container re-streaming, scoped Temp purge, no layout flattening."""

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from core.sync_download import _should_stream_to_webdataset
from core.sync_transfer import _is_manifold_zip, _purge_temp_zips
from tests.tar_fixtures import write_tar
from utils.archive_extract import _flatten_common_root


class _TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        """Fresh temporary directory per test."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)


@patch("core.sync_download._registry_marks_webdataset", return_value=False)
class StreamDecisionTests(_TempDirCase):
    """Only legacy loose-image zips are streamed into new shards (audit 2.2.2)."""

    def _zip(self, names: list[str]) -> Path:
        path = self.root / "LemGendizedDemo.zip"
        with zipfile.ZipFile(path, "w") as zf:
            for name in names:
                zf.writestr(name, b"x")
        return path

    def _decide(self, names: list[str]) -> bool:
        with redirect_stdout(io.StringIO()):
            return _should_stream_to_webdataset(self._zip(names), self.root / "LemGendizedDemo",
                                                "demo")

    def test_loose_images_stream(self, _registry: object) -> None:
        """Loose images stream, a root rejected_records.jsonl is only metadata."""
        self.assertTrue(self._decide(["images/train/a.jpg", "rejected_records.jsonl"]))

    def test_existing_containers_are_extracted(self, _registry: object) -> None:
        """Data shards, Parquet, MDS and data/ JSONL splits are never re-streamed."""
        for member in ("shards/train/shard-00000.data", "shards/shard-00000.tar",
                       "data/train.parquet", "mds/train/shard.00000.mds.zstd",
                       "data/ai_helper_train.jsonl"):
            with self.subTest(member=member):
                self.assertFalse(self._decide(["images/a.jpg", member]))

    def test_target_with_shards_is_not_streamed_into(self, _registry: object) -> None:
        """Streaming would mix new shards with the existing set."""
        write_tar(self.root / "LemGendizedDemo" / "shards" / "train" / "shard-00000.data",
                  [("a.webp", b"x")])
        self.assertFalse(self._decide(["images/train/a.jpg"]))


class TempPurgeTests(_TempDirCase):
    """The Temp sweep deletes manifold zips only (audit 2.2.3)."""

    def test_is_manifold_zip(self) -> None:
        """LemGendized* zips match; other programs' zips never do."""
        self.assertTrue(_is_manifold_zip(Path("LemGendizedA.zip"), None))
        self.assertTrue(_is_manifold_zip(Path("lemgendizeda.zip"), None))
        self.assertFalse(_is_manifold_zip(Path("holiday_photos.zip"), None))
        self.assertTrue(_is_manifold_zip(Path("LemGendizedA.zip"), "LemGendizedA"))
        self.assertFalse(_is_manifold_zip(Path("LemGendizedB.zip"), "LemGendizedA"))

    def test_purge_keeps_foreign_zips(self) -> None:
        """A forced sweep removes only the manifold zip."""
        for name in ("LemGendizedA.zip", "holiday_photos.zip"):
            (self.root / name).write_bytes(b"1234")
        count, size = _purge_temp_zips(self.root, None, 0.0, True)
        self.assertEqual((count, size), (1, 4))
        self.assertEqual([p.name for p in self.root.iterdir()], ["holiday_photos.zip"])


class FlattenTests(_TempDirCase):
    """Structural folders are never hoisted into the manifold root (audit 2.2.3)."""

    def test_lone_shards_folder_is_kept(self) -> None:
        """LemGendizedX/shards/train stays put."""
        dest = self.root / "LemGendizedX"
        write_tar(dest / "shards" / "train" / "shard-00000.data", [("a.webp", b"x")])
        (dest / "dataset_info.yaml").write_text("count: 1\n", encoding="utf-8")
        _flatten_common_root(dest)
        self.assertTrue((dest / "shards" / "train" / "shard-00000.data").is_file())
        self.assertFalse((dest / "train").exists())

    def test_real_wrapper_is_flattened(self) -> None:
        """LemGendizedX/LemGendizedX/shards becomes LemGendizedX/shards."""
        dest = self.root / "LemGendizedX"
        write_tar(dest / "LemGendizedX" / "shards" / "shard-00000.data", [("a.webp", b"x")])
        _flatten_common_root(dest)
        self.assertTrue((dest / "shards" / "shard-00000.data").is_file())
        self.assertFalse((dest / "LemGendizedX").exists())


if __name__ == "__main__":
    unittest.main()
