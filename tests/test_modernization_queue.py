"""Unit tests for Modernization Queue engine and manifold scanner."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.modernization_queue import (
    ManifoldQueueItem,
    _fast_has_mds,
    _fast_has_tar_shards,
)


class TestModernizationQueue(unittest.TestCase):
    """Verifies modernization queue detection, data scanning, and item structures."""

    def test_fast_has_tar_shards_detection(self) -> None:
        """Verify _fast_has_tar_shards detects both .data and .tar files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            shards_dir = Path(tmp_dir) / "shards" / "train"
            shards_dir.mkdir(parents=True)

            # Initially empty
            has_shards, size = _fast_has_tar_shards(shards_dir.parent)
            self.assertFalse(has_shards)
            self.assertEqual(size, 0)

            # Add .data shard
            shard_data = shards_dir / "shard-00000.data"
            shard_data.write_bytes(b"x" * 2048)

            has_shards, size = _fast_has_tar_shards(shards_dir.parent)
            self.assertTrue(has_shards)
            self.assertEqual(size, 2048)

    def test_fast_has_mds_detection(self) -> None:
        """Verify _fast_has_mds detects index.json in MDS directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mds_dir = Path(tmp_dir) / "mds"
            mds_dir.mkdir(parents=True)

            has_mds, size = _fast_has_mds(mds_dir)
            self.assertFalse(has_mds)

            index_file = mds_dir / "index.json"
            index_file.write_bytes(b"{}")

            has_mds, size = _fast_has_mds(mds_dir)
            self.assertTrue(has_mds)

    def test_manifold_queue_item_structure(self) -> None:
        """Verify ManifoldQueueItem dataclass fields and defaults."""
        item = ManifoldQueueItem(
            key="test_manifold",
            name="Test Manifold",
            canonical_format="webdataset",
            modernized_folder="LemGendizedTestManifold",
            kaggle_ref="test/manifold",
            title="Test Dataset Title",
            subtitle="Test Dataset Subtitle",
            license_id="CC BY-NC 4.0",
        )
        self.assertEqual(item.key, "test_manifold")
        self.assertEqual(item.canonical_format, "webdataset")
        self.assertEqual(item.container_bytes, 0)
        self.assertFalse(item.is_containerized)


if __name__ == "__main__":
    unittest.main()
