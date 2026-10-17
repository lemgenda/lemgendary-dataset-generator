"""Unit tests for WebDataset shard extension migration (.tar <-> .data)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.migrate_shards_to_data import migrate_shards


class TestMigration(unittest.TestCase):
    """Verifies atomic shard migration between .tar and .data extensions."""

    def test_migrate_shards_forward_and_reverse(self) -> None:
        """Verify forward migration to .data and reverse migration to .tar."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_root = Path(tmp_dir)
            manifold_dir = datasets_root / "LemGendizedTestDataset"
            shards_dir = manifold_dir / "shards" / "train"
            shards_dir.mkdir(parents=True)

            shard_file = shards_dir / "shard-00000.tar"
            shard_file.write_bytes(b"dummy_shard_bytes_123")

            # 1. Dry run: should not rename file
            code_dry = migrate_shards(datasets_root, dry_run=True, reverse=False)
            self.assertEqual(code_dry, 0)
            self.assertTrue(shard_file.exists())
            self.assertFalse((shards_dir / "shard-00000.data").exists())

            # 2. Live migration: .tar -> .data
            code_live = migrate_shards(datasets_root, dry_run=False, reverse=False)
            self.assertEqual(code_live, 0)
            self.assertFalse(shard_file.exists())
            migrated_file = shards_dir / "shard-00000.data"
            self.assertTrue(migrated_file.exists())
            self.assertEqual(migrated_file.read_bytes(), b"dummy_shard_bytes_123")

            # 3. Reverse migration: .data -> .tar
            code_rev = migrate_shards(datasets_root, dry_run=False, reverse=True)
            self.assertEqual(code_rev, 0)
            self.assertTrue(shard_file.exists())
            self.assertFalse(migrated_file.exists())


if __name__ == "__main__":
    unittest.main()
