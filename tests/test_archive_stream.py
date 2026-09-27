"""
LemGendary Dataset Compiler — Archive and Direct Streaming Unit Tests.

Phase 1 of Ecosystem Comprehensive Testing Battery.
Tests utils.archive (create, verify, extract, flatten) and
tools.stream_zip_to_container (streaming Zip-to-WebDataset with on-the-fly WebP).
"""

from __future__ import annotations

import io
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
import zipfile

from PIL import Image
import yaml

from tools.stream_zip_to_container import (
    _transcode_to_webp,
    stream_zip_to_webdataset,
)
from utils.archive import (
    _flatten_common_root,
    create_archive,
    smart_extract,
    verify_archive,
)


class TestArchiveUtils(unittest.TestCase):
    """Test smart archive creation, verification, and extraction."""

    def test_create_and_verify_zip_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_dir = Path(tmp_dir) / "source_folder"
            src_dir.mkdir()
            (src_dir / "file1.txt").write_text("Hello LemGendary 1", encoding="utf-8")
            (src_dir / "file2.txt").write_text("Hello LemGendary 2", encoding="utf-8")

            zip_out = Path(tmp_dir) / "test_archive.zip"
            success = create_archive(source_dir=src_dir, output_path=zip_out, archive_format="zip")
            self.assertTrue(success)
            self.assertTrue(zip_out.exists())
            self.assertTrue(verify_archive(zip_out))

    def test_create_and_verify_tar_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_dir = Path(tmp_dir) / "source_tar_folder"
            src_dir.mkdir()
            (src_dir / "data.bin").write_bytes(b"\x00\x01\x02\x03")

            tar_out = Path(tmp_dir) / "test_archive.tar.gz"
            success = create_archive(source_dir=src_dir, output_path=tar_out, archive_format="tar.gz")
            self.assertTrue(success)
            self.assertTrue(tar_out.exists())
            self.assertTrue(verify_archive(tar_out))

    def test_verify_archive_corrupted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file = Path(tmp_dir) / "corrupt.zip"
            bad_file.write_bytes(b"This is not a valid zip archive file contents.")
            self.assertFalse(verify_archive(bad_file))

    def test_smart_extract_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_file = Path(tmp_dir) / "payload.zip"
            with zipfile.ZipFile(zip_file, "w") as zf:
                zf.writestr("test_sub/file_a.txt", "Content A")
                zf.writestr("test_sub/file_b.txt", "Content B")

            extract_dest = Path(tmp_dir) / "extracted"
            success = smart_extract(zip_file, extract_dest, delete_after=False)
            self.assertTrue(success)
            self.assertTrue((extract_dest / "test_sub" / "file_a.txt").exists())
            self.assertEqual(
                (extract_dest / "test_sub" / "file_a.txt").read_text(encoding="utf-8"),
                "Content A",
            )
            self.assertTrue(zip_file.exists())

    def test_flatten_common_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_dir = Path(tmp_dir) / "LemGendizedTestDataset"
            dest_dir.mkdir()
            nested_wrapper = dest_dir / "LemGendizedTestDataset"
            nested_wrapper.mkdir()
            (nested_wrapper / "images").mkdir()
            (nested_wrapper / "images" / "a.png").write_bytes(b"dummy")
            (nested_wrapper / "dataset_info.yaml").write_text("name: test", encoding="utf-8")

            _flatten_common_root(dest_dir)

            self.assertTrue((dest_dir / "images" / "a.png").exists())
            self.assertTrue((dest_dir / "dataset_info.yaml").exists())
            self.assertFalse(nested_wrapper.exists())


class TestStreamZipToContainer(unittest.TestCase):
    """Test direct Zip-to-WebDataset streaming engine."""

    def setUp(self) -> None:
        # Create a small valid test PNG image
        img = Image.new("RGB", (16, 16), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.sample_png_bytes = buf.getvalue()

    def test_transcode_to_webp(self) -> None:
        raw_bytes, ext = _transcode_to_webp(self.sample_png_bytes, ".png", quality=90)
        self.assertEqual(ext, ".webp")
        self.assertTrue(raw_bytes.startswith(b"RIFF"))

        # Pre-existing WebP bytes should not be re-encoded
        webp_bytes, webp_ext = _transcode_to_webp(raw_bytes, ".webp")
        self.assertEqual(webp_ext, ".webp")
        self.assertEqual(webp_bytes, raw_bytes)

    def test_stream_paired_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            source_zip = work_dir / "legacy_paired.zip"
            target_manifold = work_dir / "CompiledManifold"
            legacy_dir = work_dir / "LegacyManifoldLarge"
            legacy_dir.mkdir()
            (legacy_dir / "stale.txt").write_text("stale data", encoding="utf-8")

            # Create synthetic zip archive with paired structure and metadata
            with zipfile.ZipFile(source_zip, "w") as zf:
                zf.writestr("images/train/sample_101.png", self.sample_png_bytes)
                zf.writestr("targets/train/sample_101.png", self.sample_png_bytes)
                zf.writestr("images/val/sample_102.png", self.sample_png_bytes)
                zf.writestr("targets/val/sample_102.png", self.sample_png_bytes)
                zf.writestr("README.md", "# Test Dataset Readme")
                zf.writestr("dataset_info.yaml", yaml.safe_dump({"name": "test_ds", "format": "legacy"}))

            success = stream_zip_to_webdataset(
                zip_path=source_zip,
                target_dir=target_manifold,
                shard_size_samples=10,
                delete_zip=True,
                legacy_dir=legacy_dir,
                transcode_webp=True,
                max_workers=2,
            )
            self.assertTrue(success, "stream_zip_to_webdataset failed")

            # 1. Root metadata extracted and updated
            readme_path = target_manifold / "README.md"
            self.assertTrue(readme_path.exists())
            ds_info_path = target_manifold / "dataset_info.yaml"
            self.assertTrue(ds_info_path.exists())
            info_data = yaml.safe_load(ds_info_path.read_text(encoding="utf-8"))
            self.assertEqual(info_data.get("format"), "webdataset")
            self.assertEqual(info_data.get("image_format"), "webp")

            # 2. Shards created
            train_shard = target_manifold / "shards" / "train" / "shard-00000.tar"
            val_shard = target_manifold / "shards" / "val" / "shard-00000.tar"
            self.assertTrue(train_shard.exists(), "Train shard was not created")
            self.assertTrue(val_shard.exists(), "Val shard was not created")

            # Verify train shard contains sample_101.webp and sample_101.target.webp
            with tarfile.open(train_shard, "r") as tf:
                names = tf.getnames()
                self.assertIn("sample_101.webp", names)
                self.assertIn("sample_101.target.webp", names)

            # 3. Cleanup: source zip and legacy dir deleted
            self.assertFalse(source_zip.exists(), "source_zip was not deleted")
            self.assertFalse(legacy_dir.exists(), "legacy_dir was not deleted")

    def test_stream_targets_only_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            source_zip = work_dir / "legacy_targets_only.zip"
            target_manifold = work_dir / "UpnManifold"

            with zipfile.ZipFile(source_zip, "w") as zf:
                zf.writestr("targets/train/hr_001.png", self.sample_png_bytes)
                zf.writestr("targets/train/hr_002.png", self.sample_png_bytes)
                zf.writestr("targets/val/hr_003.png", self.sample_png_bytes)

            success = stream_zip_to_webdataset(
                zip_path=source_zip,
                target_dir=target_manifold,
                shard_size_samples=10,
                delete_zip=False,
                legacy_dir=None,
                transcode_webp=True,
                max_workers=2,
            )
            self.assertTrue(success)

            train_shard = target_manifold / "shards" / "train" / "shard-00000.tar"
            val_shard = target_manifold / "shards" / "val" / "shard-00000.tar"
            self.assertTrue(train_shard.exists())
            self.assertTrue(val_shard.exists())

            with tarfile.open(train_shard, "r") as tf:
                names = tf.getnames()
                self.assertIn("hr_001.webp", names)
                self.assertIn("hr_002.webp", names)


if __name__ == "__main__":
    unittest.main()
