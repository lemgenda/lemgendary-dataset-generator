"""
LemGendary Dataset Compiler — Container Formats and Transcoder Unit Tests.

Phase 1 of Ecosystem Comprehensive Testing Battery.
Tests Writer factory, ImageTranscoder, WebDatasetWriter, ParquetWriter,
DirectorySampleSource, and container protocol compliance.
"""

from __future__ import annotations

import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from PIL import Image

from formats.base import Sample, make_writer, parse_also_format
from formats.directory import DirectorySampleSource, DirectoryWriter
from formats.parquet import ParquetWriter
from formats.transcode import ImageTranscoder, KeepFormatError
from formats.webdataset import ShardWriter, WebDatasetWriter
from core.config_schema import ImageFormatPolicy


class TestWriterFactory(unittest.TestCase):
    """Test format factory and CLI argument parsing."""

    def test_make_writer_valid(self) -> None:
        self.assertIsInstance(make_writer("directory"), DirectoryWriter)
        self.assertIsInstance(make_writer("webdataset"), WebDatasetWriter)
        self.assertIsInstance(make_writer("parquet"), ParquetWriter)

        from formats.litdata import LitDataWriter
        from formats.mds import MDSWriter

        self.assertIsInstance(make_writer("litdata"), LitDataWriter)
        self.assertIsInstance(make_writer("mds"), MDSWriter)

    def test_make_writer_case_and_whitespace(self) -> None:
        self.assertIsInstance(make_writer("  WebDataset  "), WebDatasetWriter)
        self.assertIsInstance(make_writer("PARQUET"), ParquetWriter)

    def test_make_writer_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_writer("unsupported_format_xyz")

    def test_parse_also_format(self) -> None:
        self.assertEqual(
            parse_also_format("wds, litdata, parquet"),
            ["wds", "litdata", "parquet"],
        )
        self.assertEqual(parse_also_format(""), [])
        self.assertEqual(parse_also_format("   "), [])


class TestImageTranscoder(unittest.TestCase):
    """Test PIL Image transcoding, quality rules, and format mappings."""

    def setUp(self) -> None:
        self.rgb_image = Image.new("RGB", (32, 32), color=(255, 0, 0))
        self.rgba_image = Image.new("RGBA", (32, 32), color=(0, 255, 0, 128))

    def test_default_webp_encoding(self) -> None:
        transcoder = ImageTranscoder()
        self.assertTrue(transcoder.enabled)

        # Image encode
        img_bytes, fmt = transcoder.encode(self.rgb_image, kind="image")
        self.assertEqual(fmt, "webp")
        self.assertTrue(img_bytes.startswith(b"RIFF"))
        self.assertIn(b"WEBP", img_bytes[:16])

        # Target encode
        tgt_bytes, tgt_fmt = transcoder.encode(self.rgb_image, kind="target")
        self.assertEqual(tgt_fmt, "webp")
        self.assertTrue(tgt_bytes.startswith(b"RIFF"))

        # Mask encode (lossless)
        mask_bytes, mask_fmt = transcoder.encode(self.rgb_image, kind="mask")
        self.assertEqual(mask_fmt, "webp")
        self.assertTrue(mask_bytes.startswith(b"RIFF"))

    def test_jpeg_encoding_and_alpha_flatten(self) -> None:
        policy = ImageFormatPolicy(format="jpeg", quality=85)
        transcoder = ImageTranscoder(policy)

        # RGB to JPEG
        img_bytes, fmt = transcoder.encode(self.rgb_image, kind="image")
        self.assertEqual(fmt, "jpeg")
        self.assertTrue(img_bytes.startswith(b"\xff\xd8"))

        # RGBA to JPEG with automatic alpha flattening
        rgba_bytes, rgba_fmt = transcoder.encode(self.rgba_image, kind="image")
        self.assertEqual(rgba_fmt, "jpeg")
        self.assertTrue(rgba_bytes.startswith(b"\xff\xd8"))

    def test_png_encoding(self) -> None:
        policy = ImageFormatPolicy(format="png")
        transcoder = ImageTranscoder(policy)
        img_bytes, fmt = transcoder.encode(self.rgb_image, kind="image")
        self.assertEqual(fmt, "png")
        self.assertTrue(img_bytes.startswith(b"\x89PNG"))

    def test_keep_format_policy_raises(self) -> None:
        policy = ImageFormatPolicy(format="keep")
        transcoder = ImageTranscoder(policy)
        self.assertFalse(transcoder.enabled)
        with self.assertRaises(KeepFormatError):
            transcoder.encode(self.rgb_image, kind="image")

    def test_extension_for(self) -> None:
        transcoder = ImageTranscoder()
        self.assertEqual(transcoder.extension_for("webp"), ".webp")
        self.assertEqual(transcoder.extension_for("jpeg"), ".jpg")
        self.assertEqual(transcoder.extension_for("png"), ".png")
        self.assertEqual(transcoder.extension_for("unknown"), ".bin")


class TestWebDatasetWriter(unittest.TestCase):
    """Test WebDataset sharding and sample output."""

    def test_write_and_read_tar_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir)
            writer = WebDatasetWriter()
            policy = ImageFormatPolicy(wds_shard_size_bytes=10_000_000)

            writer.open(out_root, policy)

            sample1 = Sample(
                name="sample_001",
                task="restoration",
                split="train",
                image_bytes=b"fake_image_bytes_1",
                image_format="webp",
                target_bytes=b"fake_target_bytes_1",
                label="test label one",
                metadata={"score": 0.95, "source": "test"},
            )
            sample2 = Sample(
                name="sample_002",
                task="quality",
                split="train",
                image_bytes=b"fake_image_bytes_2",
                image_format="webp",
            )

            writer.write(sample1)
            writer.write(sample2)
            writer.close()

            shard_file = out_root / "shards" / "shard-00000.tar"
            self.assertTrue(shard_file.exists(), "WebDataset shard tar was not created")

            with tarfile.open(shard_file, "r") as tar:
                names = tar.getnames()
                self.assertIn("sample_001.webp", names)
                self.assertIn("sample_001.target.webp", names)
                self.assertIn("sample_001.txt", names)
                self.assertIn("sample_001.json", names)
                self.assertIn("sample_002.webp", names)

                # Verify payload
                txt_member = tar.extractfile("sample_001.txt")
                self.assertIsNotNone(txt_member)
                if txt_member is not None:
                    self.assertEqual(txt_member.read(), b"test label one")

    def test_legacy_shard_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "legacy_shards"
            writer = ShardWriter(output_dir=out_dir, prefix="test_shard", max_size=10_000_000)
            writer.write("item_001", b"fake_jpeg_data", "a beautiful sunrise")
            writer.close()

            tar_files = list(out_dir.glob("test_shard-*.tar"))
            self.assertEqual(len(tar_files), 1)
            with tarfile.open(tar_files[0], "r") as tar:
                names = tar.getnames()
                self.assertIn("item_001.jpg", names)
                self.assertIn("item_001.txt", names)


class TestParquetWriter(unittest.TestCase):
    """Test Parquet columnar serialization with Zstd compression."""

    def test_write_and_read_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir)
            writer = ParquetWriter()
            policy = ImageFormatPolicy()

            writer.open(out_root, policy)

            sample1 = Sample(
                name="row_001",
                task="restoration",
                split="train",
                image_bytes=b"raw_bytes_a",
                image_format="webp",
                target_bytes=b"target_bytes_a",
                label="sample label a",
                metadata={"id": 101},
            )
            sample2 = Sample(
                name="row_002",
                task="quality",
                split="val",
                image_bytes=b"raw_bytes_b",
                image_format="png",
                metadata={"id": 102},
            )

            writer.write(sample1)
            writer.write(sample2)
            writer.close()

            parquet_path = out_root / "manifold.parquet"
            self.assertTrue(parquet_path.exists(), "Parquet output file was not created")

            import pyarrow.parquet as pq

            table = pq.read_table(str(parquet_path))
            self.assertEqual(table.num_rows, 2)
            self.assertEqual(table.column("name").to_pylist(), ["row_001", "row_002"])
            self.assertEqual(table.column("task").to_pylist(), ["restoration", "quality"])
            self.assertEqual(table.column("split").to_pylist(), ["train", "val"])
            self.assertEqual(table.column("image").to_pylist(), [b"raw_bytes_a", b"raw_bytes_b"])


class TestDirectoryWriterAndSource(unittest.TestCase):
    """Test DirectoryWriter no-op lifecycle and DirectorySampleSource streaming."""

    def test_directory_writer_noop(self) -> None:
        writer = DirectoryWriter()
        sample = Sample("n", "t", "s", b"", "webp")
        writer.open(Path("."), None)
        writer.write(sample)
        writer.close()

    def test_directory_sample_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "images" / "train").mkdir(parents=True)
            (root / "targets" / "train").mkdir(parents=True)
            (root / "labels" / "train").mkdir(parents=True)

            img_file = root / "images" / "train" / "sample_alpha.webp"
            img_file.write_bytes(b"sample_alpha_img_data")

            tgt_file = root / "targets" / "train" / "sample_alpha.webp"
            tgt_file.write_bytes(b"sample_alpha_tgt_data")

            lbl_file = root / "labels" / "train" / "sample_alpha.txt"
            lbl_file.write_text("sample alpha label text", encoding="utf-8")

            index = [
                {
                    "name": "sample_alpha",
                    "task": "restoration",
                    "split": "train",
                    "source": "synthetic_benchmark",
                    "custom_metric": 42.0,
                }
            ]

            source = DirectorySampleSource(root=root, index=index)
            self.assertEqual(len(source), 1)

            samples = list(source)
            self.assertEqual(len(samples), 1)

            s = samples[0]
            self.assertEqual(s.name, "sample_alpha")
            self.assertEqual(s.task, "restoration")
            self.assertEqual(s.split, "train")
            self.assertEqual(s.image_bytes, b"sample_alpha_img_data")
            self.assertEqual(s.image_format, "webp")
            self.assertEqual(s.target_bytes, b"sample_alpha_tgt_data")
            self.assertIsNone(s.mask_bytes)
            self.assertEqual(s.label, "sample alpha label text")
            self.assertIsNotNone(s.metadata)
            if s.metadata is not None:
                self.assertEqual(s.metadata.get("source"), "synthetic_benchmark")
                self.assertEqual(s.metadata.get("custom_metric"), 42.0)


if __name__ == "__main__":
    unittest.main()
