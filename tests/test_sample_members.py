"""Converters keep masks/labels, reject corrupt images and write groups contiguously.

Covers audit items 2.1.1 (masks/labels lost, unsafe purge), 2.1.3 (zero-byte
images packed into shards) and 2.1.4 (paired members written thousands of
entries apart) for the loose-directory and zip-stream converters.
"""

import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from tools.modernize.loose_convert import convert_loose_directory_to_webdataset
from tools.zip_stream_common import ZipStreamOptions
from tools.zip_stream_webdataset import stream_zip_to_webdataset
from utils.image_payload import (
    CorruptImageError,
    encode_mask_payload,
    validate_image_bytes,
)


def _png(colour: tuple[int, int, int] = (200, 10, 10)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(buf, format="PNG")
    return buf.getvalue()


def _palette_mask() -> bytes:
    mask = Image.new("P", (8, 8), 0)
    # A palette of 256 distinct colours stops Pillow re-indexing it on save.
    mask.putpalette([v for i in range(256) for v in (i, 255 - i, (i * 7) % 256)])
    mask.putpixel((1, 1), 3)
    mask.putpixel((2, 2), 17)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return buf.getvalue()


def _members(shard: Path) -> list[str]:
    with tarfile.open(shard, "r") as tf:
        return [m.name for m in tf.getmembers()]


def _key(name: str) -> str:
    return name.split(".", 1)[0]


def _assert_contiguous(test: unittest.TestCase, names: list[str]) -> None:
    """Every key's members must form one unbroken run."""
    seen: set[str] = set()
    previous = ""
    for name in names:
        key = _key(name)
        if key != previous:
            test.assertNotIn(key, seen, f"members of {key} are not contiguous: {names}")
            seen.add(key)
        previous = key


class TestImagePayload(unittest.TestCase):
    """Image validation and lossless mask encoding."""

    def test_zero_byte_and_garbage_are_rejected(self) -> None:
        """Empty or undecodable bytes raise CorruptImageError."""
        with self.assertRaises(CorruptImageError):
            validate_image_bytes(b"", "empty.png")
        with self.assertRaises(CorruptImageError):
            validate_image_bytes(b"not an image", "junk.png")
        validate_image_bytes(_png(), "ok.png")

    def test_palette_mask_keeps_class_indices(self) -> None:
        """A palette mask round-trips its class ids through lossless WebP."""
        payload, ext = encode_mask_payload(_palette_mask(), "m")
        self.assertEqual(ext, ".webp")
        with Image.open(io.BytesIO(payload)) as img:
            grey = img.convert("L")
            self.assertEqual(grey.getpixel((1, 1)), 3)
            self.assertEqual(grey.getpixel((2, 2)), 17)
            self.assertEqual(grey.getpixel((0, 0)), 0)


class TestLooseConvert(unittest.TestCase):
    """Loose images/targets/masks/labels to WebDataset."""

    def _layout(self, root: Path, orphan_mask: bool = False) -> Path:
        src = root / "src"
        for sub in ("images", "targets", "masks", "labels"):
            (src / sub / "train").mkdir(parents=True)
        for stem in ("a", "b", "c"):
            (src / "images" / "train" / f"{stem}.png").write_bytes(_png())
            (src / "targets" / "train" / f"{stem}.png").write_bytes(_png((0, 0, 255)))
            (src / "masks" / "train" / f"{stem}.png").write_bytes(_palette_mask())
            (src / "labels" / "train" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (src / "images" / "train" / "b.png").write_bytes(b"")  # corrupt zero-byte image
        if orphan_mask:
            (src / "masks" / "train" / "zz.png").write_bytes(_palette_mask())
        return src

    def test_masks_labels_packed_contiguously_and_corrupt_rejected(self) -> None:
        """Masks and labels become members; the zero-byte sample is skipped; purge runs."""
        with tempfile.TemporaryDirectory() as tmp:
            src = self._layout(Path(tmp))
            out = Path(tmp) / "out"
            convert_loose_directory_to_webdataset(src, out, purge_loose=True, max_workers=2)
            names = _members(out / "shards" / "train" / "shard-00000.data")
            self.assertEqual(
                names,
                [
                    "a.webp", "a.target.webp", "a.mask.webp", "a.txt",
                    "c.webp", "c.target.webp", "c.mask.webp", "c.txt",
                ],
            )
            _assert_contiguous(self, names)
            for sub in ("images", "targets", "masks", "labels"):
                self.assertFalse((src / sub).exists(), sub)

    def test_orphan_mask_blocks_purge(self) -> None:
        """A mask with no image is not in the shards, so nothing is purged."""
        with tempfile.TemporaryDirectory() as tmp:
            src = self._layout(Path(tmp), orphan_mask=True)
            convert_loose_directory_to_webdataset(
                src, Path(tmp) / "out", purge_loose=True, max_workers=2
            )
            self.assertTrue((src / "masks" / "train" / "zz.png").exists())
            self.assertTrue((src / "images").exists())


class TestZipStream(unittest.TestCase):
    """Zip archive to WebDataset in one contiguous pass."""

    def test_zip_groups_are_contiguous_with_masks_and_labels(self) -> None:
        """Images, targets, masks and labels of a key sit back to back; bad image dropped."""
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "src.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for stem in ("k1", "k2", "k3"):
                    zf.writestr(f"images/train/{stem}.png", _png())
                    zf.writestr(f"targets/train/{stem}.png", _png((0, 0, 255)))
                    zf.writestr(f"masks/train/{stem}.png", _palette_mask())
                    zf.writestr(f"labels/train/{stem}.json", '{"class_id": 2}')
                zf.writestr("images/train/k4.png", b"")
            out = Path(tmp) / "out"
            ok = stream_zip_to_webdataset(
                zip_path, out, ZipStreamOptions(shard_size_samples=10, max_workers=2)
            )
            self.assertTrue(ok)
            names = _members(out / "shards" / "train" / "shard-00000.data")
            self.assertEqual(len(names), 12)
            self.assertNotIn("k4.webp", names)
            self.assertIn("k2.mask.webp", names)
            self.assertIn("k2.json", names)
            _assert_contiguous(self, names)


if __name__ == "__main__":
    unittest.main()
