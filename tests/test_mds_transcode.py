"""MDS-to-WebDataset transcoding keeps every metadata column (audit 2.1.2)."""

import io
import json
import struct
import tarfile
import tempfile
import unittest
from pathlib import Path

import zstandard as zstd

from tools.modernize.mds_transcode import (
    _mds_sample_to_tar_members,
    transcode_mds_to_webdataset,
)

_COLUMNS = ["image", "name", "label", "class_id", "nima_scores", "split"]
_ENCODINGS = ["bytes", "str", "str", "int", "json", "str"]
_WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 fake"


def _sample(name: str, class_id: int) -> bytes:
    payloads = [
        _WEBP,
        name.encode(),
        b"cat",
        struct.pack("<q", class_id),
        json.dumps({"mean": 5.5}).encode(),
        b"train",
    ]
    sizes = struct.pack(f"<{len(payloads)}I", *(len(p) for p in payloads))
    return sizes + b"".join(payloads)


def _mds_shard(samples: list[bytes]) -> bytes:
    header = 4 + (len(samples) + 1) * 4
    offsets = [header]
    for sample in samples:
        offsets.append(offsets[-1] + len(sample))
    raw = struct.pack("<I", len(samples)) + struct.pack(f"<{len(offsets)}I", *offsets)
    return zstd.ZstdCompressor().compress(raw + b"".join(samples))


class TestMdsTranscode(unittest.TestCase):
    """Columns beyond label/split/task survive the MDS transcode."""

    def test_member_mapping_copies_typed_columns(self) -> None:
        """class_id (int) and nima_scores (json) land decoded in the JSON sidecar."""
        fields = {
            "image": _WEBP,
            "name": b"k",
            "class_id": struct.pack("<q", 7),
            "nima_scores": b'{"mean": 4.0}',
            "category": b"dog",
        }
        encodings = {"class_id": "int", "nima_scores": "json", "category": "str"}
        members = dict(_mds_sample_to_tar_members(fields, "k", encodings))
        self.assertIn("k.webp", members)
        meta = json.loads(members["k.json"])
        self.assertEqual(meta, {"class_id": 7, "nima_scores": {"mean": 4.0}, "category": "dog"})

    def test_binary_column_is_logged_not_dumped(self) -> None:
        """A pickled column is skipped with a warning instead of written as text."""
        with self.assertLogs("tools.modernize.mds_transcode", level="WARNING"):
            fields = {"image": _WEBP, "blob": b"\x80\x04"}
            members = dict(_mds_sample_to_tar_members(fields, "k", {"blob": "pkl"}))
        self.assertNotIn("k.json", members)

    def test_full_transcode_keeps_class_ids(self) -> None:
        """A synthetic MDS split transcodes with class_id intact and count verified."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "mds" / "train"
            split_dir.mkdir(parents=True)
            (split_dir / "shard.00000.mds.zstd").write_bytes(
                _mds_shard([_sample("s0", 3), _sample("s1", 9)])
            )
            index = {
                "shards": [
                    {
                        "column_names": _COLUMNS,
                        "column_encodings": _ENCODINGS,
                        "samples": 2,
                        "zip_data": {"basename": "shard.00000.mds.zstd"},
                    }
                ]
            }
            (split_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
            self.assertTrue(transcode_mds_to_webdataset(root, purge_mds=True))
            self.assertFalse((root / "mds").exists())
            with tarfile.open(root / "shards" / "train" / "shard-00000.data") as tf:
                names = tf.getnames()
                member = tf.extractfile("s1.json")
                self.assertIsNotNone(member)
                meta = json.load(io.BytesIO(member.read() if member else b"{}"))
            self.assertEqual(names, ["s0.webp", "s0.json", "s1.webp", "s1.json"])
            self.assertEqual(meta["class_id"], 9)
            self.assertEqual(meta["label"], "cat")


if __name__ == "__main__":
    unittest.main()
