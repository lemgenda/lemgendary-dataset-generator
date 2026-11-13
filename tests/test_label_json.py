"""RetinaFace landmarks reach the shard JSON; ParseNet image copies are not masks (R2-D3)."""

from __future__ import annotations

import io
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from pydantic import ValidationError

from core.config_schema import DatasetEntry
from core.kaggle_identity import default_owner
from core.paths import UNIFIED_DATA
from core.source_manifolds import find_source_manifold
from tools.modernize.loose_convert import convert_loose_directory_to_webdataset
from tools.txt_to_json import parse_retinaface_lines
from utils.label_json import face_landmarks_json
from utils.sample_group import RawMember, RawSample, encode_sample_or_reject

_LINE = ("0 0.496435 0.501429 0.987221 0.996843 0.375128 0.470168 0.627386 0.467841"
         " 0.512746 0.643001 0.403877 0.751989 0.604793 0.753116")


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 12), (90, 60, 30)).save(buf, format="JPEG")
    return buf.getvalue()


class FaceLabelJsonTests(unittest.TestCase):
    """The face_landmarks strategy keeps the box and the ten landmark values."""

    def test_full_line_keeps_landmarks(self) -> None:
        """Box plus five points: 4 box values and 10 landmark values."""
        faces = face_landmarks_json(_LINE + "\n")
        assert faces is not None
        obj = faces["objects"][0]
        self.assertEqual(faces["face_count"], 1)
        self.assertEqual(obj["category_id"], 0)
        self.assertEqual(len(obj["bbox_cxcywh_normalized"]), 4)
        self.assertEqual(len(obj["landmarks_normalized"]), 10)
        self.assertAlmostEqual(obj["landmarks_normalized"][0], 0.375128)

    def test_box_only_and_empty(self) -> None:
        """A box-only line has no landmarks; an empty label has no faces."""
        faces = face_landmarks_json("0 0.5 0.5 0.2 0.2")
        assert faces is not None
        self.assertNotIn("landmarks_normalized", faces["objects"][0])
        self.assertIsNone(face_landmarks_json(""))
        with self.assertRaises(ValueError):
            face_landmarks_json("0 a b c d")

    def test_legacy_txt_to_json_keeps_landmarks(self) -> None:
        """tools/txt_to_json uses the same builder."""
        obj = parse_retinaface_lines([_LINE + "\n"])["objects"][0]
        self.assertEqual(len(obj["landmarks_normalized"]), 10)
        self.assertEqual(parse_retinaface_lines([]), {"face_count": 0, "objects": []})


class SampleEncodingTests(unittest.TestCase):
    """The shared encoder adds the JSON sidecar and rejects image copies as masks."""

    def test_label_json_member(self) -> None:
        """A .txt label becomes the .json label; without a strategy the sample is rejected."""
        raw = RawSample("a", RawMember(_jpeg(), ".jpg"),
                        label=RawMember(_LINE.encode(), ".txt"), label_json="face_landmarks")
        sample = encode_sample_or_reject(raw, transcode_webp=True).sample
        assert sample is not None
        self.assertEqual([n for n, _ in sample.members], ["a.webp", "a.json"])
        plain = encode_sample_or_reject(RawSample("a", raw.primary, label=raw.label), True)
        self.assertIsNone(plain.sample)
        self.assertIn("without a label strategy", plain.reason)

    def test_mask_copy_of_image_is_rejected(self) -> None:
        """A 'mask' that is the image itself is not a class mask."""
        image = RawMember(_jpeg(), ".jpg")
        outcome = encode_sample_or_reject(RawSample("p", image, mask=image), True)
        self.assertIsNone(outcome.sample)
        self.assertIn("byte copy of the image", outcome.reason)


class LooseConvertTests(unittest.TestCase):
    """images/ + labels/ (the Kaggle RetinaFace layout) pack into .webp/.txt/.json."""

    def test_loose_layout_to_shards(self) -> None:
        """Every sample gets its label and a JSON sidecar with landmarks."""
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        src = root / "LemGendizedRetinaFaceMobileNetLarge"
        for split, stem in (("train", "f0"), ("train", "f1"), ("val", "f2")):
            (src / "images" / split).mkdir(parents=True, exist_ok=True)
            (src / "labels" / split).mkdir(parents=True, exist_ok=True)
            (src / "images" / split / f"{stem}.jpg").write_bytes(_jpeg())
            (src / "labels" / split / f"{stem}.txt").write_text(_LINE + "\n", encoding="utf-8")
        out = root / "LemGendizedRetinaFace"
        convert_loose_directory_to_webdataset(src, out, label_json="face_landmarks",
                                              max_workers=2)
        with tarfile.open(out / "shards" / "train" / "shard-00000.data") as tf:
            names = tf.getnames()
            handle = tf.extractfile("f0.json")
            assert handle is not None
            meta = json.loads(handle.read())
        self.assertEqual(names, ["f0.webp", "f0.json", "f1.webp", "f1.json"])
        self.assertEqual(len(meta["objects"][0]["landmarks_normalized"]), 10)


class RegistryTests(unittest.TestCase):
    """The registry names the RetinaFace source and validates the strategy."""

    def test_retinaface_source_manifold(self) -> None:
        """The Kaggle loose manifold maps to LemGendizedRetinaFace with face_landmarks."""
        found = find_source_manifold(f"{default_owner()}/lemgendizedretinafacemobilenetlarge",
                                     UNIFIED_DATA)
        assert found is not None
        self.assertEqual((found.key, found.modernized_folder, found.label_json),
                         ("retinaface_mobilenet", "LemGendizedRetinaFace", "face_landmarks"))
        self.assertIsNone(find_source_manifold(f"{default_owner()}/unknown", UNIFIED_DATA))

    def test_unknown_strategy_rejected(self) -> None:
        """label_json must name a registered builder."""
        with self.assertRaises(ValidationError):
            DatasetEntry(name="X", label_json="nope")


if __name__ == "__main__":
    unittest.main()
