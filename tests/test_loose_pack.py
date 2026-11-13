"""R4-J1: a loose YOLO images/labels manifold packs in place into JSON-labelled shards."""

from __future__ import annotations

import io
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image
from typer.testing import CliRunner

from cli import app
from services.loose_pack_service import (
    LoosePackError,
    LoosePackService,
    dataset_class_names,
)


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 12), (90, 60, 30)).save(buf, format="JPEG")
    return buf.getvalue()


class LoosePackTests(unittest.TestCase):
    """repair pack-loose: plan, apply, verification, metadata, purge."""

    def setUp(self) -> None:
        """A YOLO layout: two train images (one without objects), one val image."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.manifold = self.root / "LemGendizedDemoYolo"
        labels = {("train", "a"): "0 0.5 0.5 0.2 0.4\n1 0.3 0.3 0.1 0.1\n",
                  ("train", "b"): "", ("val", "c"): "1 0.6 0.6 0.2 0.2\n"}
        for (split, key), text in labels.items():
            (self.manifold / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.manifold / "labels" / split).mkdir(parents=True, exist_ok=True)
            (self.manifold / "images" / split / f"{key}.jpg").write_bytes(_jpeg())
            (self.manifold / "labels" / split / f"{key}.txt").write_text(text, encoding="utf-8")
        (self.manifold / "classes.txt").write_text("face\nperson\n", encoding="utf-8")
        (self.manifold / "dataset_info.yaml").write_text(
            "name: LemGendizedDemoYolo\ncount: 3\ntask: detection\ncanonical_format: directory\n",
            encoding="utf-8")
        (self.manifold / "index.json").write_text(json.dumps([
            {"id": i, "name": k, "split": s} for i, (s, k) in
            enumerate((("train", "a"), ("train", "b"), ("val", "c")))]), encoding="utf-8")

    def test_plan_counts(self) -> None:
        """The plan counts images, labels and empty labels, and reads classes.txt."""
        plan = LoosePackService("yolo_boxes").plan(self.manifold)
        train = next(s for s in plan.splits if s.name == "train")
        self.assertEqual((train.images, train.labels, train.empty_labels), (2, 2, 1))
        self.assertEqual(plan.classes, ["face", "person"])
        self.assertEqual(plan.blockers, [])

    def test_apply_packs_verifies_and_purges(self) -> None:
        """Shards hold .webp + .json only; metadata records classes; loose folders go."""
        service = LoosePackService("yolo_boxes")
        report = service.apply(service.plan(self.manifold))
        self.assertIn("3 samples", report[0])
        shard = self.manifold / "shards" / "train" / "shard-00000.data"
        with tarfile.open(shard) as tf:
            self.assertEqual(tf.getnames(), ["a.webp", "a.json", "b.webp", "b.json"])
            handle = tf.extractfile("b.json")
            assert handle is not None
            self.assertEqual(json.loads(handle.read()), {"object_count": 0, "objects": []})
        info = yaml.safe_load((self.manifold / "dataset_info.yaml").read_text(encoding="utf-8"))
        self.assertEqual((info["classes"], info["label_json"], info["canonical_format"]),
                         (["face", "person"], "yolo_boxes", "webdataset"))
        self.assertFalse((self.manifold / "images").exists())
        self.assertFalse((self.manifold / "labels").exists())

    def test_unlabelled_image_blocks(self) -> None:
        """An image without a label file blocks the pack; nothing is written."""
        (self.manifold / "labels" / "val" / "c.txt").unlink()
        service = LoosePackService("yolo_boxes")
        plan = service.plan(self.manifold)
        self.assertEqual(plan.blockers, ["1 images have no label file"])
        with self.assertRaises(LoosePackError):
            service.apply(plan)
        self.assertFalse((self.manifold / "shards").exists())

    def test_yaml_names(self) -> None:
        """A YOLO data.yaml with an id -> name mapping gives the class list."""
        (self.manifold / "classes.txt").unlink()
        (self.manifold / "data.yaml").write_text("names:\n  1: person\n  0: face\n",
                                                 encoding="utf-8")
        self.assertEqual(dataset_class_names(self.manifold), ["face", "person"])

    def test_cli_dry_run_then_apply(self) -> None:
        """The dry run changes nothing; --apply packs."""
        dry = CliRunner().invoke(app, ["repair", "pack-loose", "--manifold", str(self.manifold),
                                       "--strategy", "yolo_boxes"])
        self.assertEqual(dry.exit_code, 0, dry.output)
        self.assertTrue((self.manifold / "images").exists())
        done = CliRunner().invoke(app, ["repair", "pack-loose", "--manifold", str(self.manifold),
                                        "--strategy", "yolo_boxes", "--apply"])
        self.assertEqual(done.exit_code, 0, done.output)
        self.assertTrue((self.manifold / "shards" / "val" / "shard-00000.data").is_file())


if __name__ == "__main__":
    unittest.main()
