"""Labels-only repair: loose labels/ files into shard JSON sidecars, without recompiling."""

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
from core.source_manifolds import label_strategy_for_folder
from services.label_repair_service import LabelRepairService, LabelSource
from tests.tar_fixtures import write_tar
from tools.repair.label_restore import LabelRestoreError, index_label_files
from utils.label_json import face_landmarks_json
from utils.sample_group import RawMember, RawSample, encode_sample_or_reject
from utils.shard_audit import audit_shard

_BOX = "0.496435 0.501429 0.987221 0.996843"
_POINTS = ("0.375128 0.470168 0.627386 0.467841 0.512746 0.643001"
           " 0.403877 0.751989 0.604793 0.753116")
_FOLDER = "LemGendizedRetinaFace"


def _sidecar(box: str) -> bytes:
    """A compiled sidecar: split, filename and the box without landmarks."""
    meta = {"split": "train", "filename": "a.jpg", "face_count": 1,
            "objects": [{"category_id": 0,
                         "bbox_cxcywh_normalized": [float(v) for v in box.split()]}]}
    return json.dumps(meta, indent=2).encode("utf-8")


def _sidecars(shard: Path) -> dict[str, dict]:
    found: dict[str, dict] = {}
    with tarfile.open(shard, "r:") as tf:
        for member in tf.getmembers():
            handle = tf.extractfile(member) if member.name.endswith(".json") else None
            if handle is not None:
                found[member.name] = json.loads(handle.read())
    return found


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


class LabelRepairTestBase(unittest.TestCase):
    """A manifold with three samples and a loose labels/ folder beside it."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.manifold = self.root / _FOLDER
        self.shard = write_tar(self.manifold / "shards" / "train" / "shard-00000.data", [
            ("a.webp", b"img-a"), ("a.json", _sidecar(_BOX)),
            ("b.webp", b"img-b"),
            ("c.webp", b"img-c"), ("c.json", _sidecar(_BOX)),
        ])
        info = {"name": _FOLDER, "count": 3, "task": "detection"}
        (self.manifold / "dataset_info.yaml").write_text(yaml.safe_dump(info), encoding="utf-8")
        source = self.root / "RetinaFaceLarge"
        self.labels = source / "labels"
        self._label("train", "a", f"0 {_BOX} {_POINTS}")
        self._label("train", "b", f"0 {_BOX} {_POINTS}")
        self._label("val", "unused", f"0 {_BOX}")
        (source / "classes.txt").write_text("face\n", encoding="utf-8")

    def _label(self, split: str, key: str, text: str) -> None:
        path = self.labels / split / f"{key}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")

    def service(self) -> LabelRepairService:
        """Service over the fixture labels with the RetinaFace strategy."""
        return LabelRepairService(LabelSource(self.labels, "face_landmarks", ["face"]))


class LabelRepairServiceTests(LabelRepairTestBase):
    """Plan, apply, verification and idempotence."""

    def test_plan_reports_matches_and_gaps(self) -> None:
        """a gets landmarks, b gets a new sidecar, c has no label, one label is unused."""
        plan = self.service().plan(self.manifold)
        shard_plan = plan.shards[0]
        self.assertEqual(shard_plan.update, frozenset({"a", "b"}))
        self.assertEqual(shard_plan.added_json, 1)
        self.assertEqual(shard_plan.gaps.no_label, ["c"])
        self.assertEqual(plan.unused_labels, 1)
        self.assertEqual(plan.info_changes, {"label_json": "face_landmarks", "classes": ["face"]})
        self.assertTrue(plan.changed)

    def test_apply_merges_and_is_idempotent(self) -> None:
        """Landmarks land in the JSON, other keys stay, a second run finds nothing."""
        service = self.service()
        remaining = service.apply(service.plan(self.manifold))
        self.assertFalse(remaining.changed)
        sidecars = _sidecars(self.shard)
        obj = sidecars["a.json"]["objects"][0]
        self.assertEqual(len(obj["landmarks_normalized"]), 10)
        self.assertEqual(sidecars["a.json"]["split"], "train")
        self.assertIn("b.json", sidecars)
        self.assertNotIn("landmarks_normalized", sidecars["c.json"]["objects"][0])
        audit = audit_shard(self.shard)
        self.assertEqual(audit.samples, 3)
        self.assertFalse(audit.needs_rewrite)
        info = yaml.safe_load((self.manifold / "dataset_info.yaml").read_text(encoding="utf-8"))
        self.assertEqual((info["label_json"], info["classes"]), ("face_landmarks", ["face"]))
        self.assertEqual(list(self.root.rglob("*.repair-tmp")), [])

    def test_mismatched_box_refuses(self) -> None:
        """A label whose box differs from the recorded one stops the repair untouched."""
        self._label("train", "c", f"0 0.1 0.1 0.1 0.1 {_POINTS}")
        before = self.shard.read_bytes()
        service = self.service()
        plan = service.plan(self.manifold)
        self.assertEqual(plan.mismatched, ["c"])
        with self.assertRaises(LabelRestoreError):
            service.apply(plan)
        self.assertEqual(self.shard.read_bytes(), before)

    def test_zero_landmarks_count_as_absent(self) -> None:
        """Ten zeros (the source's 'not detected') write no landmarks, so nothing changes."""
        self._label("train", "a", f"0 {_BOX} " + " ".join(["0.0"] * 10))
        faces = face_landmarks_json(f"0 {_BOX} " + " ".join(["0.0"] * 10))
        assert faces is not None
        self.assertNotIn("landmarks_normalized", faces["objects"][0])
        self.assertEqual(self.service().plan(self.manifold).shards[0].unchanged, 1)

    def test_duplicate_label_key_is_rejected(self) -> None:
        """The same key under train/ and val/ is ambiguous."""
        self._label("val", "a", f"0 {_BOX}")
        with self.assertRaises(ValueError):
            index_label_files(self.labels)

    def test_no_dataset_info_is_noted(self) -> None:
        """Without dataset_info.yaml the shards are planned and the metadata is skipped."""
        (self.manifold / "dataset_info.yaml").unlink()
        plan = self.service().plan(self.manifold)
        self.assertEqual(plan.info_changes, {})
        self.assertTrue(plan.notes)


class LabelRepairCliTests(LabelRepairTestBase):
    """`repair labels`: dry run by default, strategy from the registry."""

    def test_dry_run_then_apply(self) -> None:
        """The dry run changes nothing; --apply writes; the registry names the strategy."""
        runner = CliRunner()
        args = ["repair", "labels", "--manifold", str(self.manifold), "--labels", str(self.labels)]
        before = self.shard.read_bytes()
        result = runner.invoke(app, args)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("face_landmarks", result.output)
        self.assertEqual(self.shard.read_bytes(), before)
        result = runner.invoke(app, [*args, "--apply"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("landmarks_normalized", _sidecars(self.shard)["a.json"]["objects"][0])

    def test_unknown_strategy_exits_1(self) -> None:
        """A manifold the registry gives no strategy needs --strategy."""
        other = self.manifold.rename(self.root / "LemGendizedUnknown")
        result = CliRunner().invoke(app, ["repair", "labels", "--manifold", str(other),
                                          "--labels", str(self.labels)])
        self.assertEqual(result.exit_code, 1, result.output)

    def test_registry_lookup(self) -> None:
        """RetinaFace has face_landmarks; other entries get their task's strategy."""
        self.assertEqual(label_strategy_for_folder(_FOLDER), "face_landmarks")
        self.assertEqual(label_strategy_for_folder("LemGendizedClassificationMaster"),
                         "class_label")
        self.assertEqual(label_strategy_for_folder("LemGendizedYoloV8n"), "yolo_boxes")
        self.assertIsNone(label_strategy_for_folder("LemGendizedUnknown"))


class EmptyTextLabelTests(unittest.TestCase):
    """An empty .txt label (no objects) is packed as an empty JSON label, not dropped."""

    def test_empty_txt_label_is_kept(self) -> None:
        """The image and the strategy's empty document; no .txt member."""
        raw = RawSample(key="k", primary=RawMember(_jpeg(), ".jpg"),
                        label=RawMember(b"", ".txt"), label_json="face_landmarks")
        outcome = encode_sample_or_reject(raw, transcode_webp=False)
        assert outcome.sample is not None, outcome.reason
        self.assertEqual(outcome.sample.member("k.txt"), b"")
        self.assertEqual(json.loads(outcome.sample.member("k.json")),
                         {"face_count": 0, "objects": []})

    def test_empty_json_label_is_rejected(self) -> None:
        """An empty JSON label is still a defect."""
        raw = RawSample(key="k", primary=RawMember(_jpeg(), ".jpg"),
                        label=RawMember(b"", ".json"))
        self.assertIsNone(encode_sample_or_reject(raw, transcode_webp=False).sample)


if __name__ == "__main__":
    unittest.main()
