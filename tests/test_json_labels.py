"""R4-J1: JSON is the only label format inside shards; text labels are repaired into it."""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from cli import app
from core.kaggle_file_download import list_dataset_files, rate_limit_wait
from core.kaggle_http import KaggleRateLimitError
from core.kaggle_identity import default_owner
from core.source_manifolds import find_registry_manifold
from services.repair_service import RepairService
from services.txt_label_service import TxtLabelService
from tests.tar_fixtures import write_tar
from tools.repair.txt_label_fix import TxtLabelError, merge_label
from utils.label_audit import audit_labels, label_audit_problems
from utils.label_json import (
    LABEL_STRATEGIES,
    TASK_STRATEGIES,
    label_document,
    label_problems,
    resolve_label_strategy,
)

_BOX = "0.5 0.5 0.2 0.4"
_FACE = f"0 {_BOX} " + " ".join(["0.4"] * 10)


def _member(shard: Path, name: str) -> bytes:
    with tarfile.open(shard) as tf:
        handle = tf.extractfile(name)
        return handle.read() if handle else b""


def _names(shard: Path) -> list[str]:
    with tarfile.open(shard) as tf:
        return tf.getnames()


class _Http429(OSError):
    """An HTTP error carrying a 429 response, like requests' HTTPError."""

    response = SimpleNamespace(status_code=429)


class LabelDocumentTests(unittest.TestCase):
    """Every strategy builds a document that passes its own schema."""

    def test_documents_are_valid(self) -> None:
        """One label per strategy, validated against that strategy."""
        cases = {
            "face_landmarks": _FACE,
            "yolo_boxes": f"3 {_BOX}\n0 {_BOX} 0.1 0.2 2\n",
            "yolo_polygons": "1 0.1 0.1 0.9 0.1 0.5 0.9",
            "nima": " ".join(["0.1"] * 10),
            "class_label": "4",
            "caption": "a red house",
        }
        self.assertEqual(set(cases), set(LABEL_STRATEGIES))
        for strategy, text in cases.items():
            with self.subTest(strategy=strategy):
                doc = label_document(strategy, text)
                assert doc is not None
                self.assertEqual(label_problems(doc, strategy), [])

    def test_yolo_boxes_layout(self) -> None:
        """Boxes are cxcywh with an int category id; extra values are keypoints."""
        doc = label_document("yolo_boxes", f"3 {_BOX}\n0 {_BOX} 0.1 0.2 2\n")
        assert doc is not None
        self.assertEqual(doc["object_count"], 2)
        self.assertEqual(doc["objects"][0], {"category_id": 3,
                                             "bbox_cxcywh_normalized": [0.5, 0.5, 0.2, 0.4]})
        self.assertEqual(doc["objects"][1]["keypoints"], [0.1, 0.2, 2.0])

    def test_empty_labels(self) -> None:
        """An empty detection label is a valid negative; an empty NIMA label is nothing."""
        self.assertEqual(label_document("yolo_boxes", ""), {"object_count": 0, "objects": []})
        self.assertEqual(label_document("face_landmarks", "\n"), {"face_count": 0, "objects": []})
        self.assertIsNone(label_document("nima", ""))

    def test_unreadable_labels(self) -> None:
        """Garbage or a lost label raises instead of being written."""
        for strategy, text in (("yolo_boxes", "0 a b c d"), ("yolo_boxes", "0 0.5"),
                               ("nima", "0.1 0.2"), ("face_landmarks", "junk"),
                               ("class_label", "1.5")):
            with self.subTest(strategy=strategy, text=text), self.assertRaises(ValueError):
                label_document(strategy, text)

    def test_class_label(self) -> None:
        """A number is a class id, anything else a class name."""
        self.assertEqual(label_document("class_label", "7\n"), {"class_id": 7})
        self.assertEqual(label_document("class_label", "cat"), {"class_name": "cat"})

    def test_problems(self) -> None:
        """Out-of-range boxes, wrong counts and missing keys are reported."""
        bad = {"object_count": 2, "objects": [{"category_id": -1,
                                                 "bbox_cxcywh_normalized": [0.5, 1.5, 0.1, 0.1]}]}
        problems = label_problems(bad, "yolo_boxes")
        self.assertEqual(len(problems), 3, problems)
        self.assertEqual(label_problems({"source": "x"}, "nima"), ["missing nima_scores"])
        self.assertEqual(label_problems([1, 2]), ["the JSON label is not an object"])

    def test_task_strategies(self) -> None:
        """label_json wins; otherwise the task decides."""
        self.assertEqual(resolve_label_strategy(None, "detection"), "yolo_boxes")
        self.assertEqual(resolve_label_strategy("face_landmarks", "pose"), "face_landmarks")
        self.assertIsNone(resolve_label_strategy(None, "forex"))
        self.assertTrue(set(TASK_STRATEGIES.values()) <= set(LABEL_STRATEGIES))


class MergeTests(unittest.TestCase):
    """Text labels merge into JSON without losing anything."""

    def test_merge_keeps_json_and_adds_text_keys(self) -> None:
        """Agreeing values keep the JSON's; keys only in the text are added."""
        meta = {"split": "train", "objects": [{"category_id": 0,
                                               "bbox_cxcywh_normalized": [0.5, 0.5, 0.2, 0.4],
                                               "landmarks_normalized": [0.1] * 10}]}
        derived = label_document("face_landmarks", f"0 {_BOX}")
        assert derived is not None
        merged, conflicts = merge_label(meta, derived)
        self.assertEqual(conflicts, [])
        self.assertEqual(merged["face_count"], 1)
        self.assertEqual(merged["split"], "train")
        self.assertEqual(merged["objects"][0]["landmarks_normalized"], [0.1] * 10)

    def test_conflicts(self) -> None:
        """A different box or object count is a conflict."""
        meta = {"objects": [{"category_id": 0, "bbox_cxcywh_normalized": [0.1, 0.5, 0.2, 0.4]}]}
        derived = label_document("yolo_boxes", f"0 {_BOX}")
        assert derived is not None
        self.assertEqual(len(merge_label(meta, derived)[1]), 1)
        two = label_document("yolo_boxes", f"0 {_BOX}\n0 {_BOX}")
        assert two is not None
        self.assertIn("entries", merge_label(meta, two)[1][0])


class TxtLabelRepairTests(unittest.TestCase):
    """repair txt-labels: dry run, apply, MISMATCH refusal, idempotence, verifier."""

    def setUp(self) -> None:
        """A manifold with text labels, one beside a JSON sidecar."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.manifold = self.root / "LemGendizedDemo"
        self.shard = self.manifold / "shards" / "train" / "shard-00000.data"
        self.sidecar = {"split": "train", "objects": [
            {"category_id": 0, "bbox_cxcywh_normalized": [0.5, 0.5, 0.2, 0.4]}]}
        self._write(self.sidecar)

    def _write(self, sidecar: dict[str, Any]) -> None:
        write_tar(self.shard, [
            ("a.webp", b"img-a"), ("a.txt", f"0 {_BOX}\n".encode()),
            ("a.json", json.dumps(sidecar).encode()),
            ("b.webp", b"img-b"), ("b.txt", b""),
            ("c.webp", b"img-c"), ("c.json", b'{"object_count":0,"objects":[]}'),
        ])

    def test_plan_apply_and_rerun(self) -> None:
        """Text labels become JSON, the .txt members go, a re-run finds nothing."""
        service = TxtLabelService("yolo_boxes")
        plan = service.plan(self.manifold)
        self.assertTrue(plan.changed)
        self.assertEqual((len(plan.shards[0].text_keys), plan.shards[0].tally.merged,
                          len(plan.shards[0].new_json)), (2, 1, 1))
        remaining = service.apply(plan)
        self.assertFalse(remaining.changed)
        self.assertEqual(_names(self.shard),
                         ["a.webp", "a.json", "b.webp", "b.json", "c.webp", "c.json"])
        merged = json.loads(_member(self.shard, "a.json"))
        self.assertEqual((merged["split"], merged["object_count"]), ("train", 1))
        self.assertEqual(json.loads(_member(self.shard, "b.json")),
                         {"object_count": 0, "objects": []})
        self.assertEqual(label_audit_problems(audit_labels(self.manifold / "shards",
                                                           "yolo_boxes")), [])

    def test_mismatch_refuses(self) -> None:
        """A text label that contradicts its JSON blocks the apply; nothing changes."""
        self._write({"objects": [{"category_id": 0,
                                  "bbox_cxcywh_normalized": [0.1, 0.1, 0.1, 0.1]}]})
        before = self.shard.read_bytes()
        service = TxtLabelService("yolo_boxes")
        plan = service.plan(self.manifold)
        self.assertEqual(plan.mismatched, ["a"])
        self.assertTrue(any("MISMATCH" in line for line in plan.report()))
        with self.assertRaises(TxtLabelError):
            service.apply(plan)
        self.assertEqual(self.shard.read_bytes(), before)

    def test_cli_and_verifier(self) -> None:
        """The verifier flags .txt members until the CLI repair runs."""
        problems = RepairService().plan(self.manifold).label_problems
        self.assertTrue(any("text label members" in p for p in problems), problems)
        dry = CliRunner().invoke(app, ["repair", "txt-labels", "--manifold", str(self.manifold),
                                       "--strategy", "yolo_boxes"])
        self.assertEqual(dry.exit_code, 0, dry.output)
        self.assertIn(".txt", _names(self.shard)[1])
        done = CliRunner().invoke(app, ["repair", "txt-labels", "--manifold", str(self.manifold),
                                        "--strategy", "yolo_boxes", "--apply"])
        self.assertEqual(done.exit_code, 0, done.output)
        self.assertEqual(RepairService().plan(self.manifold).label_problems, [])


class InvalidJsonAuditTests(unittest.TestCase):
    """The label audit reads every JSON member."""

    def test_invalid_json_reported(self) -> None:
        """Broken JSON and schema violations are both problems."""
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        write_tar(root / "shards" / "train" / "s.data", [
            ("a.webp", b"x"), ("a.json", b"{not json"),
            ("b.webp", b"x"), ("b.json", b'{"objects":[{"category_id":0}]}'),
        ])
        audit = audit_labels(root / "shards")
        self.assertEqual(sorted(audit.invalid), ["a.json", "b.json"])
        self.assertIn("2 invalid JSON labels", label_audit_problems(audit)[0])


class RegistryAndRateLimitTests(unittest.TestCase):
    """A pulled manifold gets its entry's strategy; Kaggle 429s wait and retry."""

    def test_registry_manifold(self) -> None:
        """The YOLOv8n entry packs YOLO boxes."""
        found = find_registry_manifold(f"{default_owner()}/lemgendizedyolov8n")
        assert found is not None
        self.assertEqual((found.modernized_folder, found.label_json),
                         ("LemGendizedYoloV8n", "yolo_boxes"))

    def test_rate_limit_wait(self) -> None:
        """Retry-After wins; other errors are not rate limits."""
        self.assertEqual(rate_limit_wait(KaggleRateLimitError("x", 7.0), 0), 7.0)
        self.assertEqual(rate_limit_wait(_Http429("429"), 0), 30.0)
        self.assertIsNone(rate_limit_wait(OSError("boom"), 0))

    def test_listing_retries_429(self) -> None:
        """A rate-limited listing page waits and is asked again."""
        calls: list[Any] = []
        waits: list[float] = []

        def _list(_handle: str, page_token: Any = None, page_size: int = 0) -> Any:
            calls.append((page_token, page_size))
            if len(calls) == 1:
                raise _Http429("429 Too Many Requests")
            return SimpleNamespace(files=[SimpleNamespace(name="a", total_bytes=1)],
                                   next_page_token=None)

        api = SimpleNamespace(dataset_list_files=_list)
        files = list_dataset_files(api, "o/s", sleep=waits.append)
        self.assertEqual([f.name for f in files], ["a"])
        self.assertEqual((len(calls), waits), (2, [30.0]))


if __name__ == "__main__":
    unittest.main()
