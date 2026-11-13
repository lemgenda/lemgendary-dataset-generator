"""R4-B3: RetinaFace boxes and 5-point landmarks from per-file source landmarks."""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from core.config_schema import DatasetEntry
from core.face_geometry import face_geometry, plausibility_problems
from core.paths import UNIFIED_DATA
from core.sample_policy import KeyLayout
from core.source_labels import landmark_scheme
from services.source_label_service import SourceLabelService
from tools.repair.landmark_labeler import LandmarkLabeler
from tools.repair.source_label_fix import DROPPED, LABELLED, SourceLabelError
from utils.label_audit import audit_labels, label_audit_problems
from utils.sample_group import add_tar_member
from utils.shard_audit import find_shards

_SRC = "sfhq-part-1"
_SIZE = 100
_SCHEME = landmark_scheme("sfhq_110")
_LAYOUTS = {_SRC: KeyLayout(_SRC, copies=1, photos=2, files_dir="part1")}
_OLD = {"face_count": 2, "filename": "x.jpg",
        "objects": [{"category_id": 0, "bbox_cxcywh_normalized": [0.5, 0.5, 0.9, 0.95],
                     "landmarks_normalized": [0.4, 0.5, 0.6, 0.5, 0.5, 0.6, 0.42, 0.72,
                                              0.58, 0.72]},
                    {"category_id": 0, "bbox_cxcywh_normalized": [0.1, 0.1, 0.1, 0.1]}]}


def _points(shift: float = 0.0) -> np.ndarray:
    """110 points: a jaw, brows, nose, eyes and mouth of a frontal face (iBUG 0-67)."""
    pts = np.zeros((110, 2))
    pts[0:17, 0] = np.linspace(20, 80, 17)
    pts[0:17, 1] = 50 + 40 * np.sin(np.linspace(0, np.pi, 17))
    pts[17:27] = [[25 + 5 * i, 38] for i in range(5)] + [[55 + 5 * i, 38] for i in range(5)]
    pts[27:36] = [[50, 45], [50, 50], [50, 55], [50, 62], [44, 64], [47, 65], [50, 66],
                  [53, 65], [56, 64]]
    pts[36:42] = [[33, 46], [36, 44], [40, 44], [43, 46], [40, 48], [36, 48]]
    pts[42:48] = [[57, 46], [60, 44], [64, 44], [67, 46], [64, 48], [60, 48]]
    pts[48:68] = [[38 + 24 * i / 19, 75 + 3 * np.sin(np.pi * i / 19)] for i in range(20)]
    pts[48], pts[54] = [38, 75], [62, 75]
    pts[68:] = [[50, 10]] * 42
    pts[:, 0] += shift
    return pts


def _webp() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (_SIZE, _SIZE), (100, 80, 60)).save(buf, format="WEBP")
    return buf.getvalue()


def _key(idx: int) -> str:
    return f"RetinaFace_{_SRC}_{idx:09d}"


def _manifold(root: Path, points: list[np.ndarray]) -> tuple[Path, Path]:
    """Copy 0, photos 1-2 with the old detector label; one landmark file per photo."""
    manifold = root / "LemGendizedFaces"
    for split, idx in (("train", 1), ("val", 2), ("train", 0)):
        shard = manifold / f"shards/{split}/shard-0000{idx // 2}.data"
        shard.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if shard.exists() else "w"
        with tarfile.open(shard, mode) as tf:
            add_tar_member(tf, f"{_key(idx)}.webp", _webp())
            add_tar_member(tf, f"{_key(idx)}.json", json.dumps(_OLD).encode())
    source = root / "sources/part1/landmarks/landmarks"
    source.mkdir(parents=True)
    for file_id, pts in enumerate(points):
        np.savez(source / f"SFHQ_pt1_{file_id + 1:08d}.npz", landmarks=pts)
    return manifold, root / "sources"


def _service(root: Path) -> SourceLabelService:
    return SourceLabelService(LandmarkLabeler(_SCHEME, "sfhq-110pt"), _LAYOUTS, root, 1,
                              "face_landmarks")


def _labels(manifold: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for shard in find_shards(manifold / "shards"):
        with tarfile.open(shard, "r:") as tf:
            for member in tf:
                if member.name.endswith(".json"):
                    handle = tf.extractfile(member)
                    out[member.name[:-5]] = json.loads(handle.read() if handle else b"{}")
    return out


class FaceGeometryTests(unittest.TestCase):
    """Five points in image order and a WIDER-style box."""

    def test_points_and_box(self) -> None:
        """Eye centres, nose tip, mouth corners; box from the face points plus forehead."""
        face = face_geometry(_points(), _SCHEME, _SIZE, _SIZE)
        (lex, ley), (rex, _), nose, mouth_l, mouth_r = face.points
        self.assertAlmostEqual(lex, 38.0)
        self.assertAlmostEqual(ley, 46.0)
        self.assertAlmostEqual(rex, 62.0)
        self.assertEqual(nose, (50.0, 62.0))
        self.assertEqual((mouth_l, mouth_r), ((38.0, 75.0), (62.0, 75.0)))
        x0, y0, x1, y1 = face.box
        self.assertEqual((x0, x1), (20.0, 80.0))
        self.assertAlmostEqual(y1, 90.0)
        self.assertAlmostEqual(y0, 38 - 0.2 * (90 - 38))
        self.assertEqual(plausibility_problems(face), [])

    def test_wide_smile_is_plausible(self) -> None:
        """Mouth corners at nose height pass; the mouth centre stays below the nose."""
        pts = _points()
        pts[48], pts[54] = [38, 62], [62, 62]
        self.assertEqual(plausibility_problems(face_geometry(pts, _SCHEME, _SIZE, _SIZE)), [])

    def test_nose_at_eye_height_is_implausible(self) -> None:
        """A nose tip on the eye line (a failed landmark fit) is rejected."""
        pts = _points()
        pts[30] = [50, 46]
        problems = plausibility_problems(face_geometry(pts, _SCHEME, _SIZE, _SIZE))
        self.assertIn("eyes, nose and mouth not top to bottom", problems)

    def test_mirrored_face_is_implausible(self) -> None:
        """Swapped eyes (subject-left first) fail the image-order check."""
        pts = _points()
        pts[36:42], pts[42:48] = pts[42:48].copy(), pts[36:42].copy()
        face = face_geometry(pts, _SCHEME, _SIZE, _SIZE)
        self.assertIn("left eye not left of the right eye", plausibility_problems(face))

    def test_box_is_clipped(self) -> None:
        """A face past the image edge is clipped to the image."""
        face = face_geometry(_points(shift=30), _SCHEME, _SIZE, _SIZE)
        self.assertEqual(face.box[2], float(_SIZE))

    def test_wrong_shape_is_rejected(self) -> None:
        """A landmark array of another size raises ValueError."""
        with self.assertRaises(ValueError):
            face_geometry(np.zeros((68, 2)), _SCHEME, _SIZE, _SIZE)

    def test_registry_declares_retinaface_landmarks(self) -> None:
        """RetinaFace records the landmark scheme and the label folders."""
        data = yaml.safe_load(UNIFIED_DATA.read_text(encoding="utf-8"))
        entry = DatasetEntry(**data["datasets"]["retinaface_mobilenet"])
        assert entry.source_labels is not None
        self.assertEqual(entry.source_labels.scheme, "sfhq_110")
        self.assertEqual(entry.source_labels.source, "sfhq-110pt")
        self.assertEqual([lay.files_dir for lay in entry.key_layouts],
                         ["part1", "part2", "part3", "part4"])


class LandmarkRepairTests(unittest.TestCase):
    """Plan, apply and re-plan; splits stay where they are."""

    def test_apply_replaces_the_detector_faces(self) -> None:
        """One face per photo from its own file, deltas reported, labels valid, split kept."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold, root = _manifold(Path(tmp), [_points(), _points(shift=5)])
            service = _service(root)
            plan = service.plan(manifold)
            self.assertEqual(plan.statuses()[LABELLED], 2)
            self.assertIn("box_area_ratio", plan.metrics())
            self.assertEqual(plan.metrics()["old_faces"], [2.0, 2.0])
            remaining = service.apply(plan)
            self.assertFalse(remaining.changed, remaining.report())
            labels = _labels(manifold)
            first, second = labels[_key(1)], labels[_key(2)]
            self.assertEqual(first["face_count"], 1)
            self.assertEqual(first["landmark_source"], "sfhq-110pt")
            self.assertEqual(first["filename"], "x.jpg")
            self.assertAlmostEqual(first["objects"][0]["landmarks_normalized"][0], 0.38)
            self.assertAlmostEqual(second["objects"][0]["landmarks_normalized"][0], 0.43)
            self.assertEqual(labels[_key(0)], _OLD)
            self.assertTrue((manifold / "shards/val/shard-00001.data").is_file())
            audit = audit_labels(manifold / "shards", "face_landmarks")
            self.assertEqual(label_audit_problems(audit), [])

    def test_implausible_landmarks_are_dropped(self) -> None:
        """A failed fit (swapped eyes) drops the photo; shards and metadata follow."""
        with tempfile.TemporaryDirectory() as tmp:
            bad = _points()
            bad[36:42], bad[42:48] = bad[42:48].copy(), bad[36:42].copy()
            manifold, root = _manifold(Path(tmp), [_points(), bad])
            index = [{"name": _key(i), "split": "train"} for i in range(3)]
            (manifold / "index.json").write_text(json.dumps(index), encoding="utf-8")
            (manifold / "dataset_info.yaml").write_text("count: 3\n", encoding="utf-8")
            service = _service(root)
            plan = service.plan(manifold)
            self.assertEqual(plan.statuses()[DROPPED], 1)
            self.assertEqual(plan.problems, 0)
            remaining = service.apply(plan)
            self.assertFalse(remaining.changed, remaining.report())
            self.assertNotIn(_key(2), _labels(manifold))
            self.assertFalse((manifold / "shards/val/shard-00001.data").exists())
            names = [e["name"] for e in json.loads((manifold / "index.json").read_text("utf-8"))]
            self.assertEqual(names, [_key(0), _key(1)])
            info = yaml.safe_load((manifold / "dataset_info.yaml").read_text(encoding="utf-8"))
            self.assertEqual(info["count"], 2)
            self.assertEqual(info["landmark_source"], "sfhq-110pt")

    def test_missing_landmark_files_are_refused(self) -> None:
        """Fewer landmark files than photos in the layout: nothing is planned or written."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold, root = _manifold(Path(tmp), [_points()])
            with self.assertRaises(SourceLabelError):
                _service(root).plan(manifold)
            self.assertEqual(_labels(manifold)[_key(1)], _OLD)

if __name__ == "__main__":
    unittest.main()
