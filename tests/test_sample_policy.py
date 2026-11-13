"""R4-B6: face detection sample policy (drop no-face and auxiliary images, landmark val).

The val draw of a source with a key layout spans all its file ids, so
manifolds built from the same files split alike.
"""

from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from core.compile_sources import fast_scan, is_excluded_dir
from core.config_schema import DatasetEntry
from core.paths import UNIFIED_DATA
from core.sample_policy import (
    FACE_BOX_ONLY,
    FACE_DETECTION,
    FACE_LANDMARKS,
    FACE_NONE,
    ROLE_AUXILIARY,
    ROLE_COPY,
    ROLE_PHOTO,
    KeyLayout,
    face_kind,
    key_role,
    sample_policy,
)
from tools.repair.sample_policy_fix import (
    SampleRule,
    apply_sample_policy,
    enforce_sample_policy,
    plan_sample_policy,
)
from utils.sample_group import add_tar_member
from utils.shard_audit import audit_shard, find_shards

_SRC = "sfhq-part-1"
_BOX = [0.5, 0.5, 0.9, 0.9]
_LMK = [0.4, 0.4, 0.6, 0.4, 0.5, 0.6, 0.4, 0.7, 0.6, 0.7]


def _key(idx: int) -> str:
    return f"RetinaFace_{_SRC}_{idx:09d}"


def _meta(kind: str) -> dict[str, Any]:
    if kind == FACE_NONE:
        return {"face_count": 0, "objects": []}
    obj: dict[str, Any] = {"category_id": 0, "bbox_cxcywh_normalized": _BOX}
    if kind == FACE_LANDMARKS:
        obj["landmarks_normalized"] = _LMK
    return {"face_count": 1, "objects": [obj]}


def _write_shard(path: Path, samples: list[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        for idx, kind in samples:
            add_tar_member(tf, f"{_key(idx)}.webp", b"img%d" % idx)
            add_tar_member(tf, f"{_key(idx)}.json", json.dumps(_meta(kind)).encode())


def _shard_keys(manifold: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for shard in find_shards(manifold / "shards"):
        out.setdefault(shard.parent.name, []).extend(audit_shard(shard).keys)
    return out


class PolicyRuleTests(unittest.TestCase):
    """The policy lookup, face kinds and key roles."""

    def test_policy_from_label_strategy_or_task(self) -> None:
        """face_landmarks and detection-face select the face policy; others none."""
        self.assertIs(sample_policy("face_landmarks"), FACE_DETECTION)
        self.assertIs(sample_policy(None, "detection-face"), FACE_DETECTION)
        self.assertIsNone(sample_policy(None, "restoration"))

    def test_face_kind(self) -> None:
        """No objects, a box, or a box with non-zero landmarks."""
        self.assertEqual(face_kind(None), FACE_NONE)
        self.assertEqual(face_kind(_meta(FACE_NONE)), FACE_NONE)
        self.assertEqual(face_kind(_meta(FACE_BOX_ONLY)), FACE_BOX_ONLY)
        self.assertEqual(face_kind(_meta(FACE_LANDMARKS)), FACE_LANDMARKS)
        zeros = {"objects": [{"bbox_cxcywh_normalized": _BOX, "landmarks_normalized": [0] * 10}]}
        self.assertEqual(face_kind(zeros), FACE_BOX_ONLY)

    def test_key_roles_follow_the_layout(self) -> None:
        """Copies, then photos (file id from 0), then auxiliary images."""
        layouts = {_SRC: KeyLayout(_SRC, copies=2, photos=3)}
        roles = [key_role(_key(i), layouts) for i in range(7)]
        self.assertEqual([r.role for r in roles], [ROLE_COPY, ROLE_COPY, ROLE_PHOTO, ROLE_PHOTO,
                                                   ROLE_PHOTO, ROLE_AUXILIARY, ROLE_AUXILIARY])
        self.assertEqual(roles[2].file_id, "0")
        self.assertEqual(key_role(_key(9), {}).role, ROLE_PHOTO)


class ScanExcludeTests(unittest.TestCase):
    """The compiler never scans registry-excluded folders as images."""

    def test_excluded_folders_are_not_scanned(self) -> None:
        """segmentations/ and the sample folders are skipped, images/ is kept."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ("images/images/a.jpg", "segmentations/segmentations/a.png",
                        "a small sample (550 images)/b.jpg", "A Tiny Sample (30 images)/c.jpg"):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_bytes(b"x")
            excluded = ("segmentations", "a small sample*", "a tiny sample*")
            found = [Path(p).name for p in fast_scan(str(root), {".jpg", ".png"}, excluded)]
            self.assertEqual(found, ["a.jpg"])
            self.assertEqual(len(list(fast_scan(str(root), {".jpg", ".png"}))), 4)
        self.assertTrue(is_excluded_dir("Segmentations", ("segmentations",)))

    def test_registry_declares_the_sfhq_rules(self) -> None:
        """RetinaFace excludes segmentations/ and records the old key layout."""
        data = yaml.safe_load(UNIFIED_DATA.read_text(encoding="utf-8"))
        entry = DatasetEntry(**data["datasets"]["retinaface_mobilenet"])
        self.assertIn("segmentations", entry.scan_exclude_dirs)
        starts = [layout.copies + layout.photos for layout in entry.key_layouts]
        self.assertEqual(starts, [90365, 92151, 119238, 126534])


def _manifold(root: Path) -> Path:
    """A face manifold: copy 0, photos 1-6, auxiliary 7-8, one duplicate in both splits."""
    manifold = root / "LemGendizedFaces"
    _write_shard(manifold / "shards/train/shard-00000.data",
                 [(0, FACE_LANDMARKS), (1, FACE_LANDMARKS), (2, FACE_LANDMARKS),
                  (3, FACE_BOX_ONLY), (4, FACE_NONE), (7, FACE_LANDMARKS)])
    _write_shard(manifold / "shards/val/shard-00000.data",
                 [(5, FACE_LANDMARKS), (6, FACE_BOX_ONLY), (8, FACE_NONE), (1, FACE_LANDMARKS)])
    index = [{"name": _key(i), "split": "val" if i in (5, 6, 8) else "train"}
             for i in range(9)]
    (manifold / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (manifold / "dataset_info.yaml").write_text("count: 9\n", encoding="utf-8")
    return manifold


class SamplePolicyRepairTests(unittest.TestCase):
    """Plan, apply and re-plan on a small manifold."""

    rule = SampleRule(FACE_DETECTION, 0.5, {_SRC: KeyLayout(_SRC, copies=1, photos=6)})

    def test_dry_run_counts(self) -> None:
        """The plan names every drop and move and changes nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold = _manifold(Path(tmp))
            before = _shard_keys(manifold)
            plan = plan_sample_policy(manifold, self.rule)
            self.assertEqual(plan.counts.reasons[ROLE_COPY], 1)
            self.assertEqual(plan.counts.reasons[ROLE_AUXILIARY], 2)
            self.assertEqual(plan.counts.reasons[FACE_NONE], 1)
            self.assertEqual(plan.counts.reasons["duplicate copy"], 1)
            self.assertEqual(plan.counts.after["samples"], 5)
            self.assertEqual(plan.counts.val_by_source[_SRC], (2, 3))
            self.assertEqual(_shard_keys(manifold), before)

    def test_apply_then_nothing_left(self) -> None:
        """Val holds only landmark photos, metadata follows, a re-plan is clean."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold = _manifold(Path(tmp))
            remaining = apply_sample_policy(plan_sample_policy(manifold, self.rule))
            self.assertFalse(remaining.changed, remaining.report())
            keys = _shard_keys(manifold)
            landmark_photos = {_key(i) for i in (1, 2, 5)}
            self.assertEqual(len(keys["val"]), 2)
            self.assertTrue(set(keys["val"]) <= landmark_photos)
            self.assertIn(_key(3), keys["train"])
            self.assertIn(_key(6), keys["train"])
            all_keys = keys["train"] + keys["val"]
            self.assertEqual(sorted(all_keys), sorted({*landmark_photos, _key(3), _key(6)}))
            names = sorted(p.name for p in find_shards(manifold / "shards"))
            self.assertTrue(all(n.startswith("shard-0000") for n in names))
            index = json.loads((manifold / "index.json").read_text(encoding="utf-8"))
            self.assertEqual({e["name"]: e["split"] for e in index},
                             {k: s for s, ks in keys.items() for k in ks})
            info = yaml.safe_load((manifold / "dataset_info.yaml").read_text(encoding="utf-8"))
            self.assertEqual(info["count"], 5)

    def test_split_is_deterministic(self) -> None:
        """Two plans of the same manifold choose the same val samples."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold = _manifold(Path(tmp))
            first = plan_sample_policy(manifold, self.rule).final
            self.assertEqual(plan_sample_policy(manifold, self.rule).final, first)

    def test_layout_draw_ignores_dropped_files(self) -> None:
        """Two manifolds of the same source files share one val set by file id."""
        rule = SampleRule(FACE_DETECTION, 0.5, {_SRC: KeyLayout(_SRC, copies=0, photos=12)})
        with tempfile.TemporaryDirectory() as tmp:
            full = Path(tmp) / "full"
            _write_shard(full / "shards/train/shard-00000.data",
                         [(i, FACE_LANDMARKS) for i in range(12)])
            full_val = {k for k, s in plan_sample_policy(full, rule).final.items() if s == "val"}
            self.assertEqual(len(full_val), 6)
            gone = sorted(full_val)[:2]
            part = Path(tmp) / "part"
            _write_shard(part / "shards/train/shard-00000.data",
                         [(i, FACE_LANDMARKS) for i in range(12) if _key(i) not in gone])
            plan = plan_sample_policy(part, rule)
            part_val = {k for k, s in plan.final.items() if s == "val"}
            self.assertEqual(part_val, full_val - set(gone))
            self.assertEqual(plan.counts.val_by_source[_SRC], (4, 10))

    def test_unregistered_folder_is_left_alone(self) -> None:
        """The compiler hook does nothing for a manifold without a policy."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold = _manifold(Path(tmp))
            before = _shard_keys(manifold)
            self.assertTrue(enforce_sample_policy(manifold))
            self.assertEqual(_shard_keys(manifold), before)


if __name__ == "__main__":
    unittest.main()
