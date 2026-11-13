"""R4-B1: ParseNet class masks from per-file source masks, face parsing sample policy."""

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
from core.paths import UNIFIED_DATA
from core.sample_policy import (
    FACE_PARSING,
    MASK_NONE,
    MASK_PRESENT,
    ROLE_AUXILIARY,
    ROLE_COPY,
    KeyLayout,
    mask_kind,
    sample_policy,
)
from core.source_labels import CELEBAMASK_HQ_19, mask_scheme
from services.source_label_service import SourceLabelService
from tools.repair.mask_labeler import (
    INVALID_MASK,
    MASK_MEMBER,
    MaskLabeler,
    decode_mask,
)
from tools.repair.sample_policy_fix import (
    SampleRule,
    apply_sample_policy,
    plan_sample_policy,
)
from tools.repair.source_label_fix import LABELLED, SourceFiles, SourceLabelError
from utils.label_audit import audit_labels, label_audit_problems
from utils.label_json import label_document, label_problems
from utils.sample_group import add_tar_member
from utils.shard_audit import find_shards

_SRC = "sfhq-part-1"
_SIZE = 16
_SCHEME = mask_scheme("celebamask_hq_19")


def _key(idx: int) -> str:
    return f"ParseNet_{_SRC}_{idx:09d}"


def _webp(size: int = _SIZE) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (120, 90, 60)).save(buf, format="WEBP")
    return buf.getvalue()


def _mask(seed: int, high: int = 19, size: int = _SIZE) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, high, (size, size), dtype=np.uint8)


def _png(ids: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(ids, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _manifold(root: Path, masks: dict[int, bytes] | None = None) -> tuple[Path, Path]:
    """Copy 0, photos 1-3 (file ids 0-2), seg maps 4-5; one source mask per photo."""
    manifold = root / "LemGendizedFaceParts"
    shard = manifold / "shards/train/shard-00000.data"
    shard.parent.mkdir(parents=True)
    with tarfile.open(shard, "w") as tf:
        for idx in range(6):
            add_tar_member(tf, f"{_key(idx)}.webp", _webp())
    (manifold / "dataset_info.yaml").write_text("count: 6\n", encoding="utf-8")
    source = root / "sources/part1/segmentations/segmentations"
    source.mkdir(parents=True)
    masks = masks or {i: _png(_mask(i)) for i in range(3)}
    for file_id, payload in masks.items():
        (source / f"SFHQ_pt1_{file_id + 1:08d}.png").write_bytes(payload)
    return manifold, root / "sources"


_LAYOUTS = {_SRC: KeyLayout(_SRC, copies=1, photos=3, files_dir="part1")}


def _service(root: Path, workers: int = 1) -> SourceLabelService:
    return SourceLabelService(MaskLabeler(_SCHEME, "sfhq-bisenet"), _LAYOUTS, root, workers,
                              "segmentation_mask")


def _members(manifold: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for shard in find_shards(manifold / "shards"):
        with tarfile.open(shard, "r:") as tf:
            for member in tf:
                handle = tf.extractfile(member)
                out[member.name] = handle.read() if handle else b""
    return out


class MaskSchemeTests(unittest.TestCase):
    """The label strategy, the policy lookup and the registry entry."""

    def test_segmentation_mask_label(self) -> None:
        """A mask label needs num_classes and matching class_names, and has no text form."""
        doc: dict[str, Any] = {"num_classes": 19, "class_names": list(CELEBAMASK_HQ_19),
                               "mask_source": "sfhq-bisenet"}
        self.assertEqual(label_problems(doc, "segmentation_mask"), [])
        self.assertTrue(label_problems({"num_classes": 19}, "segmentation_mask"))
        self.assertTrue(label_problems({**doc, "class_names": ["a"]}, "segmentation_mask"))
        self.assertTrue(label_problems({**doc, "mask_source": ""}))
        self.assertIsNone(label_document("segmentation_mask", ""))
        with self.assertRaises(ValueError):
            label_document("segmentation_mask", "0 0.5 0.5")

    def test_policy_and_kinds(self) -> None:
        """segmentation_mask selects the face parsing policy; kinds follow num_classes."""
        self.assertIs(sample_policy("segmentation_mask", "segmentation"), FACE_PARSING)
        self.assertEqual(mask_kind({"num_classes": 19}), MASK_PRESENT)
        self.assertEqual(mask_kind({}), MASK_NONE)
        self.assertEqual(mask_kind(None), MASK_NONE)

    def test_registry_declares_parsenet_sources(self) -> None:
        """ParseNet: mask strategy, scan excludes, key layouts with files_dir, mask scheme."""
        data = yaml.safe_load(UNIFIED_DATA.read_text(encoding="utf-8"))
        entry = DatasetEntry(**data["datasets"]["parsenet"])
        self.assertEqual(entry.label_json, "segmentation_mask")
        self.assertIn("segmentations", entry.scan_exclude_dirs)
        self.assertEqual([lay.files_dir for lay in entry.key_layouts],
                         ["part1", "part2", "part3", "part4"])
        self.assertEqual([lay.copies + lay.photos for lay in entry.key_layouts],
                         [90365, 92151, 119238, 126534])
        assert entry.source_labels is not None
        self.assertEqual(entry.source_labels.scheme, "celebamask_hq_19")
        self.assertEqual(entry.source_labels.source, "sfhq-bisenet")


class SourceFilesTests(unittest.TestCase):
    """The label file index refuses a folder that does not match the key layout."""

    def test_count_mismatch_is_refused(self) -> None:
        """Two mask files for a layout of three photos."""
        with tempfile.TemporaryDirectory() as tmp:
            _, root = _manifold(Path(tmp), {0: _png(_mask(0)), 1: _png(_mask(1))})
            with self.assertRaises(SourceLabelError):
                SourceFiles.build(root, _LAYOUTS, ".png")

    def test_file_ids_follow_sorted_names(self) -> None:
        """File id j is the j-th name in sorted order."""
        with tempfile.TemporaryDirectory() as tmp:
            _, root = _manifold(Path(tmp))
            files = SourceFiles.build(root, _LAYOUTS, ".png")
            found = files.file_for(_SRC, "2")
            self.assertEqual(found.name if found else "", "SFHQ_pt1_00000003.png")
            self.assertIsNone(files.file_for(_SRC, "3"))


class MaskRepairTests(unittest.TestCase):
    """Plan, apply, re-plan, then the face parsing sample policy."""

    def test_dry_run_changes_nothing(self) -> None:
        """Three photos to label, one copy and two seg maps left alone."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold, root = _manifold(Path(tmp))
            before = _members(manifold)
            plan = _service(root).plan(manifold)
            statuses = plan.statuses()
            self.assertEqual(statuses[LABELLED], 3)
            self.assertEqual(statuses[ROLE_COPY], 1)
            self.assertEqual(statuses[ROLE_AUXILIARY], 2)
            self.assertEqual(plan.problems, 0)
            self.assertEqual(_members(manifold), before)

    def test_apply_writes_exact_masks(self) -> None:
        """Each photo gets its own source mask, class ids exact, a valid JSON label."""
        with tempfile.TemporaryDirectory() as tmp:
            manifold, root = _manifold(Path(tmp))
            service = _service(root, workers=2)
            remaining = service.apply(service.plan(manifold))
            self.assertFalse(remaining.changed, remaining.report())
            self._check_masks(manifold)
            self._check_policy(manifold)

    def _check_masks(self, manifold: Path) -> None:
        """Masks exact per photo, none on the copy or the seg maps, labels valid."""
        members = _members(manifold)
        for file_id in range(3):
            key = _key(file_id + 1)
            ids = decode_mask(members[f"{key}{MASK_MEMBER}"])
            np.testing.assert_array_equal(ids, _mask(file_id))
            meta = json.loads(members[f"{key}.json"])
            self.assertEqual(meta["num_classes"], 19)
            self.assertEqual(meta["mask_source"], "sfhq-bisenet")
        self.assertNotIn(f"{_key(0)}{MASK_MEMBER}", members)
        self.assertNotIn(f"{_key(4)}.json", members)
        audit = audit_labels(manifold / "shards", "segmentation_mask")
        self.assertEqual(label_audit_problems(audit), [])
        info = yaml.safe_load((manifold / "dataset_info.yaml").read_text(encoding="utf-8"))
        self.assertEqual(info["label_json"], "segmentation_mask")
        self.assertEqual(info["classes"], list(CELEBAMASK_HQ_19))
        classes = (manifold / "classes.txt").read_text(encoding="utf-8").split()
        self.assertEqual(classes, list(CELEBAMASK_HQ_19))

    def _check_policy(self, manifold: Path) -> None:
        """The face parsing policy keeps the three photos and splits them by file id."""
        rule = SampleRule(FACE_PARSING, 0.34, _LAYOUTS)
        after = apply_sample_policy(plan_sample_policy(manifold, rule))
        self.assertFalse(after.changed, after.report())
        keys = {name.split(".")[0] for name in _members(manifold)}
        self.assertEqual(keys, {_key(i) for i in (1, 2, 3)})
        val = list((manifold / "shards/val").glob("shard-*"))
        self.assertEqual(len(val), 1)

    def test_invalid_mask_blocks_the_apply(self) -> None:
        """A class id of 19 is a problem; apply refuses and nothing changes."""
        with tempfile.TemporaryDirectory() as tmp:
            bad = {0: _png(_mask(0)), 1: _png(np.full((_SIZE, _SIZE), 19, np.uint8)),
                   2: _png(_mask(2))}
            manifold, root = _manifold(Path(tmp), bad)
            before = _members(manifold)
            service = _service(root)
            plan = service.plan(manifold)
            self.assertEqual(plan.statuses()[INVALID_MASK], 1)
            with self.assertRaises(SourceLabelError):
                service.apply(plan)
            self.assertEqual(_members(manifold), before)

    def test_size_mismatch_is_invalid(self) -> None:
        """A mask that is not the photo's size is a problem."""
        with tempfile.TemporaryDirectory() as tmp:
            masks = {0: _png(_mask(0, size=8)), 1: _png(_mask(1)), 2: _png(_mask(2))}
            manifold, root = _manifold(Path(tmp), masks)
            self.assertEqual(_service(root).plan(manifold).statuses()[INVALID_MASK], 1)


if __name__ == "__main__":
    unittest.main()
