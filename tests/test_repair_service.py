"""Shard audit, shard rewrite, metadata resync and the `repair` CLI on synthetic manifolds."""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

import yaml
from typer.testing import CliRunner

from cli import app
from services.repair_service import BACKUP_DIR_NAME, RepairService
from tests.tar_fixtures import write_tar
from tools.repair.metadata_fix import PhysicalSamples, plan_metadata_fix
from tools.repair.shard_fix import ShardFixError, apply_shard_fix, plan_shard_fix
from utils.shard_audit import (
    audit_shard,
    audit_shard_set,
    shard_set_problems,
    split_member_name,
)

_STAMP = "20260101T000000Z"


def _member_names(path: Path) -> list[str]:
    with tarfile.open(path, "r:") as tf:
        return tf.getnames()


def _make_manifold(root: Path, name: str = "DemoManifold") -> Path:
    """Train shard: s0 ok, s1 zero-byte target, s2 scattered. Val shard: v0 ok."""
    manifold = root / name
    write_tar(manifold / "shards" / "train" / "shard-00000.data", [
        ("s0.webp", b"img0"), ("s0.target.webp", b"tgt0"),
        ("s1.webp", b"img1"), ("s1.target.webp", b""),
        ("s2.webp", b"img2"), ("s0.json", b"{}"), ("s2.target.webp", b"tgt2"),
    ])
    write_tar(manifold / "shards" / "val" / "shard-00000.data", [
        ("v0.webp", b"imgv"), ("v0.target.webp", b"tgtv"),
    ])
    info = {"name": name, "count": 9, "total_samples": 9, "total_shards": 1,
            "path": "C:\\Users\\x\\AppData\\Local\\Temp\\.staging_build_demo\\" + name}
    (manifold / "dataset_info.yaml").write_text(yaml.safe_dump(info), encoding="utf-8")
    index = [
        {"id": 0, "name": "s0", "source": "src", "split": "train"},
        {"id": 1, "name": "s1", "source": "src", "split": "train"},
        {"id": 2, "name": "s2", "source": "", "split": "val"},
        {"id": 3, "name": "gone", "source": "src", "split": "skipped"},
        {"id": 4, "name": "s0", "source": "src", "split": "train"},
    ]
    (manifold / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return manifold


class _TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        """Fresh temporary directory per test."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)


class ShardAuditTests(_TempDirCase):
    """The header-only auditor finds empty members and scattered groups."""

    def test_split_member_name_uses_first_dot(self) -> None:
        """Split member name uses first dot."""
        self.assertEqual(split_member_name("d/a.target.webp"), ("d/a", ".target.webp"))
        self.assertEqual(split_member_name("a.webp"), ("a", ".webp"))

    def test_audit_reports_zero_byte_and_scattered(self) -> None:
        """Audit reports zero byte and scattered."""
        manifold = _make_manifold(self.root)
        audit = audit_shard(manifold / "shards" / "train" / "shard-00000.data")
        self.assertEqual(audit.samples, 3)
        self.assertEqual(audit.zero_byte_keys, {"s1"})
        self.assertEqual(audit.scattered, ["s0", "s2"])
        self.assertTrue(audit.needs_rewrite)
        self.assertEqual(audit.incomplete((".webp", ".target.webp")), [])

    def test_set_problems_and_split_counts(self) -> None:
        """Set problems and split counts."""
        manifold = _make_manifold(self.root)
        audit = audit_shard_set(manifold / "shards")
        self.assertEqual(audit.split_counts, {"train": 3, "val": 1})
        problems = shard_set_problems(audit, expected_total=9)
        self.assertTrue(any("zero-byte" in p for p in problems))
        self.assertTrue(any("not contiguous" in p for p in problems))
        self.assertTrue(any("expected 9" in p for p in problems))

    def test_corrupt_tar_is_reported_not_raised(self) -> None:
        """Corrupt tar is reported not raised."""
        bad = self.root / "shards" / "bad.data"
        bad.parent.mkdir(parents=True)
        bad.write_bytes(b"not a tar file at all" * 10)
        audit = audit_shard(bad)
        self.assertIsNotNone(audit.error)
        self.assertIsNone(plan_shard_fix(audit))


class ShardFixTests(_TempDirCase):
    """Rewrites drop whole samples, regroup members and keep a backup."""

    def test_apply_drops_sample_and_regroups(self) -> None:
        """Apply drops sample and regroups."""
        shard = _make_manifold(self.root) / "shards" / "train" / "shard-00000.data"
        fix = plan_shard_fix(audit_shard(shard))
        assert fix is not None
        self.assertEqual(fix.samples_after, 2)
        backup = self.root / "backup" / shard.name
        apply_shard_fix(fix, backup)
        self.assertEqual(_member_names(shard),
                         ["s0.webp", "s0.target.webp", "s0.json", "s2.webp", "s2.target.webp"])
        self.assertEqual(len(_member_names(backup)), 7)
        self.assertFalse(audit_shard(shard).needs_rewrite)
        self.assertEqual(list(shard.parent.glob("*.repair-tmp")), [])

    def test_failed_verification_leaves_original(self) -> None:
        """Failed verification leaves original."""
        shard = _make_manifold(self.root) / "shards" / "train" / "shard-00000.data"
        fix = plan_shard_fix(audit_shard(shard))
        assert fix is not None
        wrong = type(fix)(fix.shard, fix.drop_keys, fix.regroup, fix.samples_before + 5)
        before = shard.read_bytes()
        with self.assertRaises(ShardFixError):
            apply_shard_fix(wrong, self.root / "backup" / shard.name)
        self.assertEqual(shard.read_bytes(), before)
        self.assertEqual(list(shard.parent.glob("*.repair-tmp")), [])


class MetadataFixTests(_TempDirCase):
    """dataset_info.yaml and index.json follow the samples on disk."""

    def test_plan_resyncs_counts_path_and_index(self) -> None:
        """Plan resyncs counts path and index."""
        manifold = _make_manifold(self.root)
        physical = PhysicalSamples({"train": 2, "val": 1},
                                   {"train": {"s0", "s2"}, "val": {"v0"}}, 2)
        fix = plan_metadata_fix(manifold, physical)
        self.assertEqual(fix.info_changes["count"], (9, 3))
        self.assertEqual(fix.info_changes["total_shards"], (1, 2))
        self.assertEqual(Path(fix.info_changes["path"][1]).name, "DemoManifold")
        self.assertNotIn(".staging", fix.info_changes["path"][1])
        self.assertEqual((fix.index.removed, fix.index.added, fix.index.resplit,
                          fix.index.source_filled), (3, 1, 1, 1))
        assert fix.new_index is not None
        self.assertEqual({e["name"]: e["split"] for e in fix.new_index},
                         {"s0": "train", "s2": "train", "v0": "val"})

    def test_unmatched_names_keep_entries(self) -> None:
        """Unmatched names keep entries."""
        manifold = _make_manifold(self.root)
        physical = PhysicalSamples({"train": 2}, {"train": {"x1", "x2"}}, 1)
        fix = plan_metadata_fix(manifold, physical)
        self.assertEqual(fix.index.removed, 0)
        self.assertTrue(any("fewer than half" in n for n in fix.notes))

    def test_llm_manifold_fills_source_only(self) -> None:
        """Llm manifold fills source only."""
        manifold = self.root / "AIHelper"
        (manifold / "data").mkdir(parents=True)
        (manifold / "data" / "ai_helper_train.jsonl").write_text("{}\n{}\n", encoding="utf-8")
        (manifold / "index.json").write_text(json.dumps([{"name": "a", "split": "train"}]),
                                             encoding="utf-8")
        plan = RepairService(_STAMP).plan(manifold)
        assert plan.metadata is not None
        self.assertEqual(plan.metadata.index.source_filled, 1)
        self.assertEqual(plan.metadata.index.removed, 0)
        self.assertTrue(any("cannot be matched" in n for n in plan.metadata.notes))


class RepairServiceTests(_TempDirCase):
    """End to end: dry run changes nothing, apply repairs and backs up, rerun is a no-op."""

    def test_plan_is_read_only(self) -> None:
        """Plan is read only."""
        manifold = _make_manifold(self.root)
        before = {p: p.read_bytes() for p in manifold.rglob("*") if p.is_file()}
        plan = RepairService(_STAMP).plan(manifold)
        self.assertTrue(plan.changed)
        self.assertTrue(any("drop 1 samples" in line for line in plan.report()))
        self.assertEqual({p: p.read_bytes() for p in manifold.rglob("*") if p.is_file()}, before)

    def test_apply_repairs_backs_up_and_is_idempotent(self) -> None:
        """Apply repairs backs up and is idempotent."""
        manifold = _make_manifold(self.root)
        service = RepairService(_STAMP)
        remaining = service.apply(service.plan(manifold))
        self.assertFalse(remaining.changed, remaining.report())
        info = yaml.safe_load((manifold / "dataset_info.yaml").read_text(encoding="utf-8"))
        self.assertEqual((info["count"], info["total_samples"], info["total_shards"]), (3, 3, 2))
        index = json.loads((manifold / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(e["name"] for e in index), ["s0", "s2", "v0"])
        self.assertTrue(all(e["source"] for e in index))
        backup = self.root / BACKUP_DIR_NAME / _STAMP / manifold.name
        self.assertTrue((backup / "shards" / "train" / "shard-00000.data").is_file())
        self.assertTrue((backup / "dataset_info.yaml").is_file())
        self.assertTrue((backup / "index.json").is_file())
        second = RepairService("20260102T000000Z")
        self.assertFalse(second.plan(manifold).changed)
        self.assertFalse((self.root / BACKUP_DIR_NAME / "20260102T000000Z").exists())


class RepairCliTests(_TempDirCase):
    """`repair manifold` and `repair all` are dry runs unless --apply is given."""

    def test_manifold_dry_run_then_apply(self) -> None:
        """Manifold dry run then apply."""
        manifold = _make_manifold(self.root)
        runner = CliRunner()
        dry = runner.invoke(app, ["repair", "manifold", "--manifold", str(manifold)])
        self.assertEqual(dry.exit_code, 0, dry.output)
        self.assertIn("Dry run", dry.output)
        self.assertFalse((self.root / BACKUP_DIR_NAME).exists())
        done = runner.invoke(app, ["repair", "manifold", "--manifold", str(manifold), "--apply"])
        self.assertEqual(done.exit_code, 0, done.output)
        self.assertTrue((self.root / BACKUP_DIR_NAME).is_dir())

    def test_all_skips_backup_folder(self) -> None:
        """All skips backup folder."""
        _make_manifold(self.root, "A")
        _make_manifold(self.root, "B")
        runner = CliRunner()
        done = runner.invoke(app, ["repair", "all", "--root", str(self.root), "--apply"])
        self.assertEqual(done.exit_code, 0, done.output)
        again = runner.invoke(app, ["repair", "all", "--root", str(self.root)])
        self.assertEqual(again.exit_code, 0, again.output)
        self.assertEqual(again.output.count("nothing to repair"), 2)
        self.assertNotIn(BACKUP_DIR_NAME, again.output)


if __name__ == "__main__":
    unittest.main()
