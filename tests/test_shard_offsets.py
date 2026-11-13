"""shard_offsets.json: written by every shard writer, checked by the shard verifier."""

from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from api import manifold_scan
from cli import app
from formats.base import Sample
from formats.webdataset import ShardWriter, WebDatasetWriter
from services.repair_service import RepairService
from services.shard_offsets_service import ShardOffsetsService
from tests.tar_fixtures import write_tar
from utils.shard_audit import verify_shard_set
from utils.shard_offsets import (
    OFFSETS_FILENAME,
    OFFSETS_VERSION,
    offsets_problems,
    write_shard_offsets,
)

_MEMBERS = [("a.webp", b"image-a"), ("a.json", b"{}"), ("b.webp", b"image-bb"),
            ("b.json", b'{"k": 1}')]


def _load(shard_dir: Path) -> dict:
    return json.loads((shard_dir / OFFSETS_FILENAME).read_text(encoding="utf-8"))


class ShardOffsetsWriterTests(unittest.TestCase):
    """The index records every member's payload position in the suite's layout."""

    def setUp(self) -> None:
        """A manifold with one train shard."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.shards = self.root / "shards"
        self.train = self.shards / "train"
        self.shard = write_tar(self.train / "shard-00000.data", _MEMBERS)

    def test_layout_and_offsets(self) -> None:
        """Version 2, one entry per shard, and every offset reads back its payload."""
        results = write_shard_offsets(self.shards)
        self.assertEqual([r.shard_dir for r in results], [self.train])
        data = _load(self.train)
        self.assertEqual(data["version"], OFFSETS_VERSION)
        entry = data["shards"]["shard-00000.data"]
        stat = self.shard.stat()
        self.assertEqual((entry["size"], entry["mtime_ns"]), (stat.st_size, stat.st_mtime_ns))
        self.assertEqual(entry["names"], [name for name, _ in _MEMBERS])
        raw = self.shard.read_bytes()
        for offset, size, (_, payload) in zip(entry["offsets"], entry["sizes"], _MEMBERS,
                                              strict=True):
            self.assertEqual(raw[offset:offset + size], payload)
        self.assertEqual(offsets_problems(self.shards), [])

    def test_unchanged_shards_are_not_rescanned(self) -> None:
        """A second write reuses the entry; a rewritten shard is rescanned."""
        write_shard_offsets(self.shards)
        self.assertEqual(write_shard_offsets(self.shards)[0].scanned, 0)
        write_tar(self.shard, [("c.webp", b"other payload")])
        result = write_shard_offsets(self.shards)[0]
        self.assertEqual((result.scanned, result.members), (1, 1))
        self.assertEqual(_load(self.train)["shards"]["shard-00000.data"]["names"], ["c.webp"])

    def test_compressed_shard_gets_no_entry(self) -> None:
        """A gzip shard cannot be read by offset: no entry, and no file for such a folder."""
        gz = self.shards / "val" / "shard-00000.data"
        gz.parent.mkdir(parents=True)
        with tarfile.open(gz, "w:gz") as tf:
            info = tarfile.TarInfo("a.webp")
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))
        write_shard_offsets(self.shards)
        self.assertFalse((gz.parent / OFFSETS_FILENAME).exists())
        self.assertEqual(offsets_problems(self.shards), [])

    def test_index_of_an_emptied_folder_is_removed(self) -> None:
        """A folder that lost its shards loses its index."""
        write_shard_offsets(self.shards)
        self.shard.unlink()
        write_shard_offsets(self.shards)
        self.assertFalse((self.train / OFFSETS_FILENAME).exists())


class ShardOffsetsVerifierTests(unittest.TestCase):
    """A present index must still describe the shards; a missing one is fine."""

    def setUp(self) -> None:
        """A manifold with one indexed train shard."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.shards = self.root / "shards"
        self.train = self.shards / "train"
        self.shard = write_tar(self.train / "shard-00000.data", _MEMBERS)
        write_shard_offsets(self.shards)

    def _quiet_verify(self) -> tuple[bool, str]:
        with redirect_stdout(io.StringIO()) as out:
            ok = verify_shard_set(self.shards, "T")
        return ok, out.getvalue()

    def test_missing_index_is_not_a_problem(self) -> None:
        """Without the file there is nothing to check."""
        (self.train / OFFSETS_FILENAME).unlink()
        self.assertEqual(offsets_problems(self.shards), [])
        self.assertTrue(self._quiet_verify()[0])

    def test_new_mtime_with_same_layout_is_accepted(self) -> None:
        """An upload changes the mtime only; the header spot check accepts the entry."""
        stat = self.shard.stat()
        os.utime(self.shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
        self.assertEqual(offsets_problems(self.shards), [])

    def test_rewritten_shard_is_stale(self) -> None:
        """A shard rewritten after indexing fails the verifier."""
        write_tar(self.shard, [("z.webp", b"a different, longer payload than before")])
        ok, out = self._quiet_verify()
        self.assertFalse(ok)
        self.assertIn(OFFSETS_FILENAME, out)

    def test_same_size_other_layout_is_stale(self) -> None:
        """Same size, new mtime and other member names: the headers do not match."""
        indexed = self.shard.stat()
        write_tar(self.shard, [(name.replace("a", "q"), data) for name, data in _MEMBERS])
        os.utime(self.shard, ns=(indexed.st_atime_ns, indexed.st_mtime_ns + 5_000_000_000))
        problems = offsets_problems(self.shards)
        self.assertEqual(len(problems), 1)
        self.assertIn("headers do not match", problems[0])

    def test_unindexed_shard_and_orphan_entry(self) -> None:
        """A plain shard without an entry and an entry without a shard are both stale."""
        write_tar(self.train / "shard-00001.data", _MEMBERS)
        self.shard.rename(self.train / "shard-00009.data")
        problems = " ".join(offsets_problems(self.shards))
        self.assertIn("shard-00001.data: no entry", problems)
        self.assertIn("entries without a shard", problems)

    def test_other_version_is_stale(self) -> None:
        """A file of another layout version is reported."""
        (self.train / OFFSETS_FILENAME).write_text('{"version": 1, "shards": {}}',
                                                   encoding="utf-8")
        self.assertIn("layout version", offsets_problems(self.shards)[0])

    def test_member_count_must_match_the_audit(self) -> None:
        """A count from the shard audit that differs from the entry is reported."""
        problems = offsets_problems(self.shards, {self.shard: len(_MEMBERS) + 1})
        self.assertIn("members indexed", problems[0])

    def test_repair_manifold_rewrites_a_stale_index(self) -> None:
        """repair manifold plans the stale index and --apply refreshes it."""
        (self.train / OFFSETS_FILENAME).write_text('{"version": 2, "shards": {}}',
                                                   encoding="utf-8")
        service = RepairService()
        plan = service.plan(self.root)
        self.assertTrue(plan.offsets_problems)
        self.assertTrue(any("shard index" in line for line in plan.report()))
        remaining = service.apply(plan)
        self.assertEqual(remaining.offsets_problems, [])


class ShardWritersIndexTests(unittest.TestCase):
    """The compile writers leave an index next to the shards they close."""

    def test_webdataset_writer_indexes_each_split(self) -> None:
        """WebDatasetWriter.close writes one index per split folder."""
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            writer = WebDatasetWriter()
            writer.open(out_root, SimpleNamespace(wds_shard_size_bytes=10_000_000))
            for split in ("train", "val"):
                writer.write(Sample(name=f"s_{split}", task="quality", split=split,
                                    image_bytes=b"img", image_format="webp"))
            writer.close()
            for split in ("train", "val"):
                data = _load(out_root / "shards" / split)
                self.assertEqual(list(data["shards"]), ["shard-00000.data"])
            self.assertEqual(offsets_problems(out_root / "shards"), [])

    def test_legacy_shard_writer_indexes_its_folder(self) -> None:
        """The diffusion ShardWriter shim indexes its output folder on close."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "shards"
            writer = ShardWriter(out)
            writer.write("cap_0", b"jpg-bytes", "a caption")
            writer.close()
            self.assertIn("data-00000.data", _load(out)["shards"])


class ShardOffsetsCommandTests(unittest.TestCase):
    """repair shard-offsets: dry run reports, --apply writes, shards stay untouched."""

    def setUp(self) -> None:
        """A manifold with train and val shards and no index."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.manifold = self.root / "LemGendizedX"
        self.shards = [write_tar(self.manifold / "shards" / split / "shard-00000.data", _MEMBERS)
                       for split in ("train", "val")]

    def test_service_plan_and_apply(self) -> None:
        """The plan says missing; apply writes both and the next plan is current."""
        service = ShardOffsetsService()
        plan = service.plan(self.manifold)
        self.assertTrue(plan.changed)
        self.assertTrue(all("missing" in line for line in plan.report()[1:]))
        run = service.apply(plan)
        self.assertFalse(run.remaining.changed)
        self.assertEqual(len(run.written), 2)
        self.assertGreater(run.bytes_written, 0)
        self.assertTrue(run.report()[-1].startswith("  total:"))

    def test_cli_dry_run_then_apply(self) -> None:
        """The dry run writes nothing; --apply writes the index and leaves the shards as is."""
        before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in self.shards]
        dry = CliRunner().invoke(app, ["repair", "shard-offsets", "--manifold",
                                       str(self.manifold)])
        self.assertEqual(dry.exit_code, 0, dry.output)
        self.assertFalse(any((p.parent / OFFSETS_FILENAME).exists() for p in self.shards))
        done = CliRunner().invoke(app, ["repair", "shard-offsets", "--manifold",
                                        str(self.manifold), "--apply"])
        self.assertEqual(done.exit_code, 0, done.output)
        self.assertTrue(all((p.parent / OFFSETS_FILENAME).is_file() for p in self.shards))
        self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in self.shards])

    def test_gui_scan_does_not_count_the_index_as_a_shard(self) -> None:
        """The catalog's shard count ignores shard_offsets.json."""
        write_shard_offsets(self.manifold / "shards")
        stats = manifold_scan.calculate_dataset_file_stats(self.manifold)
        self.assertEqual(stats.shards_count, 2)


if __name__ == "__main__":
    unittest.main()
