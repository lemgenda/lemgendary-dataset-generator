"""Cross-repo check: the training suite's reader accepts the compiler's shard_offsets.json.

The compiler writes the file from the shared format spec (:mod:`utils.shard_offsets`)
without importing the training suite. This test loads the file with the suite's own
reader (``training/data/containers/shard_tables.py``) when the suite is checked out as
the sibling folder ``lemgendary-training-suite``, and is skipped otherwise.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from tests.tar_fixtures import write_tar
from utils.shard_offsets import OFFSETS_FILENAME, OFFSETS_VERSION, write_shard_offsets

SUITE_ROOT = Path(__file__).resolve().parents[2] / "lemgendary-training-suite"
_READER_MODULE = "training.data.containers.shard_tables"
_MEMBERS = [("k1.webp", b"img-1"), ("k1.json", b"{}"), ("k2.webp", b"img-22"),
            ("k2.mask.webp", b"mask"), ("k2.json", b'{"a": 1}')]


def _suite_reader() -> tuple[ModuleType | None, str]:
    """The suite's shard_tables module, or None and the reason it cannot be loaded."""
    if not (SUITE_ROOT / "training" / "data" / "containers" / "shard_tables.py").is_file():
        return None, f"training suite not checked out next to this repo ({SUITE_ROOT.name})"
    # Appended, not prepended: the suite's top-level ``cli``, ``tools`` and ``models``
    # must not shadow this repo's modules of the same names.
    if str(SUITE_ROOT) not in sys.path:
        sys.path.append(str(SUITE_ROOT))
    try:
        return importlib.import_module(_READER_MODULE), ""
    except ImportError as exc:
        return None, f"training suite reader not importable here: {exc}"


class TrainingSuiteReaderCompatTests(unittest.TestCase):
    """The suite's ShardTableStore uses the compiler's index as written."""

    reader: ModuleType

    @classmethod
    def setUpClass(cls) -> None:
        """Load the suite's reader or skip the class with the reason."""
        reader, reason = _suite_reader()
        if reader is None:
            raise unittest.SkipTest(reason)
        cls.reader = reader

    def setUp(self) -> None:
        """Two indexed shards in a train folder and a separate cache folder."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.shard_dir = self.root / "shards" / "train"
        self.shards = [write_tar(self.shard_dir / f"shard-0000{i}.data", _MEMBERS)
                       for i in range(2)]
        write_shard_offsets(self.root / "shards")
        self.cache = self.root / "cache"
        self.cache.mkdir()

    def _tables(self) -> list[list[tuple[str, int, int]] | None]:
        store = self.reader.ShardTableStore(self.shard_dir, self.cache)
        tables = [store.members(shard) for shard in self.shards]
        return [None if t is None else [tuple(ref) for ref in t] for t in tables]

    def test_format_constants_match(self) -> None:
        """File name and layout version are the suite's."""
        self.assertEqual(OFFSETS_FILENAME, self.reader.SHIPPED_INDEX_FILENAME)
        self.assertEqual(OFFSETS_VERSION, self.reader.STORE_VERSION)

    def test_reader_accepts_the_written_tables(self) -> None:
        """Every shard's table comes from the file and matches a fresh suite scan."""
        scan = importlib.import_module("training.data.containers.tar_offsets").scan_tar_members
        expected = [[tuple(ref) for ref in scan(shard)] for shard in self.shards]
        self.assertEqual(self._tables(), expected)

    def test_reader_accepts_the_tables_after_an_upload(self) -> None:
        """A new modification time (an upload or a copy) passes the suite's header check."""
        for shard in self.shards:
            stat = shard.stat()
            os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 7_000_000_000))
        self.assertTrue(all(table is not None for table in self._tables()))


if __name__ == "__main__":
    unittest.main()
