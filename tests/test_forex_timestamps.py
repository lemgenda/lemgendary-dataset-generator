"""Forex timestamps are Unix epoch seconds everywhere: data, schema and metadata (R2-D4)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from core.compile_forex import _write_dataset_info
from core.docs.metadata import FOREX_COLUMN_FIELDS
from forex.columns import COLUMN_DESCRIPTIONS, TIMESTAMP_UNIT, column_fields
from forex.schema import PARQUET_SCHEMA
from forex.shard_table import ShardLabels, build_shard_table, to_epoch_seconds

_SECONDS = 1_700_000_000


class ForexTimestampTests(unittest.TestCase):
    """Seconds in the written column, the schema metadata and dataset_info.yaml."""

    def test_finer_units_scale_to_seconds(self) -> None:
        """Seconds stay; ms, us and ns are divided down."""
        for scale in (1, 10**3, 10**6, 10**9):
            out = to_epoch_seconds(np.array([_SECONDS * scale, 0]))
            self.assertEqual(out.tolist(), [_SECONDS, 0], scale)
        self.assertEqual(to_epoch_seconds(np.array([], dtype=np.int64)).size, 0)

    def test_shard_table_stores_seconds(self) -> None:
        """A shard built from millisecond stamps holds seconds."""
        features = np.zeros((2, 4, 14), dtype=np.float32)
        labels = ShardLabels(
            y_dir=np.array([0, 2]), y_mag=np.ones((2, 2), dtype=np.float32),
            timestamps=np.array([_SECONDS * 1000, (_SECONDS + 60) * 1000]),
        )
        table = build_shard_table("EURUSD", 1, features, labels)
        self.assertEqual(table.column("timestamp").to_pylist(), [_SECONDS, _SECONDS + 60])

    def test_descriptions_and_metadata_say_seconds(self) -> None:
        """Column description, schema metadata and Kaggle fields agree on seconds."""
        self.assertEqual(TIMESTAMP_UNIT, "seconds")
        self.assertIn("seconds", COLUMN_DESCRIPTIONS["timestamp"])
        self.assertNotIn("illisecond", COLUMN_DESCRIPTIONS["timestamp"])
        meta = PARQUET_SCHEMA.metadata
        self.assertEqual(meta[b"timestamp_unit"], b"seconds")
        field_meta = PARQUET_SCHEMA.field("timestamp").metadata
        self.assertEqual(field_meta[b"description"].decode(), COLUMN_DESCRIPTIONS["timestamp"])
        self.assertEqual(FOREX_COLUMN_FIELDS, column_fields())

    def test_dataset_info_records_unit(self) -> None:
        """dataset_info.yaml of a forex manifold records ``timestamp_unit: seconds``."""
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        _write_dataset_info({}, "ForexDemo", {"pairs": ["EURUSD"]}, root)
        info = yaml.safe_load((root / "dataset_info.yaml").read_text(encoding="utf-8"))
        self.assertEqual(info["timestamp_unit"], "seconds")


if __name__ == "__main__":
    unittest.main()
