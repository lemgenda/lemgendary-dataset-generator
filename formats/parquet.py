"""
Parquet writer.

Single responsibility: write Samples to a columnar Parquet file with Zstd
compression. Used by tabular manifolds (Forex); for vision manifolds the
Parquet output stores image bytes in a binary column.

Requires: pyarrow (already a hard dependency for other reasons).

Phase 4 of the 2026 modernization roadmap.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from .base import Sample


class ParquetWriter:
    """Parquet writer, one row per Sample, Zstd-compressed."""

    def __init__(self) -> None:
        self._out_path: Path | None = None
        self._rows: list[dict[str, Any]] = []
        self._flush_threshold = 50_000

    def open(self, output_root: Path, policy: Any) -> None:
        # Runtime import via importlib so Pyrefly doesn't statically require
        # this optional dependency at lint time. pyarrow is already a hard
        # dependency, but keeping the pattern uniform with mds/litdata lets
        # the same writer factory logic handle all optional containers.
        try:
            importlib.import_module("pyarrow")
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for --also-format parquet. "
                "Install it with: pip install pyarrow"
            ) from e
        self._out_path = Path(output_root) / "manifold.parquet"
        self._out_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, sample: Sample) -> None:
        self._rows.append({
            "name": sample.name,
            "task": sample.task,
            "split": sample.split,
            "image": sample.image_bytes,
            "image_format": sample.image_format,
            "target": sample.target_bytes,
            "mask": sample.mask_bytes,
            "label": sample.label,
            "metadata_json": json.dumps(sample.metadata or {}),
        })
        if len(self._rows) >= self._flush_threshold:
            self._flush()

    def close(self) -> None:
        self._flush()
        self._out_path = None

    def _flush(self) -> None:
        if not self._rows or self._out_path is None:
            return
        pa = importlib.import_module("pyarrow")
        pq = importlib.import_module("pyarrow.parquet")

        # Image / target / mask are stored as binary columns.
        # metadata is serialized to JSON text so its schema is fixed.
        schema = pa.schema([
            ("name", pa.string()),
            ("task", pa.string()),
            ("split", pa.string()),
            ("image", pa.binary()),
            ("image_format", pa.string()),
            ("target", pa.binary()),
            ("mask", pa.binary()),
            ("label", pa.string()),
            ("metadata_json", pa.string()),
        ])

        # Append to any existing Parquet file rather than overwriting.
        if self._out_path.exists():
            existing = pq.read_table(str(self._out_path))
            existing_rows: list[dict[str, Any]] = existing.to_pylist()
            existing_rows.extend(self._rows)
            rows_to_write = existing_rows
        else:
            rows_to_write = self._rows

        columns: dict[str, list[Any]] = {name: [] for name in schema.names}
        for row in rows_to_write:
            for name in schema.names:
                columns[name].append(row.get(name))
        table = pa.table(columns, schema=schema)
        pq.write_table(table, str(self._out_path), compression="zstd", use_dictionary=True)
        self._rows = []