"""
MosaicML Streaming (MDS) writer.

Single responsibility: serialize a stream of Samples into sharded `.mds`
files with Zstd compression.

Requires: `mosaicml-streaming`. Optional dependency — `open()` raises a
clear ImportError if the package is not installed.

Phase 4 of the 2026 modernization roadmap.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from .base import Sample


# Column schema. All vision manifolds share this shape; task-specific
# fields (labels, targets, masks) are present only when the source has
# them, which the writer handles by allowing None.
_COLUMNS: dict[str, str] = {
    "image": "jpeg",       # accepts WebP bytes too; MDS sniffs the format
    "target": "jpeg",
    "mask": "jpeg",
    "label": "str",
    "task": "str",
    "split": "str",
    "metadata": "json",
}


class MDSWriter:
    """MosaicML Streaming writer with Zstd compression."""

    def __init__(self) -> None:
        self._writer: Any = None
        self._out_dir: Path | None = None

    def open(self, output_root: Path, policy: Any) -> None:
        # Runtime import via importlib so Pyrefly doesn't statically require
        # this optional dependency at lint time.
        try:
            streaming = importlib.import_module("streaming")
        except ImportError as e:
            raise ImportError(
                "mosaicml-streaming is required for --also-format mds. "
                "Install it with: pip install mosaicml-streaming"
            ) from e

        mds_writer_cls = getattr(streaming, "MDSWriter")
        self._out_dir = Path(output_root) / "mds"
        self._out_dir.mkdir(parents=True, exist_ok=True)

        size_limit = int(getattr(policy, "mds_shard_size_bytes", 512 * 1024 * 1024))
        self._writer = mds_writer_cls(
            out=str(self._out_dir),
            columns=_COLUMNS,
            compression="zstd",
            size_limit=size_limit,
            hashes=[],
        )

    def write(self, sample: Sample) -> None:
        if self._writer is None:
            return
        row: dict[str, Any] = {
            "image": sample.image_bytes,
            "target": sample.target_bytes,
            "mask": sample.mask_bytes,
            "label": sample.label,
            "task": sample.task,
            "split": sample.split,
            "metadata": json.dumps(sample.metadata or {}),
        }
        self._writer.write(row)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.finish()
            self._writer = None