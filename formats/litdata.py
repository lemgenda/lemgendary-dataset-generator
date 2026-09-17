"""
PyTorch LitData writer.

Single responsibility: serialize Samples into LitData's streaming format,
which handles variable-shape inputs (bboxes, pose keypoints) more gracefully
than MDS.

Requires: `litdata`. Optional dependency.

Because LitData has no documented single-sample append API that is stable
across versions, this writer buffers per-shard into intermediate pickle
files, then flushes via the public `optimize()` entry point in `close()`.

Phase 4 of the 2026 modernization roadmap.
"""

from __future__ import annotations

import importlib
import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any, Iterator

from .base import Sample

logger = logging.getLogger(__name__)


# Samples per intermediate shard file. Larger = fewer temp files, more RAM.
_SHARD_SIZE = 4096


class LitDataWriter:
    """LitData writer, sharded via temporary pickle files."""

    def __init__(self) -> None:
        self._out_dir: Path | None = None
        self._tmp_dir: Path | None = None
        self._buffer: list[dict[str, Any]] = []
        self._shard_idx = 0
        self._total = 0

    def open(self, output_root: Path, policy: Any) -> None:
        # Runtime import via importlib so Pyrefly doesn't statically require
        # this optional dependency at lint time.
        try:
            importlib.import_module("litdata")
        except ImportError as e:
            raise ImportError(
                "litdata is required for --also-format litdata. "
                "Install it with: pip install litdata"
            ) from e

        self._out_dir = Path(output_root) / "litdata"
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="lgd_litdata_"))

    def write(self, sample: Sample) -> None:
        entry = {
            "name": sample.name,
            "task": sample.task,
            "split": sample.split,
            "image_bytes": sample.image_bytes,
            "image_format": sample.image_format,
            "target_bytes": sample.target_bytes,
            "mask_bytes": sample.mask_bytes,
            "label": sample.label,
            "metadata": sample.metadata or {},
        }
        self._buffer.append(entry)
        self._total += 1
        if len(self._buffer) >= _SHARD_SIZE:
            self._flush_shard()

    def close(self) -> None:
        if self._buffer:
            self._flush_shard()

        # Convert the accumulated shards into LitData's final format.
        if self._tmp_dir is not None and self._out_dir is not None:
            shards = sorted(self._tmp_dir.glob("shard_*.pkl"))
            if shards:
                try:
                    litdata = importlib.import_module("litdata")
                    optimize = getattr(litdata, "optimize")
                    inputs = [(str(s),) for s in shards]
                    optimize(
                        fn=_read_shard,
                        inputs=inputs,
                        output_dir=str(self._out_dir),
                        num_workers=1,
                        chunk_bytes="64MB",
                    )
                except ImportError as e:
                    print(f"[WARN] litdata optimize() unavailable: {e}")

        # Cleanup temporary shards.
        if self._tmp_dir is not None and self._tmp_dir.exists():
            for p in self._tmp_dir.glob("shard_*.pkl"):
                try:
                    p.unlink()
                except OSError as exc:
                    logger.debug("Failed unlinking temp shard %s: %s", p, exc)
            try:
                self._tmp_dir.rmdir()
            except OSError as exc:
                logger.debug("Failed removing temp directory %s: %s", self._tmp_dir, exc)

        self._buffer = []
        self._tmp_dir = None
        self._out_dir = None

    def _flush_shard(self) -> None:
        if self._tmp_dir is None:
            return
        shard_path = self._tmp_dir / f"shard_{self._shard_idx:05d}.pkl"
        with open(shard_path, "wb") as f:
            pickle.dump(self._buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._shard_idx += 1
        self._buffer = []


def _read_shard(shard_path: str) -> Iterator[dict[str, Any]]:
    """LitData optimize() callable: load one pickle shard and yield each entry."""
    with open(shard_path, "rb") as f:
        entries = pickle.load(f)
    for entry in entries:
        yield entry