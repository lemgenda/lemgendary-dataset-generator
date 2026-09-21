"""
WebDataset TAR writer.

Single responsibility: write Samples to WebDataset `.tar` shards. Used by
diffusion manifolds where captions are bundled as `<key>.txt` alongside
`<key>.jpg`.

The `ShardWriter` class moved here from compiler_core.py in Phase 4; the
old location re-exports it for backward compatibility.

Phase 4 of the 2026 modernization roadmap.
"""

from __future__ import annotations

import io
from pathlib import Path
import sys
from typing import Any

import webdataset as wds
from webdataset.writer import ShardWriter as WdsShardWriter

from .base import Sample


class WebDatasetWriter:
    """WebDataset .tar shard writer, driven by the Sample stream."""

    def __init__(self) -> None:
        self._sink: Any = None
        self._out_dir: Path | None = None

    def open(self, output_root: Path, policy: Any) -> None:
        self._out_dir = Path(output_root) / "shards"
        self._out_dir.mkdir(parents=True, exist_ok=True)
        max_size = int(getattr(policy, "wds_shard_size_bytes", 1_000_000_000))
        shard_path = (self._out_dir / "shard-%05d.tar").resolve()
        pattern = f"file:{shard_path}" if sys.platform == "win32" else str(shard_path)
        self._sink = WdsShardWriter(
            pattern,
            maxsize=max_size,
        )

    def write(self, sample: Sample) -> None:
        if self._sink is None:
            return
        img_key = sample.image_format.lower() if sample.image_format else "webp"
        row: dict[str, Any] = {
            "__key__": sample.name,
            img_key: sample.image_bytes,
        }
        if sample.label:
            row["txt"] = sample.label.encode("utf-8")
        if sample.metadata:
            import json
            row["json"] = json.dumps(sample.metadata).encode("utf-8")
        self._sink.write(row)

    def close(self) -> None:
        if self._sink is not None:
            self._sink.close()
            self._sink = None


class ShardWriter:
    """Legacy compatibility shim.

    Pre-Phase-4 diffusion manifolds used this class directly. It is retained
    here so existing imports keep working. New code should use
    WebDatasetWriter.
    """

    def __init__(
        self,
        output_dir: str | Path,
        prefix: str = "data",
        max_size: float = 1e9,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shard_path = (self.output_dir / f"{prefix}-%05d.tar").resolve()
        pattern = f"file:{shard_path}" if sys.platform == "win32" else str(shard_path)
        self.sink: Any = WdsShardWriter(
            pattern,
            maxsize=int(max_size),
        )

    def write(self, name: str, img_bytes: bytes, caption: str) -> None:
        self.sink.write({
            "__key__": name,
            "jpg": img_bytes,
            "txt": caption,
        })

    def close(self) -> None:
        self.sink.close()