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

import json
import sys
from pathlib import Path
from typing import Any

from webdataset.writer import ShardWriter as WdsShardWriter

from .base import Sample, Writer

_DEFAULT_SPLIT = "train"


def _wds_pattern(shard_path: Path) -> str:
    """The webdataset ShardWriter pattern for ``shard_path`` (``file:`` on Windows)."""
    resolved = shard_path.resolve()
    return f"file:{resolved}" if sys.platform == "win32" else str(resolved)


class WebDatasetWriter(Writer):
    """WebDataset .tar shard writer, driven by the Sample stream.

    Shards are partitioned by split (``shards/<split>/shard-%05d.<ext>``),
    the same layout the loose and zip converters write, so the split of a
    sample is never lost.
    """

    def __init__(self) -> None:
        self._sinks: dict[str, Any] = {}
        self._out_dir: Path | None = None
        self._max_size = 1_000_000_000
        self._shard_ext = "data"

    def open(self, output_root: Path, policy: Any) -> None:
        self._out_dir = Path(output_root) / "shards"
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._max_size = int(getattr(policy, "wds_shard_size_bytes", 1_000_000_000))
        self._shard_ext = str(getattr(policy, "shard_extension", "data")).removeprefix(".")

    def _sink_for(self, split: str) -> Any:
        """The shard sink of ``split``, created on first use."""
        sink = self._sinks.get(split)
        if sink is None and self._out_dir is not None:
            split_dir = self._out_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            pattern = _wds_pattern(split_dir / f"shard-%05d.{self._shard_ext}")
            sink = WdsShardWriter(pattern, maxsize=self._max_size)
            self._sinks[split] = sink
        return sink

    def write(self, sample: Sample) -> None:
        sink = self._sink_for(sample.split or _DEFAULT_SPLIT)
        if sink is None:
            return
        img_key = sample.image_format.lower() if sample.image_format else "webp"
        row: dict[str, Any] = {
            "__key__": sample.name,
            img_key: sample.image_bytes,
        }
        if sample.target_bytes:
            row[f"target.{img_key}"] = sample.target_bytes
        if sample.mask_bytes:
            row["mask.webp"] = sample.mask_bytes
        if sample.label:
            row["txt"] = sample.label.encode("utf-8")
        if sample.metadata:
            row["json"] = json.dumps(sample.metadata).encode("utf-8")
        sink.write(row)

    def close(self) -> None:
        for sink in self._sinks.values():
            sink.close()
        self._sinks.clear()


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
        extension: str = "data",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shard_ext = extension.removeprefix(".")
        self.sink: Any = WdsShardWriter(
            _wds_pattern(self.output_dir / f"{prefix}-%05d.{shard_ext}"),
            maxsize=int(max_size),
        )

    def write(self, name: str, img_bytes: bytes, caption: str) -> None:
        """Write one image with its caption as a WebDataset sample."""
        self.sink.write({
            "__key__": name,
            "jpg": img_bytes,
            "txt": caption,
        })

    def close(self) -> None:
        """Close the underlying tar sink."""
        self.sink.close()
