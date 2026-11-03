"""Job objects passed from the compile planner to the per-sample processors.

``manifold_compile`` builds one ``ImageJob`` per source image and one
``ShardJob`` per Parquet shard and hands them to ``batch_worker`` as
``(callable, job)`` task tuples. Diffusion samples keep the plain
``(process_diffusion, img_path, prefix, slug, idx, split)`` tuple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SampleKey:
    """The parts a sample name is built from: ``{prefix}_{slug}_{idx:09d}``."""

    prefix: str
    slug: str
    idx: int

    @property
    def name(self) -> str:
        """The registry and file name of the sample."""
        return f"{self.prefix}_{self.slug}_{self.idx:09d}"

    def at(self, offset: int) -> SampleKey:
        """The key of the sample ``offset`` rows further on (Parquet shards)."""
        return SampleKey(self.prefix, self.slug, self.idx + offset)


@dataclass(frozen=True)
class SourceContext:
    """What every sample of one source ref shares: task, annotation format and output."""

    task: str
    fmt: str | None
    output_root: str
    skip_labeling: bool = False


@dataclass
class ImageJob:
    """One source image (a path, raw bytes or a Parquet ``{"bytes": ...}`` cell)."""

    img_input: Any
    key: SampleKey
    context: SourceContext
    ann_data: Any
    split: str


@dataclass(frozen=True)
class ExistingSamples:
    """Sample names already in the registry and (lower-cased) already on disk."""

    names: set[str] = field(default_factory=set)
    on_disk: set[str] = field(default_factory=set)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and (name in self.names or name.lower() in self.on_disk)


@dataclass
class ShardJob:
    """A whole Parquet shard processed as one worker task."""

    pq_path: Any
    key: SampleKey
    context: SourceContext
    train_prob: float
    existing: ExistingSamples
    keep_prob: float
    num_rows: int
