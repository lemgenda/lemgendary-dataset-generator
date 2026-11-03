"""
Container-format writer protocol and shared Sample type.

Phase 4 of the 2026 modernization roadmap.

Every container writer (MDS, LitData, WebDataset, Parquet) implements the
`Writer` Protocol. The directory layout is the canonical fallback and is
always written first; container writes happen in addition.

Lifecycle:
    writer.open(output_root, policy)
    writer.write(sample)  # repeated
    writer.close()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable


class Sample(NamedTuple):
    """One compiled sample, ready to be written to any container.

    Populated by `formats.directory.DirectorySampleSource` when migrating
    an existing manifold, or inline during compilation when `--also-format`
    is passed.

    Every field except `name`, `task`, and `split` is optional so writers
    can gracefully skip missing components (e.g. quality manifolds have no
    targets or masks).
    """

    name: str
    task: str
    split: str
    image_bytes: bytes
    image_format: str                          # 'webp' | 'jpeg' | 'png'
    target_bytes: bytes | None = None
    mask_bytes: bytes | None = None
    label: str | None = None                   # raw text from labels/<name>.txt
    metadata: dict[str, Any] | None = None     # hash, source, nima_score, caption, ...


@runtime_checkable
class Writer(Protocol):
    """Structural interface for a container writer."""

    def open(self, output_root: Path, policy: Any) -> None: ...

    def write(self, sample: Sample) -> None: ...

    def close(self) -> None: ...


def parse_also_format(value: str) -> list[str]:
    """Split a comma-separated --also-format value into a list of format names."""
    return [f.strip().lower() for f in value.split(",") if f.strip()]
