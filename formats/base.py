"""
Container-format writer protocol.

Defines the shape that every container writer (Phase 4) must implement.
Nothing here is called by the current compiler — Phase 4 wires these into
`manifold_compile.py` when `--also-format <name>` is passed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable


class Sample(NamedTuple):
    """One compiled sample, ready to be written to any container.

    Populated by Phase 4 from the current directory layout. Every field
    except `name` is optional so writers can gracefully skip missing
    components (e.g. quality manifolds have no targets or masks).
    """
    name: str
    image_bytes: bytes
    image_format: str                          # 'webp' | 'jpeg' | 'png'
    target_bytes: bytes | None = None
    mask_bytes: bytes | None = None
    label: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None     # hash, source, task, split, nima_score, caption, style_tag


@runtime_checkable
class Writer(Protocol):
    """Structural interface for a container writer.

    Lifecycle:
        writer.open(output_root, config)
        while samples remain: writer.write(sample)
        writer.close()
    """

    def open(self, output_root: Path, config: Any) -> None: ...

    def write(self, sample: Sample) -> None: ...

    def close(self) -> None: ...