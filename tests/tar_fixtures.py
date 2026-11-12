"""Shared builder of small synthetic tar shards for the shard and repair tests."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path


def write_tar(path: Path, members: list[tuple[str, bytes]]) -> Path:
    """Write ``members`` (name, payload) in order to a plain tar at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path
