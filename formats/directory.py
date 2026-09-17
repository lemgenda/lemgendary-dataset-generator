"""
Canonical directory layout.

Single responsibility: (a) act as a no-op Writer since the directory layout
is produced by the main compile pipeline, and (b) expose a Sample iterator
that any other writer can consume when migrating an existing manifold.

The directory layout is:
    <root>/images/<split>/<name>.<ext>
    <root>/labels/<split>/<name>.txt
    <root>/targets/<split>/<name>.<ext>    (restoration / SR)
    <root>/masks/<split>/<name>.<ext>      (segmentation)

Phase 4 of the 2026 modernization roadmap.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

from .base import Sample


_VALID_SPLITS = ("train", "val", "test")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class DirectoryWriter:
    """No-op writer.

    The canonical directory layout is always produced by the compile
    pipeline. The presence of this class in the writer registry means
    "directory" is a legal --also-format value that produces zero
    additional output.
    """

    def open(self, output_root: Path, policy: Any) -> None:
        return

    def write(self, sample: Sample) -> None:
        return

    def close(self) -> None:
        return


class DirectorySampleSource:
    """Iterate an existing compiled manifold as a stream of Samples.

    Builds a per-split name→path lookup once, then yields Samples in the
    order given by `index`. The index entries must carry `name`, `task`,
    `split`, and `source` keys (matching the schema written by
    manifold_compile.py).
    """

    def __init__(self, root: str | Path, index: list[dict[str, Any]]) -> None:
        self.root = Path(root)
        self.index = index
        self._image_index: dict[str, dict[str, Path]] = {}
        self._target_index: dict[str, dict[str, Path]] = {}
        self._mask_index: dict[str, dict[str, Path]] = {}
        self._label_index: dict[str, dict[str, Path]] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        for split in _VALID_SPLITS:
            self._image_index[split] = self._index_dir(self.root / "images" / split)
            self._target_index[split] = self._index_dir(self.root / "targets" / split)
            self._mask_index[split] = self._index_dir(self.root / "masks" / split)
            self._label_index[split] = self._index_dir(self.root / "labels" / split, any_ext=True)

    @staticmethod
    def _index_dir(d: Path, any_ext: bool = False) -> dict[str, Path]:
        """Build {name_without_ext: full_path} for a directory."""
        out: dict[str, Path] = {}
        if not d.exists():
            return out
        try:
            with os.scandir(str(d)) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    name = entry.name
                    dot = name.rfind(".")
                    if dot <= 0:
                        continue
                    if not any_ext:
                        ext = name[dot:].lower()
                        if ext not in _IMAGE_EXTS:
                            continue
                    stem = name[:dot]
                    out[stem] = Path(entry.path)
        except OSError:
            pass
        return out

    def __iter__(self) -> Iterator[Sample]:
        for entry in self.index:
            sample = self._to_sample(entry)
            if sample is not None:
                yield sample

    def __len__(self) -> int:
        return len(self.index)

    def _to_sample(self, entry: dict[str, Any]) -> Sample | None:
        name = entry.get("name")
        split = entry.get("split")
        task = entry.get("task", "quality")
        if not isinstance(name, str) or not isinstance(split, str):
            return None
        if split not in _VALID_SPLITS:
            return None

        img_path = self._image_index.get(split, {}).get(name)
        if img_path is None or not img_path.exists():
            return None

        try:
            image_bytes = img_path.read_bytes()
        except OSError:
            return None

        image_format = img_path.suffix.lstrip(".").lower() or "bin"

        target_path = self._target_index.get(split, {}).get(name)
        target_bytes = None
        if target_path is not None and target_path.exists():
            try:
                target_bytes = target_path.read_bytes()
            except OSError:
                pass

        mask_path = self._mask_index.get(split, {}).get(name)
        mask_bytes = None
        if mask_path is not None and mask_path.exists():
            try:
                mask_bytes = mask_path.read_bytes()
            except OSError:
                pass

        label_path = self._label_index.get(split, {}).get(name)
        label_text = None
        if label_path is not None and label_path.exists():
            try:
                label_text = label_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                pass

        metadata = {
            k: v for k, v in entry.items()
            if k not in ("id", "name", "split", "task") and v is not None
        }

        return Sample(
            name=name,
            task=task,
            split=split,
            image_bytes=image_bytes,
            image_format=image_format,
            target_bytes=target_bytes,
            mask_bytes=mask_bytes,
            label=label_text,
            metadata=metadata,
        )