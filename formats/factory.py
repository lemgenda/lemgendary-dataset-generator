"""Container writer factory: maps a format name to its Writer."""

from __future__ import annotations

from .base import Writer
from .directory import DirectoryWriter
from .litdata import LitDataWriter
from .mds import MDSWriter
from .parquet import ParquetWriter
from .webdataset import WebDatasetWriter


def make_writer(fmt: str) -> Writer:
    """Factory: return an unopened Writer for the given format name.

    Raises ValueError for unknown formats. Optional-dependency import errors
    are raised at .open() time, not at factory time, so the CLI can report
    a clean message.
    """
    normalized = fmt.strip().lower()
    if normalized == "directory":
        return DirectoryWriter()
    if normalized == "mds":
        return MDSWriter()
    if normalized == "litdata":
        return LitDataWriter()
    if normalized == "webdataset":
        return WebDatasetWriter()
    if normalized == "parquet":
        return ParquetWriter()
    raise ValueError(f"Unknown container format: {fmt}")
