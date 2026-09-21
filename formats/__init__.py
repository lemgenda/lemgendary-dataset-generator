"""
LemGendary Dataset Compiler — Container format layer.

Phase 4 of the 2026 modernization roadmap.

Public surface:
    Sample               — NamedTuple passed to every writer
    Writer               — Protocol implemented by every writer
    make_writer(name)    — Writer factory
    parse_also_format(v) — CLI --also-format parser

    DirectoryWriter      — no-op; the directory layout is always written
    DirectorySampleSource— iterate an existing manifold as Sample stream
    MDSWriter            — MosaicML Streaming
    LitDataWriter        — PyTorch LitData
    WebDatasetWriter     — WebDataset .tar
    ParquetWriter        — Parquet (tabular + binary)
    ShardWriter          — legacy WebDataset shim

    ImageTranscoder      — Phase 3 (see .transcode)
    KeepFormatError      — Phase 3
"""

from __future__ import annotations

__all__ = [
    "Sample",
    "Writer",
    "make_writer",
    "parse_also_format",
    "DirectoryWriter",
    "DirectorySampleSource",
    "MDSWriter",
    "LitDataWriter",
    "WebDatasetWriter",
    "ParquetWriter",
    "ShardWriter",
    "ImageTranscoder",
    "KeepFormatError",
]

from .base import Sample, Writer, make_writer, parse_also_format
from .directory import DirectoryWriter, DirectorySampleSource
from .mds import MDSWriter
from .litdata import LitDataWriter
from .webdataset import WebDatasetWriter, ShardWriter
from .parquet import ParquetWriter
from .transcode import ImageTranscoder, KeepFormatError