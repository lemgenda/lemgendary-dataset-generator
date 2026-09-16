"""
MosaicML Streaming (MDS) writer.

Single responsibility: serialize a stream of Samples into sharded `.mds`
files with Zstd compression and true global shuffle support.

Requires: mosaicml-streaming

Phase 4. Populated only when `--also-format mds` is passed and the hardlink
pre-flight gate permits container write for that manifold.
"""

from __future__ import annotations

# Phase 4 will define: class MDSWriter(Writer)