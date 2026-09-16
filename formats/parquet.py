"""
Parquet writer.

Single responsibility: write Samples to columnar Parquet files with Zstd
compression. Currently used only for the Forex manifold (annual shards).

Phase 4. The Forex pipeline currently writes Parquet directly via pyarrow;
this module exists so future manifolds can opt into the same container.
"""

from __future__ import annotations

# Phase 4 will define: class ParquetWriter(Writer)