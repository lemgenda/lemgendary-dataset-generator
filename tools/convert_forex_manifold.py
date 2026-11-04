#!/usr/bin/env python3
"""
LemGendary Forex Manifold Parquet Conversion & Verification Engine
===================================================================
Backwards-compatible shim module delegating to forex.converter.
"""

from forex.converter import (
    convert_year,
    get_dir_size_gb,
    get_file_size_gb,
    main,
)

__all__ = [
    "convert_year",
    "get_dir_size_gb",
    "get_file_size_gb",
    "main",
]

if __name__ == "__main__":
    main()
