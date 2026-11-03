#!/usr/bin/env python3
"""
LemGendary Forex Manifold Parquet Metadata & Column Description Injector
=======================================================================
Backwards-compatible shim module delegating to forex.injector.
"""

from forex.injector import main, process_parquet_file

__all__ = [
    "main",
    "process_parquet_file",
]

if __name__ == "__main__":
    main()
