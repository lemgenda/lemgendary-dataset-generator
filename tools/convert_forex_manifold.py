#!/usr/bin/env python3
"""
LemGendary Forex Manifold Parquet Conversion & Verification Engine
===================================================================
Backwards-compatible shim module delegating to forex.converter.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "tools" else Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
