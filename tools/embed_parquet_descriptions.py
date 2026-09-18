#!/usr/bin/env python3
"""
LemGendary Forex Manifold Parquet Metadata & Column Description Injector
=======================================================================
Backwards-compatible shim module delegating to forex.injector.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "tools" else Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from forex.injector import main, process_parquet_file

__all__ = [
    "main",
    "process_parquet_file",
]

if __name__ == "__main__":
    main()
