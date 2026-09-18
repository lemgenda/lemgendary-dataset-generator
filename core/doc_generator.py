"""LemGendary Dataset Documentation Generator (Facade).

Backward-compatible facade delegating to the modular core.docs subpackage.
"""

from __future__ import annotations

from pathlib import Path

from core.docs import (
    FOREX_COLUMN_FIELDS,
    MANIFEST_CACHE,
    MANIFOLD_TASK_MAP,
    MODELS_META,
    TASK_ARCH_BASE,
    TASK_META,
    UNIFIED_DATA,
    clean_readme as _clean_readme,
    format_source,
    generate_dataset_docs,
    main,
    regenerate_all_docs,
    regenerate_all_non_forex,
    scan_forex_manifold,
)
from core.docs.metadata import (
    MANIFEST_CACHE_PATH,
    MODELS_META_FILE,
    ROOT,
    UNIFIED_DATA_FILE,
)

__all__ = [
    "FOREX_COLUMN_FIELDS",
    "MANIFEST_CACHE",
    "MANIFEST_CACHE_PATH",
    "MANIFOLD_TASK_MAP",
    "MODELS_META",
    "MODELS_META_FILE",
    "ROOT",
    "TASK_ARCH_BASE",
    "TASK_META",
    "UNIFIED_DATA",
    "UNIFIED_DATA_FILE",
    "_clean_readme",
    "format_source",
    "generate_dataset_docs",
    "main",
    "regenerate_all_docs",
    "regenerate_all_non_forex",
    "scan_forex_manifold",
]

if __name__ == "__main__":
    main()