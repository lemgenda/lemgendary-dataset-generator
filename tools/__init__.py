"""
LemGendary Dataset Compiler - Auxiliary Tools & Migration Scripts.

Uses PEP 562 lazy module attribute loading so importing a single submodule
(e.g. `from tools.modernization_queue import load_queue_manifest`) does not
eagerly import all 14 CLI tools and heavy runtime dependencies (`torch`,
`core.compiler_core`, etc.).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "convert_forex_manifold",
    "embed_parquet_descriptions",
    "forex_schema",
    "generate_cli",
    "generate_degrade",
    "manifold_reduce",
    "manifold_sync",
    "migrate_manifold_format",
    "migrate_manifold_image_format",
    "modernization_queue",
    "modernize_manifold",
    "mt5_bridge",
    "mt5_pipeline",
    "regenerate_manifolds_md",
    "sync_kaggle_metadata",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
