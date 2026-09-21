"""
LemGendary Dataset Compiler — Core Engine Package.

Consolidated package for compiler coordinator, manifest validation,
registry lifecycle, presets, CLI arguments, and documentation generation.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import (
        cli_args,
        common_sync,
        compiler_core,
        config_schema,
        doc_generator,
        manifold_compile,
        presets,
        registry,
    )

__all__ = [
    "cli_args",
    "common_sync",
    "compiler_core",
    "config_schema",
    "doc_generator",
    "manifold_compile",
    "presets",
    "registry",
]


def __getattr__(name: str) -> Any:
    """Lazy module loader (PEP 562) preventing circular import cascades."""
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

