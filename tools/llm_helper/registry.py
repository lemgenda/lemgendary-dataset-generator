"""Dynamic Plugin Registry for LLM Ingestion, Chunking, and Synthesis.

Provides an extensible registry architecture to eliminate hardcoded model/task
branches from generic compiler modules.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_SOURCE_ADAPTERS: dict[str, Callable[..., Any]] = {}
_CHUNKERS: dict[str, Callable[..., Any]] = {}
_SYNTHESIZERS: dict[str, Callable[..., Any]] = {}


def register_source_adapter(kind: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a source ingestor adapter."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _SOURCE_ADAPTERS[kind] = fn
        return fn
    return decorator


def get_source_adapter(kind: str) -> Callable[..., Any] | None:
    """Retrieve a registered source adapter."""
    return _SOURCE_ADAPTERS.get(kind)


def register_chunker(strategy: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a chunking strategy function or class."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _CHUNKERS[strategy] = fn
        return fn
    return decorator


def get_chunker(strategy: str) -> Callable[..., Any] | None:
    """Retrieve a registered chunker strategy."""
    return _CHUNKERS.get(strategy)


def register_synthesizer(task: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register an instruction synthesis engine for a task."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _SYNTHESIZERS[task] = fn
        return fn
    return decorator


def get_synthesizer(task: str) -> Callable[..., Any] | None:
    """Retrieve a registered synthesis engine."""
    return _SYNTHESIZERS.get(task)
