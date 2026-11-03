"""
LLMService — LLM instruction manifold compilation and its process entry point.

Single Responsibility: compile an LLM manifold declared in unified_data.yaml
and describe the result. The CLI and the synchronous API route call
:func:`compile_llm_manifold` in-process; the asynchronous API job runs
``python -m services.llm_service`` with the arguments of
:func:`llm_compile_invocation`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import yaml

from tools.llm_helper import base_compiler

from .invocation import Flag, FlagTable, ServiceInvocation, add_flags, flag_args

logger = logging.getLogger(__name__)

#: Expected failures of an LLM compile (bad key, bad data, missing files or deps).
LLM_COMPILE_ERRORS: tuple[type[BaseException], ...] = (
    KeyError,
    ValueError,
    NotImplementedError,
    OSError,
    ImportError,
    RuntimeError,
    yaml.YAMLError,
)

LLM_COMPILE_FLAGS: FlagTable = (Flag("dataset", "--dataset", "text"),)


def compile_llm_manifold(dataset_key: str) -> dict[str, Any]:
    """Compile the LLM manifold ``dataset_key``; returns the compiler's result mapping."""
    return base_compiler.BaseLLMCompiler().compile_manifold(dataset_key)


def summary_lines(result: dict[str, Any]) -> list[str]:
    """Human-readable record and shard counts of a compile result."""
    return [
        f"  Total records: {result.get('total_records')}",
        f"  Train shard:   {result.get('train_path')} ({result.get('train_count')} samples)",
        f"  Val shard:     {result.get('val_path')} ({result.get('val_count')} samples)",
    ]


def llm_compile_invocation(dataset_key: str) -> ServiceInvocation:
    """The isolated-process form of ``compile_llm_manifold(dataset_key)``."""
    return ServiceInvocation(__name__, tuple(flag_args({"dataset": dataset_key},
                                                       LLM_COMPILE_FLAGS)))


def main(argv: list[str] | None = None) -> int:
    """Process entry point: ``python -m services.llm_service --dataset KEY``."""
    parser = add_flags(argparse.ArgumentParser(), LLM_COMPILE_FLAGS)
    ns = parser.parse_args(argv)
    if not ns.dataset:
        parser.error("--dataset is required")
    print(f"Compiling LLM manifold '{ns.dataset}'...")
    try:
        result = compile_llm_manifold(ns.dataset)
    except LLM_COMPILE_ERRORS as exc:
        logger.exception("LLM compilation failed for dataset %s", ns.dataset)
        print(f"LLM compilation error: {exc}")
        return 1
    print(f"Successfully compiled {ns.dataset}:")
    print("\n".join(summary_lines(result)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
