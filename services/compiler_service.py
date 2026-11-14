"""
CompilerService — compilation service and its process entry point.

Single Responsibility: Coordinate and invoke manifold compilation pipelines
programmatically, bridging CLI options, presets, and the core coordinator.

The CLI calls :func:`run_compile` in-process; an API job runs
``python -m services.compiler_service`` with the arguments of
:func:`compile_invocation`. Both end in :func:`compile_with_args`.
"""

from __future__ import annotations

import logging
import sys
from typing import TypedDict, Unpack

from rich.console import Console

from core.lazy_imports import load_manifold_compile

from ._errors import SERVICE_ERRORS
from .invocation import Flag, FlagTable, ServiceInvocation, flag_args

console = Console()
logger = logging.getLogger(__name__)


class CompileOptions(TypedDict, total=False):
    """Keyword options of :func:`run_compile` (all optional)."""

    model: str | None
    preset: str | None
    max_gb: float | None
    suffix: str | None
    workers: int | None
    no_vetting: bool
    no_labeling: bool
    no_hash: bool
    image_format: str | None
    image_quality: int | None
    target_quality: int | None
    mask_format: str | None
    also_format: str | None
    force_duplicate: bool
    accept_space_loss: bool
    label_strategy: str | None
    prompt_strategy: str | None
    mask_strategy: str | None
    shard_ext: str | None
    start_date: str | None
    end_date: str | None
    cleanup: bool


# How each option becomes ``core.cli_args.build_parser`` arguments, in CLI order.
COMPILE_FLAGS: FlagTable = (
    Flag("model", "--model", "text"),
    Flag("preset", "--preset", "text"),
    Flag("max_gb", "--max_gb", "float"),
    Flag("suffix", "--suffix", "text"),
    Flag("workers", "--workers", "int"),
    Flag("no_vetting", "--no-vetting", "switch"),
    Flag("no_labeling", "--no-labeling", "switch"),
    Flag("no_hash", "--no-hash", "switch"),
    Flag("image_format", "--image-format", "text"),
    Flag("image_quality", "--image-quality", "int"),
    Flag("target_quality", "--target-quality", "int"),
    Flag("mask_format", "--mask-format", "text"),
    Flag("also_format", "--also-format", "text"),
    Flag("shard_ext", "--shard-ext", "text"),
    Flag("start_date", "--start-date", "text"),
    Flag("end_date", "--end-date", "text"),
    Flag("label_strategy", "--label-strategy", "text"),
    Flag("force_duplicate", "--force-duplicate", "switch"),
    Flag("accept_space_loss", "--accept-space-loss", "switch"),
    Flag("label_strategy", "--label-strategy", "text"),
    Flag("prompt_strategy", "--prompt-strategy", "text"),
    Flag("mask_strategy", "--mask-strategy", "text"),
    Flag("cleanup", "--cleanup", "switch"),
)


def build_compile_args(options: CompileOptions) -> list[str]:
    """Translate :func:`run_compile` options into compile CLI arguments."""
    return flag_args(options, COMPILE_FLAGS)


def compile_invocation(options: CompileOptions) -> ServiceInvocation:
    """The isolated-process form of ``run_compile(**options)``."""
    return ServiceInvocation(__name__, tuple(build_compile_args(options)))


def compile_with_args(argv: list[str]) -> int:
    """Parse compile arguments (presets applied), compile and return the exit code.

    The exit code comes from ``finalize_compile``: 1 when no manifold was
    compiled or one produced no samples. ``--cleanup`` purges loose images
    only after a successful compile.
    """
    manifold_compile = load_manifold_compile()
    parsed = manifold_compile.parse_compile_args(argv)
    try:
        outcomes = manifold_compile.process_dataset(parsed)
    except SERVICE_ERRORS as exc:
        logger.exception("Compilation error")
        console.print(f"[bold red]Compilation error:[/bold red] {exc}")
        return 1
    return manifold_compile.finalize_compile(outcomes, purge_loose=parsed.cleanup)


def run_compile(**options: Unpack[CompileOptions]) -> int:
    """Run a compile in-process with the given CLI options; returns the exit code."""
    return compile_with_args(build_compile_args(options))


def main(argv: list[str] | None = None) -> int:
    """Process entry point: ``python -m services.compiler_service <compile args>``."""
    return compile_with_args(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    sys.exit(main())
