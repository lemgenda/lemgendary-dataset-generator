"""
CompilerService — In-process compilation service.

Single Responsibility: Coordinate and invoke manifold compilation pipelines
programmatically, bridging CLI options, presets, and the core coordinator.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict, Unpack

from rich.console import Console

from core.lazy_imports import load_manifold_compile

from ._errors import SERVICE_ERRORS

console = Console()
logger = logging.getLogger(__name__)


class CompileOptions(TypedDict, total=False):
    """Keyword options of ``CompilerService.run_compile`` (all optional)."""

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


# How each option becomes compile CLI arguments, in CLI order:
#   "text"   -> ``flag value`` when the value is truthy
#   "number" -> ``flag str(value)`` when the value is not None
#   "switch" -> ``flag`` when the value is truthy
_FlagKind = Literal["text", "number", "switch"]
_CLI_FLAGS: tuple[tuple[str, str, _FlagKind], ...] = (
    ("model", "--model", "text"),
    ("preset", "--preset", "text"),
    ("max_gb", "--max_gb", "number"),
    ("suffix", "--suffix", "text"),
    ("workers", "--workers", "number"),
    ("no_vetting", "--no-vetting", "switch"),
    ("no_labeling", "--no-labeling", "switch"),
    ("no_hash", "--no-hash", "switch"),
    ("image_format", "--image-format", "text"),
    ("image_quality", "--image-quality", "number"),
    ("target_quality", "--target-quality", "number"),
    ("mask_format", "--mask-format", "text"),
    ("also_format", "--also-format", "text"),
    ("shard_ext", "--shard-ext", "text"),
    ("force_duplicate", "--force-duplicate", "switch"),
    ("accept_space_loss", "--accept-space-loss", "switch"),
    ("label_strategy", "--label-strategy", "text"),
    ("prompt_strategy", "--prompt-strategy", "text"),
    ("mask_strategy", "--mask-strategy", "text"),
)
_KNOWN_OPTIONS = frozenset(name for name, _flag, _kind in _CLI_FLAGS)


def build_compile_args(options: CompileOptions) -> list[str]:
    """Translate ``run_compile`` options into ``parse_compile_args`` arguments."""
    values: dict[str, Any] = dict(options)
    unknown = set(values) - _KNOWN_OPTIONS
    if unknown:
        raise TypeError(f"Unknown compile option(s): {', '.join(sorted(unknown))}")

    args_list: list[str] = []
    for name, flag, kind in _CLI_FLAGS:
        value = values.get(name)
        if kind == "number":
            if value is not None:
                args_list.extend([flag, str(value)])
        elif value:
            args_list.extend([flag] if kind == "switch" else [flag, str(value)])
    return args_list


def run_compile(**options: Unpack[CompileOptions]) -> int:
    """Run a compile in-process with the given CLI options; returns the exit code."""
    manifold_compile = load_manifold_compile()

    parsed = manifold_compile.parse_compile_args(build_compile_args(options))
    try:
        manifold_compile.process_dataset(parsed)
        return 0
    except SERVICE_ERRORS as exc:
        logger.exception("Compilation error")
        console.print(f"[bold red]Compilation error:[/bold red] {exc}")
        return 1
