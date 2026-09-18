"""
CompilerService — In-process compilation service.

Single Responsibility: Coordinate and invoke manifold compilation pipelines
programmatically, bridging CLI options, presets, and the core coordinator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


class CompilerService:
    """Encapsulates dataset compilation orchestration."""

    @staticmethod
    def run_compile(
        model: str | None = None,
        preset: str | None = None,
        max_gb: float | None = None,
        suffix: str | None = None,
        workers: int | None = None,
        no_vetting: bool = False,
        no_labeling: bool = False,
        no_hash: bool = False,
        image_format: str | None = None,
        image_quality: int | None = None,
        target_quality: int | None = None,
        mask_format: str | None = None,
        also_format: str | None = None,
        force_duplicate: bool = False,
        accept_space_loss: bool = False,
        label_strategy: str | None = None,
        prompt_strategy: str | None = None,
        mask_strategy: str | None = None,
    ) -> int:
        from core import manifold_compile

        # Build list of CLI arguments for parse_compile_args
        args_list: list[str] = []
        if model:
            args_list.extend(["--model", model])
        if preset:
            args_list.extend(["--preset", preset])
        if max_gb is not None:
            args_list.extend(["--max_gb", str(max_gb)])
        if suffix:
            args_list.extend(["--suffix", suffix])
        if workers is not None:
            args_list.extend(["--workers", str(workers)])
        if no_vetting:
            args_list.append("--no-vetting")
        if no_labeling:
            args_list.append("--no-labeling")
        if no_hash:
            args_list.append("--no-hash")
        if image_format:
            args_list.extend(["--image-format", image_format])
        if image_quality is not None:
            args_list.extend(["--image-quality", str(image_quality)])
        if target_quality is not None:
            args_list.extend(["--target-quality", str(target_quality)])
        if mask_format:
            args_list.extend(["--mask-format", mask_format])
        if also_format:
            args_list.extend(["--also-format", also_format])
        if force_duplicate:
            args_list.append("--force-duplicate")
        if accept_space_loss:
            args_list.append("--accept-space-loss")
        if label_strategy:
            args_list.extend(["--label-strategy", label_strategy])
        if prompt_strategy:
            args_list.extend(["--prompt-strategy", prompt_strategy])
        if mask_strategy:
            args_list.extend(["--mask-strategy", mask_strategy])

        parsed = manifold_compile.parse_compile_args(args_list)
        try:
            manifold_compile.process_dataset(parsed)
            return 0
        except Exception as exc:
            console.print(f"[bold red]Compilation error:[/bold red] {exc}")
            return 1
