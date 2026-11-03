"""
GenerationService — In-process smart generation service.

Single Responsibility: Coordinate and invoke label, prompt, and mask
generators over manifolds in-process without subprocess spawning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict, Unpack

from rich.console import Console

from tools import generate_cli

from ._errors import SERVICE_ERRORS

console = Console()
logger = logging.getLogger(__name__)


class GenerateOptions(TypedDict, total=False):
    """Optional settings for :meth:`GenerationService.generate`."""

    template: str
    device: str
    sample: int | None
    dry_run: bool


def run_generation(
    manifold: Path,
    kind: str,
    strategy: str,
    **options: Unpack[GenerateOptions],
) -> int:
    """Run smart generation pass over an existing manifold.

    ``options`` defaults: ``template="diffusers-v1"``, ``device="cpu"``,
    ``sample=None`` (no cap) and ``dry_run=False``.
    """

    try:
        return generate_cli.run(
            manifold=manifold,
            kind=kind,
            strategy=strategy,
            template=options.get("template", "diffusers-v1"),
            device=options.get("device", "cpu"),
            sample_limit=options.get("sample"),
            dry_run=options.get("dry_run", False),
        )
    except SERVICE_ERRORS as exc:
        logger.exception("Generation error")
        console.print(f"[bold red]Generation error:[/bold red] {exc}")
        return 1
