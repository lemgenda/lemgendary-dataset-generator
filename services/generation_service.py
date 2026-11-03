"""
GenerationService — In-process smart generation service.

Single Responsibility: Coordinate and invoke label, prompt, and mask
generators over manifolds in-process without subprocess spawning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from rich.console import Console

from ._errors import SERVICE_ERRORS

console = Console()
logger = logging.getLogger(__name__)


class GenerationService:
    """Encapsulates label, prompt, and mask generation."""

    @staticmethod
    def generate(
        manifold: Path,
        kind: str,
        strategy: str,
        template: str = "diffusers-v1",
        device: str = "cpu",
        sample: int | None = None,
        dry_run: bool = False,
    ) -> int:
        """Run smart generation pass over an existing manifold."""
        from tools import generate_cli

        try:
            return generate_cli.run(
                manifold=manifold,
                kind=kind,
                strategy=strategy,
                template=template,
                device=device,
                sample_limit=sample,
                dry_run=dry_run,
            )
        except SERVICE_ERRORS as exc:
            logger.exception("Generation error")
            console.print(f"[bold red]Generation error:[/bold red] {exc}")
            return 1
