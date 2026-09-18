"""
DegradeService — In-process degradation synthesis service.

Single Responsibility: Coordinate and invoke synthetic restoration manifold
generation programmatically without subprocess spawning.
"""

from __future__ import annotations

from typing import Literal, cast
from rich.console import Console

console = Console()


class DegradeService:
    """Encapsulates degradation manifold synthesis."""

    @staticmethod
    def run_degrade(
        source: str,
        output: str,
        profile: str = "motion-blur+iso-noise",
        intensity: str = "medium",
        pairs: int | None = None,
        val_split: float = 0.12,
        seed: int = 42,
        image_format: str = "webp",
        workers: int | None = None,
        dry_run: bool = False,
    ) -> int:
        """Execute synthetic degradation manifold generation in-process."""
        from tools import generate_degrade

        fmt = cast(Literal["webp", "jpeg", "png", "keep"], image_format)
        try:
            return generate_degrade.synthesize_manifold(
                source=source,
                output=output,
                profile_expr=profile,
                intensity=intensity,
                pairs=pairs,
                val_split=val_split,
                seed=seed,
                image_format=fmt,
                workers=workers,
                dry_run=dry_run,
            )
        except Exception as exc:
            console.print(f"[bold red]Degradation synthesis error:[/bold red] {exc}")
            return 1
