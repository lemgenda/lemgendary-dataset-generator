"""Top-level smart-generation and degradation commands (label, prompt, mask, degrade)."""

from __future__ import annotations

import logging

import typer

from commands.common import (
    _is_server_available,
    _resolve_target,
    route_through_sidecar,
)
from services import (
    DegradeService,
    GenerationService,
)

logger = logging.getLogger(__name__)


# ─── smart generation (Phase 5) ─────────────────────────────────────────────
def label(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key"),
    manifold: str | None = typer.Option(
        None, "--manifold", help="Direct path to a manifold folder"
    ),
    strategy: str = typer.Option(
        "blip_caption",
        "--strategy",
        help="blip_caption | clip_zeroshot | yolo_detection | parsenet_segmentation | nima_quality",
    ),
    device: str = typer.Option("cpu", "--device", help="cpu | cuda | cuda:N"),
    sample: int | None = typer.Option(None, "--sample", help="Cap the image scan"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Generate labels for an existing manifold."""
    target = _resolve_target(model, manifold)
    if target is None:
        raise typer.Exit(code=1)
    code = GenerationService.generate(
        manifold=target,
        kind="label",
        strategy=strategy,
        device=device,
        sample=sample,
        dry_run=dry_run,
    )
    raise typer.Exit(code=code)


def prompt(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key"),
    manifold: str | None = typer.Option(
        None, "--manifold", help="Direct path to a manifold folder"
    ),
    template: str = typer.Option(
        "diffusers-v1", "--template", help="diffusers-v1 | sd-v1 | flux-v1 | minimal"
    ),
    device: str = typer.Option("cpu", "--device", help="cpu | cuda | cuda:N"),
    sample: int | None = typer.Option(None, "--sample", help="Cap the image scan"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Generate structured prompts for diffusion manifolds."""
    target = _resolve_target(model, manifold)
    if target is None:
        raise typer.Exit(code=1)
    code = GenerationService.generate(
        manifold=target,
        kind="prompt",
        strategy=template,
        template=template,
        device=device,
        sample=sample,
        dry_run=dry_run,
    )
    raise typer.Exit(code=code)


def mask(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key"),
    manifold: str | None = typer.Option(
        None, "--manifold", help="Direct path to a manifold folder"
    ),
    strategy: str = typer.Option("parsenet", "--strategy", help="parsenet | sam | modnet"),
    device: str = typer.Option("cpu", "--device", help="cpu | cuda | cuda:N"),
    sample: int | None = typer.Option(None, "--sample", help="Cap the image scan"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Generate masks for a segmentation manifold."""
    target = _resolve_target(model, manifold)
    if target is None:
        raise typer.Exit(code=1)
    code = GenerationService.generate(
        manifold=target,
        kind="mask",
        strategy=strategy,
        device=device,
        sample=sample,
        dry_run=dry_run,
    )
    raise typer.Exit(code=code)


# ─── degradation engine (Phase 6) ───────────────────────────────────────────
def degrade(
    source: str = typer.Option(
        ..., "--source", "-s", help="Source clean dataset directory or manifold name"
    ),
    output: str = typer.Option(..., "--output", "-o", help="Target synthetic manifold name"),
    profile: str = typer.Option(
        "motion-blur+iso-noise",
        "--profile",
        "-p",
        help="Degradation profile expression or preset alias",
    ),
    intensity: str = typer.Option("medium", "--intensity", help="low | medium | high"),
    pairs: int | None = typer.Option(None, "--pairs", help="Max sample pairs to synthesize"),
    val_split: float = typer.Option(0.12, "--val-split", help="Validation split ratio"),
    seed: int = typer.Option(42, "--seed", help="Deterministic RNG seed"),
    image_format: str = typer.Option("webp", "--image-format", help="webp | jpeg | png"),
    workers: int | None = typer.Option(None, "--workers", "-w", help="Worker thread count"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview plan without writing"),
    no_server: bool = typer.Option(
        False, "--no-server", help="Bypass API server and execute in-process"
    ),
) -> None:
    """Compiler-time degradation synthesis (blur, noise, haze, rain, jpeg, film)."""
    if not no_server and _is_server_available():
        payload = {
            "source": source,
            "output": output,
            "profile": profile,
            "intensity": intensity,
            "pairs": pairs,
            "val_split": val_split,
            "seed": seed,
            "image_format": image_format,
            "workers": workers,
            "dry_run": dry_run,
        }
        route_through_sidecar("degrade", payload)

    code = DegradeService.run_degrade(
        source=source,
        output=output,
        profile=profile,
        intensity=intensity,
        pairs=pairs,
        val_split=val_split,
        seed=seed,
        image_format=image_format,
        workers=workers,
        dry_run=dry_run,
    )
    raise typer.Exit(code=code)
