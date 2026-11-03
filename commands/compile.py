"""Top-level compile, transcode, audit, version and test commands."""

from __future__ import annotations

import logging
import unittest
from pathlib import Path

import typer

from commands.common import (
    _is_server_available,
    _resolve_target,
    console,
    route_through_sidecar,
)
from core.cli_args import PROJECT_NAME, __version__
from services import (
    AuditService,
    CompilerService,
    MigrationService,
)

logger = logging.getLogger(__name__)


# ─── Top-level commands ─────────────────────────────────────────────────────
def version() -> None:
    """Print the current version of the Dataset Compiler Suite."""
    console.print(f"[bold cyan]LemGendary Dataset Compiler Suite[/bold cyan] v{__version__}")
    console.print(f"[dim]Project: {PROJECT_NAME}[/dim]")


def compile_manifold(
    model: str | None = typer.Option(None, "--model", "-m", help="Model key to compile"),
    preset: str | None = typer.Option(
        None, "--preset", "-p", help="Compiler preset profile name (from presets.yaml)"
    ),
    max_gb: float | None = typer.Option(None, "--max-gb", help="Override max_size_gb"),
    suffix: str | None = typer.Option(None, "--suffix", help="Manifold name suffix override"),
    workers: int | None = typer.Option(None, "--workers", help="Parallel worker count"),
    no_vetting: bool = typer.Option(False, "--no-vetting", help="Disable NIMA quality gate"),
    no_labeling: bool = typer.Option(False, "--no-labeling", help="Disable YOLO auto-labeling"),
    no_hash: bool = typer.Option(False, "--no-hash", help="Disable deduplication hash"),
    image_format: str | None = typer.Option(
        None, "--image-format", help="webp | jpeg | png | keep"
    ),
    image_quality: int | None = typer.Option(None, "--image-quality"),
    target_quality: int | None = typer.Option(None, "--target-quality"),
    mask_format: str | None = typer.Option(None, "--mask-format", help="webp-lossless | png"),
    also_format: str | None = typer.Option(
        None, "--also-format", help="Comma-separated container formats (mds,litdata,...)"
    ),
    force_duplicate: bool = typer.Option(False, "--force-duplicate"),
    accept_space_loss: bool = typer.Option(False, "--accept-space-loss"),
    label_strategy: str | None = typer.Option(None, "--label-strategy"),
    prompt_strategy: str | None = typer.Option(None, "--prompt-strategy"),
    mask_strategy: str | None = typer.Option(None, "--mask-strategy"),
    shard_ext: str = typer.Option(
        "data", "--shard-ext", help="WebDataset shard file extension without dot (default: 'data')"
    ),
    no_server: bool = typer.Option(
        False, "--no-server", help="Bypass API server and execute in-process"
    ),
) -> None:
    """Compile a manifold from raw sources (delegates to CompilerService or API)."""
    if not no_server and _is_server_available():
        payload = {
            "model": model,
            "preset": preset,
            "max_gb": max_gb,
            "suffix": suffix,
            "workers": workers,
            "no_vetting": no_vetting,
            "no_labeling": no_labeling,
            "no_hash": no_hash,
            "image_format": image_format or "webp",
            "image_quality": image_quality or 92,
            "target_quality": target_quality or 95,
            "mask_format": mask_format or "webp-lossless",
            "also_format": also_format,
            "shard_ext": shard_ext,
            "force_duplicate": force_duplicate,
            "accept_space_loss": accept_space_loss,
            "label_strategy": label_strategy,
            "prompt_strategy": prompt_strategy,
            "mask_strategy": mask_strategy,
        }
        route_through_sidecar("compile", payload)

    code = CompilerService.run_compile(
        model=model,
        preset=preset,
        max_gb=max_gb,
        suffix=suffix,
        workers=workers,
        no_vetting=no_vetting,
        no_labeling=no_labeling,
        no_hash=no_hash,
        image_format=image_format,
        image_quality=image_quality,
        target_quality=target_quality,
        mask_format=mask_format,
        also_format=also_format,
        shard_ext=shard_ext,
        force_duplicate=force_duplicate,
        accept_space_loss=accept_space_loss,
        label_strategy=label_strategy,
        prompt_strategy=prompt_strategy,
        mask_strategy=mask_strategy,
    )
    raise typer.Exit(code=code)


# ─── transcode (Phase 3) ────────────────────────────────────────────────────
def transcode(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key"),
    manifold: str | None = typer.Option(
        None, "--manifold", help="Direct path to a manifold folder"
    ),
    image_format: str = typer.Option(
        "webp", "--image-format", help="Output format (webp/jpeg/png/keep)"
    ),
    image_quality: int = typer.Option(92, "--image-quality", help="Quality for images (1-100)"),
    target_quality: int = typer.Option(95, "--target-quality", help="Quality for targets (1-100)"),
    mask_format: str = typer.Option(
        "webp-lossless", "--mask-format", help="Mask format (webp-lossless/png)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without writing"),
) -> None:
    """Retroactively transcode an existing manifold's images, in place."""
    target = _resolve_target(model, manifold)
    if target is None:
        raise typer.Exit(code=1)
    code = MigrationService.transcode_images(
        manifold_path=target,
        image_format=image_format,
        image_quality=image_quality,
        target_quality=target_quality,
        mask_format=mask_format,
        dry_run=dry_run,
    )
    raise typer.Exit(code=code)


# ─── audit (Phase 2) ────────────────────────────────────────────────────────
def audit(
    model: str | None = typer.Option(None, "--model", "-m", help="Registry key to audit"),
    manifold: str | None = typer.Option(
        None, "--manifold", help="Direct path to a manifold folder"
    ),
    sample: int | None = typer.Option(None, "--sample", help="Cap image scan for a quick preview"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Full image audit pass (resolution, black-frame, dedup, hardlinks)."""
    target = _resolve_target(model, manifold)
    if target is None:
        raise typer.Exit(code=1)
    code = AuditService.audit_manifold(
        manifold_path=target,
        sample=sample,
        as_json=as_json,
    )
    raise typer.Exit(code=code)


# ─── Entry point ────────────────────────────────────────────────────────────

def test_datasets(
    pattern: str = typer.Option(
        "test_*.py", "--pattern", "-p", help="Test file pattern to execute"
    ),
) -> None:
    """Run full test suite for the Dataset Compiler Suite in-process using unittest."""
    project_root = Path(__file__).resolve().parent
    tests_dir = project_root / "tests"
    console.print(
        f"[bold cyan]Running LemGendary Dataset Compiler tests in {tests_dir}...[/bold cyan]"
    )
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        console.print(
            f"[bold red][FAIL] Tests failed ({len(result.failures)} failures, {len(result.errors)}"
            " errors)[/bold red]"
        )
        raise typer.Exit(1)
    console.print(
        f"[bold green][PASS] All {result.testsRun} tests passed successfully![/bold green]"
    )
