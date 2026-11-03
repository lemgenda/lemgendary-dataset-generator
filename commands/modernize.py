"""`modernize` sub-commands: modernization queue, transcode and reduction."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.table import Table

from commands.common import (
    console,
)
from services import (
    DocService,
    MigrationService,
)
from tools.modernization_queue import load_queue_manifest

logger = logging.getLogger(__name__)
modernize_app = typer.Typer(help="Dataset modernization queue, transcode, and scale reduction.")


# ─── modernize manifolds (Phase 1.6 / Modernization) ────────────────────────
# Registered on the `modernize` sub-app: a top-level `modernize` command would be
# shadowed by the sub-app of the same name and never be reachable.
@modernize_app.command("manifolds")
def modernize_manifolds(
    all_: bool = typer.Option(False, "--all", help="Select all eligible manifolds"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation gate"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan, do not modify anything"),
    datasets: str | None = typer.Option(
        None, "--datasets", help="Comma-separated folder names or dataset keys"
    ),
    skip_kaggle: bool = typer.Option(False, "--skip-kaggle", help="Skip Kaggle re-upload"),
    image_format: str = typer.Option(
        "webp", "--image-format", help="Transcode images during modernize (webp/jpeg/png/keep)"
    ),
    image_quality: int = typer.Option(92, "--image-quality", help="Quality for images (1-100)"),
    also_format: str | None = typer.Option(
        None,
        "--also-format",
        help=(
            "Optional container format to convert dataset into (e.g., webdataset, mds, litdata,"
            " parquet)"
        ),
    ),
    skip_transcode: bool = typer.Option(False, "--skip-transcode", help="Skip WebP transcoding"),
    skip_container: bool = typer.Option(
        False, "--skip-container", help="Skip container format conversion"
    ),
) -> None:
    """Modernize manifolds: retire the legacy suffix, transcode to WebP, write containers."""
    code = MigrationService.modernize_manifolds(
        all_=all_,
        yes=yes,
        dry_run=dry_run,
        datasets=datasets,
        skip_kaggle=skip_kaggle,
        image_format=image_format,
        image_quality=image_quality,
        also_format=also_format,
        skip_transcode=skip_transcode,
        skip_container=skip_container,
    )
    raise typer.Exit(code=code)


# ─── modernize sub-app ──────────────────────────────────────────────────────
@modernize_app.command("queue")
def modernize_queue() -> None:
    """Display modernization queue status and pending conversions."""
    datasets_root = Path(__file__).resolve().parent.parent / "LemGendaryDatasets"
    registry_yaml = Path(__file__).resolve().parent / "unified_data.yaml"
    items = load_queue_manifest(datasets_root.resolve(), registry_yaml.resolve())

    table = Table(title="LemGendary Dataset Modernization Queue")
    table.add_column("Dataset Key", style="cyan")
    table.add_column("Folder", style="green")
    table.add_column("Format", style="magenta")
    table.add_column("Status", style="yellow")
    table.add_column("Reason", style="dim")

    for item in items:
        if item.is_containerized:
            status_label, reason = (
                "CONTAINERIZED",
                f"{item.container_bytes / (1024**3):.1f} GB ready",
            )
        elif item.zip_path:
            status_label, reason = "PENDING STREAM", item.zip_path.name
        elif item.has_loose_files:
            status_label, reason = "PENDING PACK", "Loose images"
        elif item.legacy_dir:
            status_label, reason = "PENDING LEGACY", item.legacy_dir.name
        else:
            status_label, reason = "MISSING SOURCE", "-"
        table.add_row(
            item.key,
            item.modernized_folder,
            item.canonical_format,
            status_label,
            reason,
        )
    console.print(table)


@modernize_app.command("run")
def modernize_run(
    dataset: str | None = typer.Option(
        None, "--dataset", "-d", help="Specific dataset key to modernize"
    ),
    workers: int | None = typer.Option(
        None, "--workers", "-w", help="Worker threads for WebP transcoding"
    ),
    push_kaggle: bool = typer.Option(
        False, "--push-kaggle", help="Push to Kaggle after conversion"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview plan without executing conversions"
    ),
    force: bool = typer.Option(False, "--force", help="Force re-conversion"),
    purge_source: bool = typer.Option(
        False,
        "--purge-source",
        help="Delete zip, legacy folder and loose images after a successful conversion",
    ),
) -> None:
    """Run batch modernization queue conversion (same service as POST /api/modernize/run)."""
    code = MigrationService.run_modernization_queue(
        dataset=dataset,
        workers=workers,
        push_kaggle=push_kaggle,
        dry_run=dry_run,
        force=force,
        purge_source=purge_source,
    )
    raise typer.Exit(code=code)


@modernize_app.command("reduce")
def modernize_reduce(
    dataset: str | None = typer.Option(
        None, "--dataset", "-d", help="Manifold folder name to reduce"
    ),
    fraction: float | None = typer.Option(
        None, "--fraction", help="Share of samples to keep, in (0, 1]"
    ),
    max_gb: float | None = typer.Option(
        None, "--max-gb", help="Target max size for reduced variant"
    ),
    suffix: str | None = typer.Option(
        None, "--suffix", help="Suffix for the reduced manifold (default: Reduced)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not prompt; requires --dataset"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report the plan only; write nothing"),
) -> None:
    """Create a downsampled manifold variant (interactive unless --yes)."""
    code = MigrationService.reduce_manifolds(
        max_gb=max_gb, suffix=suffix, dataset=dataset, fraction=fraction, yes=yes, dry_run=dry_run
    )
    raise typer.Exit(code=code)


@modernize_app.command("regen")
def modernize_regen(
    check: bool = typer.Option(False, "--check", help="Dry-run: print summary only"),
) -> None:
    """Regenerate manifolds.md registry index."""
    code = DocService.rebuild_manifolds_md(check_only=check)
    raise typer.Exit(code=code)
