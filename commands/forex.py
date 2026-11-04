"""`forex` sub-commands: Forex universe and Parquet conversion."""

from __future__ import annotations

import logging

import typer

from commands.common import (
    PROJECT_ROOT,
    console,
)
from core.lazy_imports import load_forex_module

logger = logging.getLogger(__name__)
forex_app = typer.Typer(help="Forex universe and Parquet conversion operations.")


# ─── forex sub-app ──────────────────────────────────────────────────────────
@forex_app.command("convert")
def forex_convert(
    year: int | None = typer.Option(
        None, "--year", "-y", help="Specific year to convert (2019..2026)"
    ),
    all_: bool = typer.Option(False, "--all", help="Convert all years 2019..2026 sequentially"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate conversion without writing"),
    skip_cleanup: bool = typer.Option(
        False, "--skip-cleanup", help="Keep source .npy directories after conversion"
    ),
) -> None:
    """Convert raw .npy Forex manifolds into unified Apache Parquet files."""
    datasets_root = PROJECT_ROOT.parent / "LemGendaryDatasets"
    base_manifold = (datasets_root / "LemGendizedForexUniverse").resolve()
    if not base_manifold.exists() and (datasets_root / "LemGendizedForexUniverseLarge").exists():
        base_manifold = (datasets_root / "LemGendizedForexUniverseLarge").resolve()
    if not base_manifold.exists():
        console.print(f"[red]Base manifold directory does not exist: {base_manifold}[/red]")
        raise typer.Exit(code=1)

    years = [year] if year else (list(range(2019, 2027)) if all_ else [])
    if not years:
        console.print("[yellow]Specify --year <YYYY> or --all[/yellow]")
        raise typer.Exit(code=0)

    for yr in years:
        ok = load_forex_module("converter").convert_year(
            base_manifold, yr, dry_run=dry_run, skip_cleanup=skip_cleanup
        )
        if not ok:
            raise typer.Exit(code=1)
    raise typer.Exit(code=0)


@forex_app.command("embed")
def forex_embed() -> None:
    """Embed column descriptions and schema metadata into Forex Parquet files."""
    load_forex_module("injector").main()
