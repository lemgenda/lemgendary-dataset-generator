"""`config` sub-commands: configuration inspection and validation."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
import yaml
from rich.table import Table

from commands.common import (
    PROJECT_ROOT,
    console,
)
from core.config_schema import _cli_validate

logger = logging.getLogger(__name__)
config_app = typer.Typer(help="Configuration inspection and validation.")


# ─── config sub-app ─────────────────────────────────────────────────────────
@config_app.command("validate")
def config_validate() -> None:
    """Validate unified_data.yaml against the Pydantic schema."""
    raise typer.Exit(code=_cli_validate())


@config_app.command("show")
def config_show() -> None:
    """Print unified_data.yaml."""
    p = PROJECT_ROOT / "unified_data.yaml"
    if not p.exists():
        p = Path("./unified_data.yaml")
    if not p.exists():
        console.print("[red]unified_data.yaml not found[/red]")
        raise typer.Exit(code=3)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    rm = data.get("_registry_metadata", {})
    table = Table(title="Registry Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("version", str(rm.get("version", "")))
    table.add_row("name_prefix", str(rm.get("name_prefix", "")))
    table.add_row("name_suffix", repr(rm.get("name_suffix", "")))
    table.add_row("output_folder_name", str(rm.get("output_folder_name", "")))
    gc = rm.get("global_constraints", {})
    table.add_row("min_size_gb", str(gc.get("min_size_gb", "")))
    table.add_row("max_size_gb", str(gc.get("max_size_gb", "")))
    table.add_row("datasets", str(len(data.get("datasets", {}))))
    console.print(table)
