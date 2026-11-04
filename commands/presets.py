"""`presets` sub-commands: canonical compiler preset profiles."""

from __future__ import annotations

import logging

import typer
from rich.table import Table

from commands.common import (
    console,
)
from core import presets

logger = logging.getLogger(__name__)
presets_app = typer.Typer(help="Inspect and manage canonical compiler preset profiles.")


# ─── presets (Phase 8) ───────────────────────────────────────────────────────
@presets_app.command("list")
def presets_list() -> None:
    """List available compiler presets with format and gate details."""
    all_p = presets.list_presets()
    table = Table(title="LemGendary Compiler Presets")
    table.add_column("Preset", style="cyan", no_wrap=True)
    table.add_column("Format", style="green")
    table.add_column("Quality", style="magenta")
    table.add_column("Vetting", style="yellow")
    table.add_column("Labeling", style="blue")
    table.add_column("Description", style="dim")

    for name, p in sorted(all_p.items()):
        table.add_row(
            name,
            p.image_format,
            f"{p.image_quality}%",
            "Enabled" if p.vetting_enabled else "Bypassed",
            "Enabled" if p.labeling_enabled else "Disabled",
            p.description,
        )
    console.print(table)
