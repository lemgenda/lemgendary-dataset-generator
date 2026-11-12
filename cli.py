"""
LemGendary Dataset Compiler - Unified CLI.

Phase 1.5 of the 2026 modernization roadmap, extended through Phase 8.

Thin Typer shell over the service layer and existing entry points. Each
command group lives in its own ``commands.<group>`` module; this file only
assembles them into ``app``.

Usage:
    python cli.py --help
    python cli.py compile --model nima_technical
    python cli.py env validate
    python cli.py version
"""

from __future__ import annotations

import typer

from commands.compile import audit, compile_manifold, test_datasets, transcode, version
from commands.config import config_app
from commands.container import format_app
from commands.docs import docs_app
from commands.env import env_app
from commands.forex import forex_app
from commands.generate import degrade, label, mask, prompt
from commands.llm import llm_app
from commands.modernize import modernize_app, modernize_reduce
from commands.presets import presets_app
from commands.repair import repair_app
from commands.server import server_app
from commands.sync import sync_app

app = typer.Typer(
    name="lemgendary",
    help="LemGendary Dataset Compiler Suite - unified CLI.",
    no_args_is_help=True,
    add_completion=False,
)

# Top-level commands.
app.command()(version)
app.command(name="compile")(compile_manifold)
app.command()(transcode)
app.command()(audit)
app.command()(label)
app.command()(prompt)
app.command()(mask)
app.command()(degrade)
app.command("test")(test_datasets)
app.command("reduce")(modernize_reduce)

# Command groups.
app.add_typer(env_app, name="env")
app.add_typer(sync_app, name="sync")
app.add_typer(docs_app, name="docs")
app.add_typer(config_app, name="config")
app.add_typer(format_app, name="format")
app.add_typer(presets_app, name="presets")
app.add_typer(server_app, name="server")
app.add_typer(forex_app, name="forex")
app.add_typer(modernize_app, name="modernize")
app.add_typer(llm_app, name="llm")
app.add_typer(repair_app, name="repair")


if __name__ == "__main__":
    app()
