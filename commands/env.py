"""`env` sub-commands: delegate code/infra operations to lem-env."""

from __future__ import annotations

import logging

import typer

from commands.common import (
    _run,
    console,
)
from core.cli_args import PROJECT_NAME, resolve_lem_env

logger = logging.getLogger(__name__)
env_app = typer.Typer(help="Delegate code/infra operations to lem-env (Environment Manager).")


# ─── env sub-app ────────────────────────────────────────────────────────────
@env_app.command("validate")
def env_validate() -> None:
    """Delegate code validation to lem-env (py_compile, lint, YAML, JSON, WCAG)."""
    try:
        lem_env = resolve_lem_env()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=3)
    raise typer.Exit(code=_run([lem_env, "validate", "--project", PROJECT_NAME]))


@env_app.command("status")
def env_status() -> None:
    """Delegate ecosystem health audit to lem-env (fast mode)."""
    try:
        lem_env = resolve_lem_env()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=3)
    raise typer.Exit(code=_run([lem_env, "audit", "--fast"]))


@env_app.command("install")
def env_install(
    no_clean: bool = typer.Option(False, "--no-clean", help="Do not purge .venv first"),
) -> None:
    """Delegate environment provisioning to lem-env."""
    try:
        lem_env = resolve_lem_env()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=3)
    cmd = [lem_env, "install", "--project", PROJECT_NAME]
    if no_clean:
        cmd += ["--no-clean"]
    raise typer.Exit(code=_run(cmd))
