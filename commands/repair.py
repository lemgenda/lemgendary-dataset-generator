"""`repair` sub-commands: fix damaged shards and stale metadata of compiled manifolds.

Every command is a dry run unless ``--apply`` is given: it prints what would
change and exits 0. With ``--apply`` each repaired file is verified and then
replaces its original atomically; no copy of the original is kept (see
:mod:`services.repair_service`).
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Annotated

import typer

from commands.common import console
from commands.options import ManifoldOpt, ModelOpt, TargetOptions, require_target
from core.paths import DEFAULT_OUTPUT_FOLDER, from_root
from services.repair_service import ManifoldRepairPlan, RepairService
from tools.repair.shard_fix import ShardFixError

repair_app = typer.Typer(help="Repair compiled manifolds (dry run unless --apply).")

ApplyOpt = Annotated[
    bool, typer.Option("--apply", help="Write the repairs (default: dry run, report only)")
]
RootOpt = Annotated[
    str | None,
    typer.Option("--root", help=f"Datasets root (default: {DEFAULT_OUTPUT_FOLDER})"),
]
_REPAIR_ERRORS = (OSError, tarfile.TarError, ShardFixError, ValueError)


def _print(plan: ManifoldRepairPlan) -> None:
    for line in plan.report():
        console.print(line, markup=False, highlight=False)


def _run(service: RepairService, manifold_dir: Path, apply: bool) -> bool:
    """Plan (and with ``apply`` carry out) the repair of one manifold; False on failure."""
    plan = service.plan(manifold_dir)
    _print(plan)
    if not apply or not plan.changed:
        return True
    try:
        remaining = service.apply(plan)
    except _REPAIR_ERRORS as exc:
        console.print(f"[red]Repair of {manifold_dir.name} failed: {exc}[/red]")
        return False
    console.print("  applied", markup=False, highlight=False)
    if remaining.changed:
        console.print("[yellow]  still not clean after the repair:[/yellow]")
        _print(remaining)
        return False
    return True


@repair_app.command("manifold")
def repair_manifold(model: ModelOpt = None, manifold: ManifoldOpt = None,
                    apply: ApplyOpt = False) -> None:
    """Repair one manifold: zero-byte samples, scattered groups, png/jpg members, counts, paths."""
    manifold_dir = require_target(TargetOptions(model=model, manifold=manifold))
    ok = _run(RepairService(), manifold_dir, apply)
    if not apply:
        console.print("[cyan]Dry run: nothing was changed. Re-run with --apply.[/cyan]")
    raise typer.Exit(code=0 if ok else 1)


@repair_app.command("all")
def repair_all(root: RootOpt = None, apply: ApplyOpt = False) -> None:
    """Repair every manifold folder under the datasets root."""
    datasets_root = from_root(root or DEFAULT_OUTPUT_FOLDER)
    if not datasets_root.is_dir():
        console.print(f"[red]Datasets root not found: {datasets_root}[/red]")
        raise typer.Exit(code=1)
    service = RepairService()
    manifolds = sorted(p for p in datasets_root.iterdir()
                       if p.is_dir() and not p.name.startswith("."))
    failed = [m.name for m in manifolds if not _run(service, m, apply)]
    if not apply:
        console.print("[cyan]Dry run: nothing was changed. Re-run with --apply.[/cyan]")
    if failed:
        console.print(f"[red]Repairs failed for: {', '.join(failed)}[/red]")
    raise typer.Exit(code=1 if failed else 0)
