"""`repair` sub-commands: fix damaged shards and stale metadata of compiled manifolds.

Every command is a dry run unless ``--apply`` is given: it prints what would
change and exits 0. With ``--apply`` each repaired file is verified and then
replaces its original atomically; no copy of the original is kept (see
:mod:`services.repair_service`, :mod:`services.label_repair_service` for
``repair labels`` and :mod:`services.txt_label_service` for
``repair txt-labels``). ``repair manifold`` also verifies the labels inside
the shards and exits 1 while any is a text label or an invalid JSON label.
"""

from __future__ import annotations

import dataclasses
import tarfile
from pathlib import Path
from typing import Annotated

import typer
import yaml

from commands.common import console
from commands.options import (
    ManifoldOpt,
    ModelOpt,
    TargetOptions,
    options_command,
    require_target,
)
from core.paths import DEFAULT_OUTPUT_FOLDER, from_root
from core.source_manifolds import label_strategy_for_folder
from services.label_repair_service import (
    LabelRepairPlan,
    LabelRepairService,
    LabelSource,
    read_classes,
)
from services.loose_pack_service import LoosePackError, LoosePackPlan, LoosePackService
from services.repair_service import ManifoldRepairPlan, RepairService
from services.sample_policy_service import SamplePolicyService
from services.txt_label_service import TxtLabelPlan, TxtLabelService
from tools.repair.label_restore import LabelRestoreError
from tools.repair.sample_policy_fix import SamplePolicyError, SamplePolicyPlan
from tools.repair.shard_fix import ShardFixError
from tools.repair.txt_label_fix import TxtLabelError
from utils.label_json import LABEL_JSON_BUILDERS

repair_app = typer.Typer(help="Repair compiled manifolds (dry run unless --apply).")

ApplyOpt = Annotated[
    bool, typer.Option("--apply", help="Write the repairs (default: dry run, report only)")
]
RootOpt = Annotated[
    str | None,
    typer.Option("--root", help=f"Datasets root (default: {DEFAULT_OUTPUT_FOLDER})"),
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class LabelRepairOptions:
    """Options of ``repair labels``."""

    target: TargetOptions
    labels: Annotated[
        Path,
        typer.Option("--labels", help="Local copy of the source's labels/ folder (train/, val/)"),
    ]
    strategy: Annotated[
        str | None,
        typer.Option("--strategy", help="label_json strategy (default: the registry entry's)"),
    ] = None
    classes: Annotated[
        Path | None,
        typer.Option("--classes", help="classes.txt to record (default: next to --labels)"),
    ] = None
    apply: ApplyOpt = False


_REPAIR_ERRORS = (OSError, tarfile.TarError, ShardFixError, ValueError)
_LABEL_ERRORS = (OSError, tarfile.TarError, LabelRestoreError, ValueError, UnicodeDecodeError)
_CLASSES_FILE = "classes.txt"
_POLICY_ERRORS = (OSError, tarfile.TarError, SamplePolicyError, ValueError, yaml.YAMLError)
_TXT_ERRORS = (OSError, tarfile.TarError, TxtLabelError)
_PACK_ERRORS = (OSError, tarfile.TarError, LoosePackError, ValueError, yaml.YAMLError)

RepairPlan = (ManifoldRepairPlan | LabelRepairPlan | SamplePolicyPlan | TxtLabelPlan
              | LoosePackPlan)


def _print(plan: RepairPlan) -> None:
    for line in plan.report():
        console.print(line, markup=False, highlight=False)


def _run(service: RepairService, manifold_dir: Path, apply: bool) -> bool:
    """Plan (and with ``apply`` carry out) the repair of one manifold; False on failure."""
    plan = service.plan(manifold_dir)
    _print(plan)
    if not apply or not plan.changed:
        return not plan.label_problems
    try:
        remaining = service.apply(plan)
    except _REPAIR_ERRORS as exc:
        console.print(f"[red]Repair of {manifold_dir.name} failed: {exc}[/red]")
        return False
    console.print("  applied", markup=False, highlight=False)
    if remaining.changed or remaining.label_problems:
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


def _label_strategy(manifold_dir: Path, strategy: str | None) -> str:
    """``strategy``, else the registry entry's label strategy; exit 1 when neither is known."""
    try:
        chosen = strategy or label_strategy_for_folder(manifold_dir.name)
    except (OSError, yaml.YAMLError) as exc:
        console.print(f"[red]Cannot read the registry: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    if chosen not in LABEL_JSON_BUILDERS:
        console.print(f"[red]No usable label strategy for {manifold_dir.name} ({chosen});"
                      f" pass --strategy, one of {sorted(LABEL_JSON_BUILDERS)}[/red]")
        raise typer.Exit(code=1)
    return chosen


def _label_source(manifold_dir: Path, opts: LabelRepairOptions) -> LabelSource:
    if not opts.labels.is_dir():
        console.print(f"[red]Labels folder not found: {opts.labels}[/red]")
        raise typer.Exit(code=1)
    return LabelSource(labels_dir=opts.labels,
                       strategy=_label_strategy(manifold_dir, opts.strategy),
                       classes=read_classes(opts.classes or opts.labels.parent / _CLASSES_FILE))


@repair_app.command("labels")
@options_command(LabelRepairOptions)
def repair_labels(opts: LabelRepairOptions) -> None:
    """Write labels from a loose labels/ folder into the shards' JSON sidecars (no recompile)."""
    manifold_dir = require_target(opts.target)
    service = LabelRepairService(_label_source(manifold_dir, opts))
    apply = opts.apply
    try:
        plan = service.plan(manifold_dir)
        _print(plan)
        if apply and plan.changed:
            remaining = service.apply(plan)
            console.print("  applied", markup=False, highlight=False)
            if remaining.changed:
                console.print("[yellow]  still not clean after the repair:[/yellow]")
                _print(remaining)
                raise typer.Exit(code=1)
    except _LABEL_ERRORS as exc:
        console.print(f"[red]Label repair of {manifold_dir.name} failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not apply:
        console.print("[cyan]Dry run: nothing was changed. Re-run with --apply.[/cyan]")
    raise typer.Exit(code=1 if plan.mismatched else 0)


StrategyOpt = Annotated[
    str | None,
    typer.Option("--strategy", help="Label strategy (default: the registry entry's)"),
]


@repair_app.command("txt-labels")
def repair_txt_labels(model: ModelOpt = None, manifold: ManifoldOpt = None,
                      strategy: StrategyOpt = None, apply: ApplyOpt = False) -> None:
    """Convert the shards' .txt label members into their JSON labels and remove them."""
    manifold_dir = require_target(TargetOptions(model=model, manifold=manifold))
    service = TxtLabelService(_label_strategy(manifold_dir, strategy))
    try:
        plan = service.plan(manifold_dir)
        _print(plan)
        if apply and plan.changed:
            remaining = service.apply(plan)
            console.print("  applied", markup=False, highlight=False)
            if remaining.changed:
                console.print("[yellow]  still not clean after the repair:[/yellow]")
                _print(remaining)
                raise typer.Exit(code=1)
    except _TXT_ERRORS as exc:
        console.print(f"[red]Text label repair of {manifold_dir.name} failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not apply:
        console.print("[cyan]Dry run: nothing was changed. Re-run with --apply.[/cyan]")
    raise typer.Exit(code=1 if plan.mismatched or plan.unreadable else 0)


@repair_app.command("pack-loose")
def repair_pack_loose(model: ModelOpt = None, manifold: ManifoldOpt = None,
                      strategy: StrategyOpt = None, apply: ApplyOpt = False) -> None:
    """Pack a loose images/labels manifold into JSON-labelled shards, then drop the loose files."""
    manifold_dir = require_target(TargetOptions(model=model, manifold=manifold))
    service = LoosePackService(_label_strategy(manifold_dir, strategy))
    try:
        plan = service.plan(manifold_dir)
        _print(plan)
        if apply and plan.changed:
            for line in service.apply(plan):
                console.print(line, markup=False, highlight=False)
            console.print("  applied", markup=False, highlight=False)
    except _PACK_ERRORS as exc:
        console.print(f"[red]Packing {manifold_dir.name} failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not apply:
        console.print("[cyan]Dry run: nothing was changed. Re-run with --apply.[/cyan]")
    raise typer.Exit(code=1 if plan.blockers else 0)


def _policy_service(manifold_dir: Path) -> SamplePolicyService:
    try:
        service = SamplePolicyService.for_manifold(manifold_dir)
    except (OSError, yaml.YAMLError) as exc:
        console.print(f"[red]Cannot read the registry: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    if service is None:
        console.print(f"[red]{manifold_dir.name}: the registry entry has no sample policy"
                      " (label_json or task)[/red]")
        raise typer.Exit(code=1)
    return service


@repair_app.command("sample-policy")
def repair_sample_policy(model: ModelOpt = None, manifold: ManifoldOpt = None,
                         apply: ApplyOpt = False) -> None:
    """Apply the registry's sample policy: drop copies, auxiliary images and no-face
    samples, and re-split so val holds landmark-bearing photos."""
    manifold_dir = require_target(TargetOptions(model=model, manifold=manifold))
    service = _policy_service(manifold_dir)
    try:
        plan = service.plan(manifold_dir)
        _print(plan)
        if apply and plan.changed:
            remaining = service.apply(plan)
            console.print("  applied", markup=False, highlight=False)
            if remaining.changed:
                console.print("[yellow]  still not clean after the repair:[/yellow]")
                _print(remaining)
                raise typer.Exit(code=1)
    except _POLICY_ERRORS as exc:
        console.print(f"[red]Sample policy repair of {manifold_dir.name} failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not apply:
        console.print("[cyan]Dry run: nothing was changed. Re-run with --apply.[/cyan]")
    raise typer.Exit(code=0)
