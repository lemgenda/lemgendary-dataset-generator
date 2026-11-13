"""`repair` sub-commands: fix damaged shards and stale metadata of compiled manifolds.

Every command is a dry run unless ``--apply`` is given: it prints what would
change and exits 0. With ``--apply`` each repaired file is verified and then
replaces its original atomically; no copy of the original is kept (see
:mod:`services.repair_service`, :mod:`services.label_repair_service` for
``repair labels`` and :mod:`services.txt_label_service` for
``repair txt-labels``, :mod:`services.source_label_service` for ``repair
masks``, ``repair face-landmarks`` and ``repair class-labels``,
:mod:`services.pseudo_label_service` for ``repair pseudo-labels``, which only
writes a label table outside the manifold, :mod:`services.shard_offsets_service`
for ``repair shard-offsets``, which only writes ``shard_offsets.json``). ``repair
manifold`` also verifies the labels inside the shards and exits 1 while any is a
text label or an invalid JSON label.
"""

from __future__ import annotations

import dataclasses
import os
import tarfile
from collections.abc import Callable
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
from core.source_labels import LandmarkScheme, MaskScheme
from core.source_manifolds import label_strategy_for_folder
from services.label_repair_service import (
    LabelRepairPlan,
    LabelRepairService,
    LabelSource,
    read_classes,
)
from services.loose_pack_service import LoosePackError, LoosePackPlan, LoosePackService
from services.pseudo_label_service import PseudoLabelService, PseudoPlan
from services.repair_service import ManifoldRepairPlan, RepairService
from services.sample_policy_service import SamplePolicyService
from services.shard_offsets_service import ShardOffsetsPlan, ShardOffsetsService
from services.source_label_service import SourceLabelPlan, SourceLabelService
from services.txt_label_service import TxtLabelPlan, TxtLabelService
from tools.repair.label_restore import LabelRestoreError
from tools.repair.pseudo_labels import ShardPseudoResult
from tools.repair.sample_policy_fix import SamplePolicyError, SamplePolicyPlan
from tools.repair.shard_fix import ShardFixError
from tools.repair.source_label_fix import ShardLabelResult, SourceLabelError
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
_SOURCE_LABEL_ERRORS = (OSError, tarfile.TarError, SourceLabelError, ValueError,
                        yaml.YAMLError)
_PSEUDO_ERRORS = (OSError, tarfile.TarError, SourceLabelError, ValueError, yaml.YAMLError,
                  RuntimeError)
_PROGRESS_EVERY = 10
_SECONDS_PER_HOUR = 3600.0

RepairPlan = (ManifoldRepairPlan | LabelRepairPlan | SamplePolicyPlan | TxtLabelPlan
              | LoosePackPlan | SourceLabelPlan | PseudoPlan | ShardOffsetsPlan)


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
    """Apply the registry's sample policy: drop copies, auxiliary images and the kinds
    the policy drops (no-face samples), and re-split by source file id."""
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


WorkersOpt = Annotated[
    int, typer.Option("--workers", help="Worker processes (default: CPU count minus 2)")
]


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 1) - 2)


def _progress(total: list[int]) -> Callable[[ShardLabelResult], None]:
    def _report(result: ShardLabelResult) -> None:
        total[0] += 1
        if total[0] % _PROGRESS_EVERY == 0:
            console.print(f"  ... {total[0]} shards ({result.shard.name})", markup=False,
                          highlight=False)
    return _report


ServiceBuilder = Callable[[Path], SourceLabelService]


def _source_labels(target: TargetOptions, build: ServiceBuilder, apply: bool) -> None:
    """Plan (and with ``apply`` write) the labels ``build``'s service gives one manifold."""
    manifold_dir = require_target(target)
    try:
        service = build(manifold_dir)
        service.on_shard = _progress([0])
        plan = service.plan(manifold_dir)
        _print(plan)
        if apply and plan.changed and not plan.problems:
            remaining = service.apply(plan)
            console.print("  applied", markup=False, highlight=False)
            _print(remaining)
            if remaining.changed or remaining.problems:
                raise typer.Exit(code=1)
    except _SOURCE_LABEL_ERRORS as exc:
        console.print(f"[red]Source label repair of {manifold_dir.name} failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not apply:
        console.print("[cyan]Dry run: nothing was changed. Re-run with --apply.[/cyan]")
    raise typer.Exit(code=1 if plan.problems else 0)


@repair_app.command("masks")
def repair_masks(
    masks_root: Annotated[Path, typer.Option(
        "--masks-root", help="Folder holding each source's mask folder (key_layouts files_dir)")],
    model: ModelOpt = None, manifold: ManifoldOpt = None, workers: WorkersOpt = 0,
    apply: ApplyOpt = False,
) -> None:
    """Write each photo's source class mask as <key>.mask.webp (lossless) plus its JSON label."""
    _require_dir(masks_root)
    _source_labels(TargetOptions(model=model, manifold=manifold),
                   lambda d: SourceLabelService.for_manifold(d, masks_root, MaskScheme,
                                                             workers or _default_workers()),
                   apply)


@repair_app.command("face-landmarks")
def repair_face_landmarks(
    landmarks_root: Annotated[Path, typer.Option(
        "--landmarks-root",
        help="Folder holding each source's landmark folder (key_layouts files_dir)")],
    model: ModelOpt = None, manifold: ManifoldOpt = None, workers: WorkersOpt = 0,
    apply: ApplyOpt = False,
) -> None:
    """Rebuild each photo's face box and 5 landmarks from its source landmark file."""
    _require_dir(landmarks_root)
    _source_labels(TargetOptions(model=model, manifold=manifold),
                   lambda d: SourceLabelService.for_manifold(d, landmarks_root, LandmarkScheme,
                                                             workers or _default_workers()),
                   apply)


def _require_dir(root: Path) -> None:
    if not root.is_dir():
        console.print(f"[red]Label folder not found: {root}[/red]")
        raise typer.Exit(code=1)


TableRootOpt = Annotated[Path, typer.Option(
    "--table-root", help="Folder of the pseudo-label table (repair pseudo-labels writes it)")]


@repair_app.command("class-labels")
def repair_class_labels(table_root: TableRootOpt, model: ModelOpt = None,
                        manifold: ManifoldOpt = None, workers: WorkersOpt = 0,
                        apply: ApplyOpt = False) -> None:
    """Write each sample's class (registry class_labels: key rules, pseudo labels) into its JSON."""
    _source_labels(TargetOptions(model=model, manifold=manifold),
                   lambda d: SourceLabelService.for_class_labels(d, table_root,
                                                                 workers or _default_workers()),
                   apply)


def _eta_line(results: list[ShardPseudoResult], remaining: int) -> str:
    samples = sum(r.samples for r in results)
    seconds = sum(r.seconds for r in results)
    rate = samples / seconds if seconds > 0 else 0.0
    eta = remaining / rate / _SECONDS_PER_HOUR if rate > 0 else 0.0
    return (f"  throughput {rate:.1f} samples/s over {samples} samples;"
            f" {remaining} samples left, ETA {eta:.2f} h")


def _shard_line(result: ShardPseudoResult) -> None:
    console.print(f"  {result.shard.parent.name}/{result.shard.name}: {result.samples} samples"
                  f" in {result.seconds:.0f} s", markup=False, highlight=False)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PseudoLabelOptions:
    """Options of ``repair pseudo-labels``."""

    target: TargetOptions
    table_root: TableRootOpt
    workers: WorkersOpt = 0
    max_shards: Annotated[int | None, typer.Option(
        "--max-shards", help="Label at most this many shards (measure throughput first)")] = None
    apply: ApplyOpt = False


@repair_app.command("pseudo-labels")
@options_command(PseudoLabelOptions)
def repair_pseudo_labels(opts: PseudoLabelOptions) -> None:
    """Pseudo-label the samples of the registry's pseudo rules into a table (shards untouched)."""
    manifold_dir = require_target(opts.target)
    table_root, apply = opts.table_root, opts.apply
    try:
        plan = PseudoLabelService.plan_only(manifold_dir, table_root)
        _print(plan)
        if apply and plan.remaining:
            results = PseudoLabelService.run_onnx(manifold_dir, table_root,
                                                  opts.workers or _default_workers(),
                                                  opts.max_shards, _shard_line)
            left = PseudoLabelService.plan_only(manifold_dir, table_root).remaining
            console.print(_eta_line(results, sum(s.samples for s in left)), markup=False,
                          highlight=False)
    except _PSEUDO_ERRORS as exc:
        console.print(f"[red]Pseudo labels of {manifold_dir.name} failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not apply:
        console.print("[cyan]Dry run: nothing was changed. Re-run with --apply.[/cyan]")
    raise typer.Exit(code=0)


_OFFSETS_ERRORS = (OSError, tarfile.TarError)


@repair_app.command("shard-offsets")
def repair_shard_offsets(model: ModelOpt = None, manifold: ManifoldOpt = None,
                         apply: ApplyOpt = False) -> None:
    """Write each shard folder's shard_offsets.json (member offsets); shards are not changed."""
    manifold_dir = require_target(TargetOptions(model=model, manifold=manifold))
    service = ShardOffsetsService()
    try:
        plan = service.plan(manifold_dir)
        _print(plan)
        if apply and plan.changed:
            run = service.apply(plan)
            for line in run.report():
                console.print(line, markup=False, highlight=False)
            if run.remaining.changed:
                console.print("[yellow]  still not current after writing:[/yellow]")
                _print(run.remaining)
                raise typer.Exit(code=1)
    except _OFFSETS_ERRORS as exc:
        console.print(f"[red]Shard index of {manifold_dir.name} failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not apply:
        console.print("[cyan]Dry run: nothing was changed. Re-run with --apply.[/cyan]")
    raise typer.Exit(code=0)
