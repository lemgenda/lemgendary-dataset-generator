"""Unified verification CLI command for LemGendary manifolds.

Enforces:
- Schema conformance (per-sample, members, JSON labels)
- Split integrity (zero duplicate keys across train/val/test)
- Domain integrity (visual QA / forex horizon / LLM tokens)
- Shard sidecar integrity (<shard>.data.index.json & shard_offsets.json)
- Provenance completeness (dataset_info.yaml, README.md, count matching)
- Determinism check (cryptographic manifest hash)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from commands.common import PROJECT_ROOT, console
from core.compiler_config import DATASETS_META
from core.manifold_verification import (
    ManifoldVerificationReport,
    ManifoldVerificationResult,
    verify_all_manifolds,
    verify_manifold,
)

__all__ = ["verify_cmd"]


def _format_result_table(res: ManifoldVerificationResult) -> Table:
    """Format an individual manifold verification outcome into a Rich Table."""
    status_style = "bold green" if res.passed else "bold red"
    status_label = "[PASS]" if res.passed else "[FAIL]"

    table = Table(
        title=f"Manifold Verification: {res.manifold_name} ({res.manifold_type}) {status_label}",
        title_style=status_style,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Contract Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    for chk_name, chk in res.checks.items():
        chk_style = "green" if chk.passed else "red"
        chk_mark = "OK" if chk.passed else "FAIL"
        details = chk.details
        if chk.errors:
            details += f" | Errors: {'; '.join(chk.errors[:2])}"
        if chk.warnings:
            details += f" | Warnings: {'; '.join(chk.warnings[:2])}"
        table.add_row(chk_name, f"[{chk_style}]{chk_mark}[/{chk_style}]", details)

    return table


def _resolve_target_manifold(target: str, datasets_root: Path) -> Path:
    """Resolve a manifold target argument into an existing filesystem path."""
    direct = Path(target).resolve()
    if direct.exists():
        return direct

    if target in DATASETS_META:
        pascal_name = DATASETS_META[target].get("name", target.replace("_", ""))
        full_name = f"LemGendized{pascal_name}"
        target_path = datasets_root / full_name
        if not target_path.exists():
            target_path = datasets_root / pascal_name
        if target_path.exists():
            return target_path

    console.print(f"[bold red]Cannot locate manifold: '{target}'[/bold red]")
    raise typer.Exit(code=1)


def _run_all_manifolds(
    datasets_root: Path, as_json: bool, strict: bool, limit: int
) -> None:
    """Audit and verify all manifolds present under datasets_root."""
    report: ManifoldVerificationReport = verify_all_manifolds(
        datasets_root=datasets_root,
        strict=strict,
        sample_limit=limit,
    )
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        console.print(
            f"\n[bold cyan]=== Verified {report.total_manifolds} Manifolds ===[/bold cyan]"
        )
        for m_res in report.results.values():
            console.print(_format_result_table(m_res))
            console.print()
        summary_style = "bold green" if report.passed else "bold red"
        summary_label = "[PASS]" if report.passed else "[FAIL]"
        counts = (
            f"Total: {report.total_manifolds} | "
            f"Passed: {report.passed_manifolds} | Failed: {report.failed_manifolds}"
        )
        console.print(f"[{summary_style}]{summary_label} {counts}[/{summary_style}]")

    raise typer.Exit(code=0 if report.passed else 1)


def _run_single_manifold(
    target_path: Path, as_json: bool, strict: bool, limit: int
) -> None:
    """Audit and verify an individual manifold folder."""
    res = verify_manifold(target_path, strict=strict, sample_limit=limit)
    if as_json:
        typer.echo(json.dumps(res.to_dict(), indent=2))
    else:
        console.print(_format_result_table(res))
        if res.errors:
            console.print(f"[bold red]Defects found ({len(res.errors)}):[/bold red]")
            for err in res.errors:
                console.print(f"  - [red]{err}[/red]")
        if res.warnings:
            console.print(f"[bold yellow]Warnings ({len(res.warnings)}):[/bold yellow]")
            for warn in res.warnings:
                console.print(f"  - [yellow]{warn}[/yellow]")

    raise typer.Exit(code=0 if res.passed else 1)


def verify_cmd(
    target: Annotated[
        str | None,
        typer.Option(
            "--target",
            "-t",
            "--model",
            "-m",
            "--manifold",
            "-d",
            help="Registry model key or direct path to manifold folder",
        ),
    ] = None,
    all_manifolds: Annotated[
        bool,
        typer.Option("--all", "-a", help="Verify all manifolds in LemGendaryDatasets"),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", "-j", help="Emit machine-readable JSON verification report"),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Treat any warning as a validation failure"),
    ] = False,
    sample_limit: Annotated[
        int,
        typer.Option("--sample-limit", help="Cap per-shard visual samples for preview"),
    ] = 100,
) -> None:
    """Audit and verify dataset manifold contracts (schema, splits, QA, determinism)."""
    datasets_root = (PROJECT_ROOT.parent / "LemGendaryDatasets").resolve()
    if not datasets_root.is_dir():
        datasets_root = (PROJECT_ROOT / "LemGendaryDatasets").resolve()

    if all_manifolds:
        _run_all_manifolds(datasets_root, as_json=as_json, strict=strict, limit=sample_limit)
        return

    if not target:
        console.print(
            "[bold yellow]Specify --model, --manifold, or --all. "
            "Run 'python cli.py verify --help'.[/bold yellow]"
        )
        raise typer.Exit(code=1)

    target_path = _resolve_target_manifold(target, datasets_root)
    _run_single_manifold(target_path, as_json=as_json, strict=strict, limit=sample_limit)
