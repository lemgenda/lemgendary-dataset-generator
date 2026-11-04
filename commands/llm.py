"""`llm` sub-commands: LLM instruction dataset compilation and validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import requests
import typer

from api.auth import get_or_create_token
from commands.common import (
    DEFAULT_JOB_TIMEOUT_S,
    _is_server_available,
    _wait_for_sidecar_job,
    console,
    sidecar_url,
)
from services.llm_service import LLM_COMPILE_ERRORS, compile_llm_manifold, summary_lines
from tools.llm_helper.quality_validator import QualityValidator

logger = logging.getLogger(__name__)
llm_app = typer.Typer(help="LLM instruction dataset compilation and quality invariant validation.")


@llm_app.command("compile")
def llm_compile(
    dataset: str = typer.Option(
        ..., "--dataset", "-d", help="Dataset key declared in unified_data.yaml (e.g. test_llm)"
    ),
    no_server: bool = typer.Option(
        False, "--no-server", help="Bypass API server and execute in-process"
    ),
    timeout: float = typer.Option(
        DEFAULT_JOB_TIMEOUT_S,
        "--timeout",
        min=1.0,
        help="Seconds to wait for the sidecar job before giving up (exit 1)",
    ),
) -> None:
    """Compile an LLM instruction manifold using BaseLLMCompiler and unified_data.yaml."""
    if not no_server and _is_server_available():
        token = get_or_create_token()
        payload = {"dataset_key": dataset, "async_mode": True}
        try:
            resp = requests.post(
                sidecar_url("/api/llm/compile"),
                json=payload,
                headers={"X-API-Key": token},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                job_id = data.get("job_id")
                if not job_id:
                    console.print(
                        "[bold red]Sidecar accepted the request but returned no job id:[/bold red]"
                        f" {data}"
                    )
                    raise typer.Exit(1)
                console.print(
                    f"[green]Enqueued LLM compile job {job_id} on sidecar daemon.[/green]"
                )
                raise typer.Exit(code=_wait_for_sidecar_job(job_id, token, timeout))
            console.print(
                f"[yellow]API returned {resp.status_code}. Executing in-process...[/yellow]"
            )
        except requests.RequestException as e:
            console.print(f"[yellow]Could not route to API ({e}). Executing in-process...[/yellow]")

    console.print(f"[bold cyan]Compiling LLM manifold '{dataset}'...[/bold cyan]")
    try:
        res = compile_llm_manifold(dataset)
    except LLM_COMPILE_ERRORS as exc:
        logger.error("LLM compilation failed for dataset %s: %s", dataset, exc)
        console.print(f"[bold red]LLM compilation error:[/bold red] {exc}")
        console.print_exception()
        raise typer.Exit(1) from exc
    console.print(f"[bold green]Successfully compiled {dataset}:[/bold green]")
    for line in summary_lines(res):
        console.print(line)


@llm_app.command("validate")
def llm_validate(
    file_path: Annotated[
        Path, typer.Option("--file", "-f", help="Path to .jsonl dataset file to validate")
    ],
) -> None:
    """Run QualityValidator on an LLM dataset JSONL file for 4 quality invariants."""
    if not file_path.exists():
        console.print(f"[bold red]File not found:[/bold red] {file_path}")
        raise typer.Exit(1)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Could not read {file_path}:[/bold red] {exc}")
        raise typer.Exit(1)

    console.print(
        f"[cyan]Loaded {len(records)} records from {file_path.name}. Running"
        " QualityValidator...[/cyan]"
    )
    is_valid, issues = QualityValidator().validate_corpus(records)
    errors = [iss for iss in issues if iss.severity == "ERROR"]
    warnings = [iss for iss in issues if iss.severity == "WARNING"]
    if is_valid:
        console.print(
            f"[bold green][PASS] Quality verification passed with 0 errors "
            f"and {len(warnings)} warnings across {len(records)} records.[/bold green]"
        )
        return

    console.print(
        f"[bold red][FAIL] Validation failed with {len(errors)} errors and {len(warnings)}"
        " warnings.[/bold red]"
    )
    shown = 20
    for iss in (errors + warnings)[:shown]:
        console.print(
            f"  - {iss.severity} [{iss.field}] ({iss.record_id}): {iss.issue}",
            markup=False,
            highlight=False,
        )
    if len(issues) > shown:
        console.print(f"  ... and {len(issues) - shown} more issues.")
    raise typer.Exit(1)
