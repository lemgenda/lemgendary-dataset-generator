#!/usr/bin/env python3
"""LemGendary dataset_info.yaml Generator Script.

Serializes machine-readable dataset specification, sample counts, lineage sources,
and container topology to dataset_info.yaml for automated PyTorch loaders.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from core.docs.context import ManifoldContext
from core.docs.generators.common import (
    count_lines,
    ensure_context,
    jsonl_split_files,
    manifold_parser,
    run_manifold_cli,
)
from core.docs.scanner import scan_forex_manifold

logger = logging.getLogger(__name__)


_DEFAULT_FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "NAS100", "DE40", "USOIL", "US500"]
_DEFAULT_TIMEFRAMES = [1, 5, 15, 60, 240, 1440]


def _canonical_format(ctx: ManifoldContext) -> str:
    """Return the declared canonical container format, or detect it from the task/layout."""
    canonical_format = ctx.spec.unified_entry.get("canonical_format")
    if canonical_format:
        return canonical_format
    if ctx.task == "forex":
        return "parquet"
    if ctx.is_llm:
        return "jsonl"
    if (ctx.manifold_path / "mds").exists():
        return "mds"
    return "webdataset"


def _forex_fields(ctx: ManifoldContext, overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Return the Forex-specific fields (pairs, rungs, start date, lookback, format)."""
    forex_scan = scan_forex_manifold(ctx.manifold_path)

    def pick(key: str, *fallbacks: Any) -> Any:
        value = (overrides.get(key) if overrides else None) or ctx.spec.overrides.get(key)
        for fallback in fallbacks:
            value = value or fallback
        return value

    pairs = pick("pairs", forex_scan.get("pairs"), _DEFAULT_FOREX_PAIRS)
    tfs = pick("timeframe_rungs", forex_scan.get("timeframes"), _DEFAULT_TIMEFRAMES)
    return {
        "dataset_type": "forex",
        "pairs": [str(p) for p in pairs],
        "timeframe_rungs": [int(tf) for tf in tfs],
        "start_date": str(pick("start_date", "2019-01-01")),
        "lookback_bars": int(pick("lookback_bars", 168)),
        "format": "parquet",
        "compression": "zstd",
    }


def _instruction_fields(ctx: ManifoldContext) -> dict[str, Any]:
    """Return the train/val line counts of an instruction (JSONL) manifold."""
    train_file, val_file = jsonl_split_files(ctx)
    return {"train_samples": count_lines(train_file), "val_samples": count_lines(val_file)}


def _task_type(ctx: ManifoldContext, overrides: dict[str, Any] | None) -> str | None:
    """Compiler task id (task_taxonomy.json) of the manifold.

    The compiler passes it as the ``task_type`` override. A docs-only
    regeneration takes it from the registry entry, else keeps the value of the
    existing dataset_info.yaml.
    """
    if overrides and overrides.get("task_type"):
        return str(overrides["task_type"])
    entry = ctx.spec.unified_entry
    registered = entry.get("task_override") or entry.get("task")
    if registered:
        return str(registered)
    existing = ctx.manifold_path / "dataset_info.yaml"
    if existing.exists():
        try:
            data = yaml.safe_load(existing.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError) as exc:
            logger.warning("Cannot read task_type from %s: %s", existing, exc)
            return None
        if isinstance(data, dict) and data.get("task_type"):
            return str(data["task_type"])
    return None


def generate_dataset_info_yaml(
    context_or_path: ManifoldContext | str | Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Generate and write dataset_info.yaml for a manifold.

    Args:
        context_or_path: Resolved ManifoldContext or directory path.
        overrides: Optional field overrides.

    Returns:
        Path to the generated dataset_info.yaml file.
    """
    ctx = ensure_context(context_or_path, overrides=overrides)
    manifold_name = ctx.manifold_name

    # Resolve sources list
    src_keys = list(ctx.lineage.sources.keys()) if ctx.lineage.sources else (
        ctx.lineage.provenance_sources
    )
    if not src_keys:
        src_keys = [f"{manifold_name}-source"]

    info_payload: dict[str, Any] = {
        "name": manifold_name,
        "count": ctx.lineage.total_samples,
        "task": ctx.task,
        "task_type": _task_type(ctx, overrides),
        "category": ctx.profile.category,
        "canonical_format": _canonical_format(ctx),
        "original_sources": src_keys,
        "path": str(Path("..") / "LemGendaryDatasets" / manifold_name),
        "source": f"{manifold_name}-manifold",
        "last_processed": datetime.now(UTC).isoformat(),
    }

    if info_payload["task_type"] is None:
        del info_payload["task_type"]
    if ctx.task == "forex":
        info_payload.update(_forex_fields(ctx, overrides))
    if ctx.is_llm:
        info_payload.update(_instruction_fields(ctx))

    if overrides:
        for k, v in overrides.items():
            if k not in ["name", "last_processed"]:
                info_payload[k] = v

    target_file = ctx.manifold_path / "dataset_info.yaml"
    with open(target_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            info_payload,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            explicit_start=True,
        )

    return target_file


def _cli_process(manifold_dir: Path, _args: argparse.Namespace) -> str:
    """Generate dataset_info.yaml for one manifold and describe the result."""
    out = generate_dataset_info_yaml(manifold_dir)
    with open(out, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return f"{out.name} (samples={data.get('count')}, task={data.get('task')})"


def main() -> None:
    """CLI entrypoint for dataset_info.yaml generation."""
    parser = manifold_parser(
        "Generate dataset_info.yaml for dataset manifolds",
        "Generate dataset_info.yaml for all manifolds",
    )
    run_manifold_cli(parser, _cli_process, "Generated dataset_info.yaml")


if __name__ == "__main__":
    main()
