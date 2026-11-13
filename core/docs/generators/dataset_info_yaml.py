#!/usr/bin/env python3
"""LemGendary dataset_info.yaml Generator Script.

Serializes machine-readable dataset specification, sample counts, lineage sources,
and container topology to dataset_info.yaml for automated PyTorch loaders.

The label fields a repair records from the data (:data:`RECORDED_LABEL_FIELDS`:
the label strategy, the class names, the label source tags) are not derivable
from the registry or the folder scan; a regeneration carries them over from the
existing dataset_info.yaml.
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

# Fields the label repairs write (repair masks, face-landmarks, class-labels,
# pack-loose); a regeneration keeps them.
RECORDED_LABEL_FIELDS: tuple[str, ...] = ("label_json", "classes", "landmark_source")


class _IndentedDumper(yaml.SafeDumper):
    """YAML dumper that indents sequence items under parent keys for yamllint compliance."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow=flow, indentless=False)


def write_dataset_info(target_file: Path, payload: dict[str, Any]) -> None:
    """Serialize ``payload`` as dataset_info.yaml (the one writer of this file's layout)."""
    with open(target_file, "w", encoding="utf-8", newline="\n") as f:
        yaml.dump(
            payload,
            f,
            Dumper=_IndentedDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            explicit_start=True,
            width=100,
        )


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
    """Return the train/val/test line counts of an instruction (JSONL) manifold."""
    train_file, val_file = jsonl_split_files(ctx)
    entry = ctx.spec.unified_entry
    test_file_rel = entry.get("test_file")
    test_file = (ctx.manifold_path / test_file_rel) if test_file_rel else None
    if not test_file or not test_file.exists():
        for base in (ctx.manifold_path / "data", ctx.manifold_path):
            cand = base / "ai_helper_test.jsonl"
            if cand.exists():
                test_file = cand
                break

    fields = {"train_samples": count_lines(train_file), "val_samples": count_lines(val_file)}
    if test_file and test_file.exists():
        fields["test_samples"] = count_lines(test_file)
    return fields


def _existing_info(ctx: ManifoldContext) -> dict[str, Any]:
    """The manifold's current dataset_info.yaml; empty when missing or unreadable."""
    existing = ctx.manifold_path / "dataset_info.yaml"
    if not existing.exists():
        return {}
    try:
        data = yaml.safe_load(existing.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("Cannot read %s: %s", existing, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _recorded_label_fields(existing: dict[str, Any]) -> dict[str, Any]:
    """The label fields a repair recorded in the existing dataset_info.yaml."""
    return {k: existing[k] for k in RECORDED_LABEL_FIELDS if existing.get(k) is not None}


def _task_type(ctx: ManifoldContext, overrides: dict[str, Any] | None,
               existing: dict[str, Any]) -> str | None:
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
    if existing.get("task_type"):
        return str(existing["task_type"])
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
    existing = _existing_info(ctx)

    # Resolve sources list
    src_keys = (
        list(ctx.lineage.sources.keys())
        if ctx.lineage.sources
        else ctx.lineage.provenance_sources
    )
    if not src_keys:
        src_keys = [f"{manifold_name}-source"]

    info_payload: dict[str, Any] = {
        "name": manifold_name,
        "count": ctx.lineage.total_samples,
        "task": ctx.task,
        "task_type": _task_type(ctx, overrides, existing),
        "category": ctx.profile.category,
        "canonical_format": _canonical_format(ctx),
        "original_sources": src_keys,
        "source": f"{manifold_name}-manifold",
        "last_processed": datetime.now(UTC).isoformat(),
        **_recorded_label_fields(existing),
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
    write_dataset_info(target_file, info_payload)
    return target_file


def _cli_process(manifold_dir: Path, _args: argparse.Namespace) -> str:
    """Generate dataset_info.yaml for one manifold and describe the result."""
    out = generate_dataset_info_yaml(manifold_dir)
    with open(out, encoding="utf-8") as f:
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
