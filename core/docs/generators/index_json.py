#!/usr/bin/env python3
"""LemGendary index.json Generator Script.

Serializes and validates compiled sample index manifests (index.json) mapping
sample IDs, shard indices, train/val splits, and provenance tags across manifolds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from core.docs.context import ManifoldContext
from core.docs.generators.common import (
    ensure_context,
    jsonl_split_files,
    manifold_parser,
    run_manifold_cli,
)
from core.docs.scanner import scan_forex_manifold

logger = logging.getLogger(__name__)


def _existing_index_is_valid(index_file: Path) -> bool:
    """True when index.json already holds a non-empty list of samples."""
    if not index_file.exists() or index_file.stat().st_size <= 2:
        return False
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            logger.info("Validated existing index.json (%d entries)", len(data))
            return True
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Corrupted index.json at %s (%s); rebuilding...", index_file, exc)
    return False


def _instruction_hash(line_str: str, split: str, line_num: int) -> str:
    """Hash the prompt of a JSONL record, or its position when it is not a JSON object."""
    try:
        record = json.loads(line_str)
    except json.JSONDecodeError as exc:
        logger.debug(
            "Line %d of %s split is not JSON (%s); hashing its position",
            line_num,
            split,
            exc,
        )
        record = None
    if isinstance(record, dict):
        prompt_text = str(record.get("prompt", record.get("instruction", "")))
        return hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
    return hashlib.sha256(f"{split}_{line_num}".encode()).hexdigest()[:16]


def _build_instruction_index(ctx: ManifoldContext) -> list[dict[str, Any]]:
    """Build the sample index of an instruction (JSONL) manifold."""
    train_file, val_file = jsonl_split_files(ctx)
    prefix = "AIHelper" if ctx.task == "ai_helper" else ctx.slug.title().replace("_", "")
    constructed_index: list[dict[str, Any]] = []
    for split, filepath in [("train", train_file), ("val", val_file)]:
        if not filepath.exists():
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line_str = line.strip()
                if not line_str:
                    continue
                constructed_index.append({
                    "id": len(constructed_index),
                    "name": f"{prefix}_{split}_{line_num:06d}",
                    "source": "LemGendary-AI-Documentation",
                    "task": ctx.task,
                    "split": split,
                    "hash": _instruction_hash(line_str, split, line_num),
                    "nima_score": 1.0,
                    "caption": None,
                    "style_tag": None,
                    "cluster_id": -1,
                })
    return constructed_index


def _build_forex_index(ctx: ManifoldContext) -> list[dict[str, Any]]:
    """Build the shard-level index of a Forex manifold."""
    forex_data = scan_forex_manifold(ctx.manifold_path)
    return [
        {
            "id": idx_counter,
            "name": f"Forex_{entry['pair']}_{entry['timeframe']}m_{entry['year']}",
            "source": "MetaTrader-5-Native-Cache",
            "task": "forex",
            "split": "train" if entry["year"] < 2025 else "val",
            "hash": None,
            "year": entry["year"],
            "pair": entry["pair"],
            "timeframe": entry["timeframe"],
            "samples": entry["count"],
        }
        for idx_counter, entry in enumerate(forex_data.get("details", []))
    ]


def generate_index_json(
    context_or_path: ManifoldContext | str | Path,
    final_index: list[dict[str, Any]] | None = None,
) -> Path:
    """Generate, serialize, or validate index.json for a manifold.

    Args:
        context_or_path: Resolved ManifoldContext or directory path.
        final_index: Optional compiled sample index list from compiler pipeline.

    Returns:
        Path to the generated index.json file.
    """
    ctx = ensure_context(context_or_path, final_index=final_index)
    index_file = ctx.manifold_path / "index.json"

    # 1. If explicit final_index provided by compilation pipeline
    if final_index is not None:
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(final_index, f, indent=2)
        return index_file

    # 2. If index.json already exists and is non-empty, validate
    if _existing_index_is_valid(index_file):
        return index_file

    # 3. Task-specific index reconstruction
    constructed_index: list[dict[str, Any]] = []
    if ctx.is_llm:
        constructed_index = _build_instruction_index(ctx)
    elif ctx.task == "forex":
        constructed_index = _build_forex_index(ctx)

    if constructed_index:
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(constructed_index, f, indent=2)

    return index_file


def _cli_process(manifold_dir: Path, _args: argparse.Namespace) -> str:
    """Generate or validate index.json for one manifold and describe the result."""
    out = generate_index_json(manifold_dir)
    sz = out.stat().st_size if out.exists() else 0
    return f"{out.name} ({sz:,} bytes)"


def main() -> None:
    """CLI entrypoint for index.json generation."""
    parser = manifold_parser(
        "Generate/Validate index.json for dataset manifolds",
        "Process index.json for all manifolds",
    )
    run_manifold_cli(parser, _cli_process, "Processed index.json")


if __name__ == "__main__":
    main()
