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
import sys
from typing import Any

_MODULE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from core.docs.context import ManifoldContext, resolve_manifold_context
from core.docs.scanner import scan_forex_manifold

logger = logging.getLogger(__name__)


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
    if isinstance(context_or_path, ManifoldContext):
        ctx = context_or_path
    else:
        ctx = resolve_manifold_context(Path(context_or_path), final_index=final_index)

    target_dir = ctx.manifold_path
    index_file = target_dir / "index.json"

    # 1. If explicit final_index provided by compilation pipeline
    if final_index is not None:
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(final_index, f, indent=2)
        return index_file

    # 2. If index.json already exists and is non-empty, validate
    if index_file.exists() and index_file.stat().st_size > 2:
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                logger.info("Validated existing index.json (%d entries)", len(data))
                return index_file
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupted index.json at %s (%s); rebuilding...", index_file, exc)

    # 3. Task-specific index reconstruction
    constructed_index: list[dict[str, Any]] = []

    if ctx.task == "ai_helper":
        # Build index from JSONL instruction datasets
        train_file = target_dir / "ai_helper_train.jsonl"
        val_file = target_dir / "ai_helper_val.jsonl"
        idx_counter = 0

        for split, filepath in [("train", train_file), ("val", val_file)]:
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f):
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            record = json.loads(line_str)
                            prompt_text = str(record.get("prompt", record.get("instruction", "")))
                            h = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
                        except Exception:
                            h = hashlib.sha256(f"{split}_{line_num}".encode("utf-8")).hexdigest()[:16]

                        constructed_index.append({
                            "id": idx_counter,
                            "name": f"AIHelper_{split}_{line_num:06d}",
                            "source": "LemGendary-AI-Documentation",
                            "task": "ai_helper",
                            "split": split,
                            "hash": h,
                            "nima_score": 1.0,
                            "caption": None,
                            "style_tag": None,
                            "cluster_id": -1,
                        })
                        idx_counter += 1

    elif ctx.task == "forex":
        # Build shard-level index for Forex Parquet files
        forex_data = scan_forex_manifold(target_dir)
        idx_counter = 0
        for entry in forex_data.get("details", []):
            constructed_index.append({
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
            })
            idx_counter += 1

    if constructed_index:
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(constructed_index, f, indent=2)

    return index_file


def main() -> None:
    """CLI entrypoint for index.json generation."""
    parser = argparse.ArgumentParser(description="Generate/Validate index.json for dataset manifolds")
    parser.add_argument("--manifold", type=str, default=None, help="Manifold directory name or path")
    parser.add_argument("--all", action="store_true", help="Process index.json for all manifolds")
    args = parser.parse_args()

    cand = _MODULE_ROOT.parent / "LemGendaryDatasets"
    datasets_root = cand if cand.exists() else _MODULE_ROOT / "LemGendaryDatasets"

    if args.all:
        if not datasets_root.exists():
            print(f"[ERROR] Datasets directory not found at {datasets_root}")
            sys.exit(1)
        count = 0
        for m_dir in sorted(datasets_root.iterdir()):
            if m_dir.is_dir() and not m_dir.name.startswith("."):
                out = generate_index_json(m_dir)
                sz = out.stat().st_size if out.exists() else 0
                print(f"[OK] {m_dir.name} -> {out.name} ({sz:,} bytes)")
                count += 1
        print(f"Processed index.json for {count} manifolds.")
    elif args.manifold:
        p = Path(args.manifold)
        if not p.is_dir() and (datasets_root / args.manifold).is_dir():
            p = datasets_root / args.manifold
        out = generate_index_json(p)
        sz = out.stat().st_size if out.exists() else 0
        print(f"[OK] {p.name} -> {out.name} ({sz:,} bytes)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
