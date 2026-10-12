#!/usr/bin/env python3
"""LemGendary dataset_info.yaml Generator Script.

Serializes machine-readable dataset specification, sample counts, lineage sources,
and container topology to dataset_info.yaml for automated PyTorch loaders.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
from typing import Any
import yaml

_MODULE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from core.docs.context import ManifoldContext, resolve_manifold_context
from core.docs.scanner import scan_forex_manifold

logger = logging.getLogger(__name__)


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
    if isinstance(context_or_path, ManifoldContext):
        ctx = context_or_path
    else:
        ctx = resolve_manifold_context(Path(context_or_path), overrides=overrides)

    target_dir = ctx.manifold_path
    manifold_name = ctx.manifold_name
    relative_path = str(Path("..") / "LemGendaryDatasets" / manifold_name)

    # Resolve sources list
    src_keys = list(ctx.sources.keys()) if ctx.sources else ctx.provenance_sources
    if not src_keys:
        src_keys = [f"{manifold_name}-source"]

    # Canonical format detection
    canonical_format = ctx.unified_entry.get("canonical_format")
    if not canonical_format:
        if ctx.task == "forex":
            canonical_format = "parquet"
        elif ctx.task == "ai_helper":
            canonical_format = "jsonl"
        elif (target_dir / "mds").exists():
            canonical_format = "mds"
        else:
            canonical_format = "webdataset"

    info_payload: dict[str, Any] = {
        "name": manifold_name,
        "count": ctx.total_samples,
        "task": ctx.task,
        "category": ctx.category,
        "canonical_format": canonical_format,
        "original_sources": src_keys,
        "path": relative_path,
        "source": f"{manifold_name}-manifold",
        "last_processed": datetime.now(timezone.utc).isoformat(),
    }

    if ctx.task == "forex":
        forex_scan = scan_forex_manifold(target_dir)
        pairs = (
            (overrides.get("pairs") if overrides else None)
            or ctx.overrides.get("pairs")
            or forex_scan.get("pairs")
            or ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "NAS100", "DE40", "USOIL", "US500"]
        )
        tfs = (
            (overrides.get("timeframe_rungs") if overrides else None)
            or ctx.overrides.get("timeframe_rungs")
            or forex_scan.get("timeframes")
            or [1, 5, 15, 60, 240, 1440]
        )
        start_date = (
            (overrides.get("start_date") if overrides else None)
            or ctx.overrides.get("start_date")
            or "2019-01-01"
        )
        lookback = int(
            (overrides.get("lookback_bars") if overrides else None)
            or ctx.overrides.get("lookback_bars")
            or 168
        )

        info_payload["dataset_type"] = "forex"
        info_payload["pairs"] = [str(p) for p in pairs]
        info_payload["timeframe_rungs"] = [int(tf) for tf in tfs]
        info_payload["start_date"] = str(start_date)
        info_payload["lookback_bars"] = lookback
        info_payload["format"] = "parquet"
        info_payload["compression"] = "zstd"

    if ctx.task == "ai_helper":
        train_file = target_dir / "ai_helper_train.jsonl"
        val_file = target_dir / "ai_helper_val.jsonl"
        tr_cnt = sum(1 for _ in train_file.open(encoding="utf-8")) if train_file.exists() else 0
        vl_cnt = sum(1 for _ in val_file.open(encoding="utf-8")) if val_file.exists() else 0
        info_payload["train_samples"] = tr_cnt
        info_payload["val_samples"] = vl_cnt

    if overrides:
        for k, v in overrides.items():
            if k not in ["name", "last_processed"]:
                info_payload[k] = v

    target_file = target_dir / "dataset_info.yaml"
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


def main() -> None:
    """CLI entrypoint for dataset_info.yaml generation."""
    parser = argparse.ArgumentParser(description="Generate dataset_info.yaml for dataset manifolds")
    parser.add_argument("--manifold", type=str, default=None, help="Manifold directory name or path")
    parser.add_argument("--all", action="store_true", help="Generate dataset_info.yaml for all manifolds")
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
                out = generate_dataset_info_yaml(m_dir)
                with open(out, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                print(f"[OK] {m_dir.name} -> {out.name} (samples={data.get('count')}, task={data.get('task')})")
                count += 1
        print(f"Generated dataset_info.yaml for {count} manifolds.")
    elif args.manifold:
        p = Path(args.manifold)
        if not p.is_dir() and (datasets_root / args.manifold).is_dir():
            p = datasets_root / args.manifold
        out = generate_dataset_info_yaml(p)
        with open(out, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        print(f"[OK] {p.name} -> {out.name} (samples={data.get('count')}, task={data.get('task')})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
