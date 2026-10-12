#!/usr/bin/env python3
"""LemGendary README.md Generator Script.

Renders authoritative Markdown documentation including dataset overview,
composition lineage matrices, model benchmark profiles, and directory topologies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_MODULE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from core.docs.context import ManifoldContext, resolve_manifold_context
from core.docs.metadata import MODELS_META
from core.docs.scanner import scan_forex_manifold
from core.docs.templates import (
    build_models_markdown,
    build_structure_text,
    clean_readme,
    render_forex_readme,
    render_standard_readme,
)


def generate_readme_md(
    context_or_path: ManifoldContext | str | Path,
) -> tuple[Path, str]:
    """Generate and write authoritative README.md for a manifold.

    Args:
        context_or_path: Resolved ManifoldContext or directory path.

    Returns:
        Tuple of (Path to README.md, full rendered string content).
    """
    if isinstance(context_or_path, ManifoldContext):
        ctx = context_or_path
    else:
        ctx = resolve_manifold_context(Path(context_or_path))

    target_dir = ctx.manifold_path
    manifold_name = ctx.manifold_name

    if ctx.task == "forex":
        forex_data = scan_forex_manifold(target_dir)
        pairs_list = forex_data.get("pairs", [])
        tfs_list = forex_data.get("timeframes", [])

        tf_names = {
            1: "M1 (1min)", 5: "M5 (5min)", 15: "M15 (15min)",
            60: "H1 (60min)", 240: "H4 (240min)", 1440: "D1 (1440min)",
        }
        tf_labels = [tf_names.get(tf, f"{tf}min") for tf in tfs_list]
        pairs_display = ", ".join(pairs_list) if pairs_list else "All Primary FX Pairs"
        tfs_display = ", ".join(tf_labels) if tf_labels else "M1, M5, M15, H1, H4, D1"

        models_markdown = build_models_markdown(manifold_name, MODELS_META)

        details = forex_data.get("details", [])
        if details:
            pivot: dict[int, dict[int, int]] = {}
            all_tfs: set[int] = set()
            for d in details:
                year_map = pivot.setdefault(int(d["year"]), {})
                tf_key = int(d["timeframe"])
                all_tfs.add(tf_key)
                year_map[tf_key] = year_map.get(tf_key, 0) + int(d["count"])
            tf_cols = sorted(all_tfs)
            year_table = "| Year | " + " | ".join(f"{tf}min" for tf in tf_cols) + " | Total |\n"
            year_table += "| :--- |" + " :--- |" * (len(tf_cols) + 1) + "\n"
            for year in sorted(pivot):
                row = pivot[year]
                cells = " | ".join(f"{row.get(tf, 0):,}" for tf in tf_cols)
                year_table += f"| {year} | {cells} | {sum(row.values()):,} |\n"
        else:
            year_table = "| Year | Shard File | Compression | Format |\n"
            year_table += "| :--- | :--- | :--- | :--- |\n"
            for y in range(2019, 2027):
                year_table += f"| {y} | ForexUniverse{y}.parquet | Zstandard | Apache Parquet |\n"

        structure_text = build_structure_text(target_dir, is_forex=True, task_key="forex")

        readme_text = render_forex_readme(
            manifold_name=manifold_name,
            category_str=ctx.category,
            pairs_display=pairs_display,
            tfs_display=tfs_display,
            start_date_str="2019-01-01",
            lookback_bars=168,
            total_samples=ctx.total_samples,
            year_table=year_table.strip(),
            models_markdown=models_markdown.strip(),
            structure_text=structure_text.strip(),
        )

    else:
        # Standard Vision or Instruction Manifold
        table_rows: list[str] = []
        total_train_all = 0
        total_val_all = 0

        for src, counts in sorted(
            ctx.sources.items(),
            key=lambda x: (x[1].get("total", 0) if isinstance(x[1].get("total"), int) else 0),
            reverse=True,
        ):
            tr = counts.get("train", 0)
            vl = counts.get("val", 0)
            tot = counts.get("total", 0)
            if isinstance(tr, int):
                total_train_all += tr
            if isinstance(vl, int):
                total_val_all += vl
            tr_str = f"{tr:,}" if isinstance(tr, int) else str(tr)
            vl_str = f"{vl:,}" if isinstance(vl, int) else str(vl)
            tot_str = f"{tot:,}" if isinstance(tot, int) else str(tot)
            table_rows.append(f"| **{src}** | {tr_str} | {vl_str} | {tot_str} samples |")

        table_text = "\n".join(table_rows)
        if not table_text:
            table_text = "| **Curated Production Synthesis** | N/A | N/A | Full Contribution |"

        total_samples_display = f"{ctx.total_samples:,}" if ctx.total_samples > 0 else "Curated Benchmark Split"

        if total_train_all == 0 and ctx.total_samples > 0:
            total_val_all = int(ctx.total_samples * 0.12)
            total_train_all = ctx.total_samples - total_val_all

        manifest_rows: list[str] = []
        if ctx.task == "ai_helper":
            manifest_rows.append(f"| **ai_helper_train.jsonl** | {total_train_all:,} | - |")
            manifest_rows.append(f"| **ai_helper_val.jsonl** | - | {total_val_all:,} |")
        else:
            manifest_rows.append(f"| **images** | {total_train_all:,} | {total_val_all:,} |")
            if (target_dir / "targets").exists() or ctx.task in ["restoration", "super-resolution"]:
                manifest_rows.append(f"| **targets** | {total_train_all:,} | {total_val_all:,} |")
            if (target_dir / "labels").exists() or ctx.task in ["detection", "pose", "classification"]:
                manifest_rows.append(f"| **labels** | {total_train_all:,} | {total_val_all:,} |")

        manifest_text = "\n".join(manifest_rows)

        structure_text = build_structure_text(
            target_dir,
            is_forex=False,
            task_key=ctx.task,
        )

        models_markdown = build_models_markdown(manifold_name, MODELS_META)

        readme_text = render_standard_readme(
            manifold_name=manifold_name,
            resolved_desc=ctx.description,
            category=ctx.category,
            total_samples_display=total_samples_display,
            arch_base=ctx.arch_base,
            resolved_obj=ctx.objective,
            table_text=table_text.strip(),
            models_markdown=models_markdown.strip(),
            structure_text=structure_text.strip(),
            manifest_text=manifest_text.strip(),
        )

    readme_clean = clean_readme(readme_text)
    readme_file = target_dir / "README.md"
    readme_file.write_text(readme_clean, encoding="utf-8")
    return readme_file, readme_clean


def main() -> None:
    """CLI entrypoint for README.md generation."""
    parser = argparse.ArgumentParser(description="Generate README.md for dataset manifolds")
    parser.add_argument("--manifold", type=str, default=None, help="Manifold directory name or path")
    parser.add_argument("--all", action="store_true", help="Generate README.md for all manifolds")
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
                out, _ = generate_readme_md(m_dir)
                print(f"[OK] {m_dir.name} -> {out.name} ({out.stat().st_size:,} bytes)")
                count += 1
        print(f"Generated README.md for {count} manifolds.")
    elif args.manifold:
        p = Path(args.manifold)
        if not p.is_dir() and (datasets_root / args.manifold).is_dir():
            p = datasets_root / args.manifold
        out, _ = generate_readme_md(p)
        print(f"[OK] {p.name} -> {out.name} ({out.stat().st_size:,} bytes)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
