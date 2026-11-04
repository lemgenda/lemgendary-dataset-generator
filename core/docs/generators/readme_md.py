#!/usr/bin/env python3
"""LemGendary README.md Generator Script.

Renders authoritative Markdown documentation including dataset overview,
composition lineage matrices, model benchmark profiles, and directory topologies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from core.docs.context import ManifoldContext
from core.docs.generators.common import (
    ensure_context,
    manifold_parser,
    run_manifold_cli,
)
from core.docs.metadata import MODELS_META
from core.docs.scanner import scan_forex_manifold
from core.docs.templates import (
    ForexCoverage,
    ForexReadmeFields,
    ReadmeCommon,
    StandardOverview,
    StandardReadmeFields,
    build_models_markdown,
    build_structure_text,
    clean_readme,
    render_forex_readme,
    render_standard_readme,
)
from core.kaggle_identity import default_owner

_TF_NAMES = {
    1: "M1 (1min)", 5: "M5 (5min)", 15: "M15 (15min)",
    60: "H1 (60min)", 240: "H4 (240min)", 1440: "D1 (1440min)",
}


def _year_table(details: list[dict[str, Any]]) -> str:
    """Render the per-year x timeframe sample table (or the shard listing when unscanned)."""
    if not details:
        year_table = "| Year | Shard File | Compression | Format |\n"
        year_table += "| :--- | :--- | :--- | :--- |\n"
        for y in range(2019, 2027):
            year_table += f"| {y} | ForexUniverse{y}.parquet | Zstandard | Apache Parquet |\n"
        return year_table

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
    return year_table


def _render_forex(ctx: ManifoldContext) -> str:
    """Render the README of a Forex temporal manifold."""
    forex_data = scan_forex_manifold(ctx.manifold_path)
    pairs_list = forex_data.get("pairs", [])
    tf_labels = [_TF_NAMES.get(tf, f"{tf}min") for tf in forex_data.get("timeframes", [])]
    structure_text = build_structure_text(ctx.manifold_path, is_forex=True, task_key="forex")
    return render_forex_readme(
        ForexReadmeFields(
            common=ReadmeCommon(
                manifold_name=ctx.manifold_name,
                models_markdown=build_models_markdown(ctx.manifold_name, MODELS_META).strip(),
                structure_text=structure_text.strip(),
                kaggle_owner=default_owner(),
            ),
            category_str=ctx.profile.category,
            coverage=ForexCoverage(
                pairs_display=", ".join(pairs_list) if pairs_list else "All Primary FX Pairs",
                tfs_display=", ".join(tf_labels) if tf_labels else "M1, M5, M15, H1, H4, D1",
                start_date_str="2019-01-01",
                lookback_bars=168,
            ),
            total_samples=ctx.lineage.total_samples,
            year_table=_year_table(forex_data.get("details", [])).strip(),
        )
    )


def _format_count(value: Any) -> str:
    """Format integer counts with thousands separators; other values verbatim."""
    return f"{value:,}" if isinstance(value, int) else str(value)


def _lineage_table(sources: dict[str, dict[str, Any]]) -> tuple[str, int, int]:
    """Render the Composition & Lineage rows; also return the summed train/val counts."""
    table_rows: list[str] = []
    total_train_all = 0
    total_val_all = 0
    for src, counts in sorted(
        sources.items(),
        key=lambda x: (x[1].get("total", 0) if isinstance(x[1].get("total"), int) else 0),
        reverse=True,
    ):
        tr = counts.get("train", 0)
        vl = counts.get("val", 0)
        if isinstance(tr, int):
            total_train_all += tr
        if isinstance(vl, int):
            total_val_all += vl
        disp_src = src.split(" (", 1)[0] if len(src) > 36 and " (" in src else src
        if len(disp_src) > 36:
            disp_src = disp_src[:33].rstrip() + "..."
        table_rows.append(
            f"| **{disp_src}** | {_format_count(tr)} | {_format_count(vl)} |"
            f" {_format_count(counts.get('total', 0))} samples |"
        )
    table_text = "\n".join(table_rows)
    if not table_text:
        table_text = "| **Curated Production Synthesis** | N/A | N/A | Full Contribution |"
    return table_text, total_train_all, total_val_all


def _manifest_text(ctx: ManifoldContext, train_total: int, val_total: int) -> str:
    """Render the Physical Data Manifest rows (JSONL splits or image/target/label folders)."""
    if ctx.is_llm:
        entry = ctx.spec.unified_entry
        train_file_rel = entry.get("train_file", f"data/{ctx.slug}_train.jsonl")
        val_file_rel = entry.get("val_file", f"data/{ctx.slug}_val.jsonl")
        return "\n".join([
            f"| **{train_file_rel}** | {train_total:,} | - |",
            f"| **{val_file_rel}** | - | {val_total:,} |",
        ])

    target_dir = ctx.manifold_path
    folders = ["images"]
    if (target_dir / "targets").exists() or ctx.task in ["restoration", "super-resolution"]:
        folders.append("targets")
    if (target_dir / "labels").exists() or ctx.task in ["detection", "pose", "classification"]:
        folders.append("labels")
    return "\n".join(f"| **{folder}** | {train_total:,} | {val_total:,} |" for folder in folders)


def _render_standard(ctx: ManifoldContext) -> str:
    """Render the README of a vision or instruction manifold."""
    total_samples = ctx.lineage.total_samples
    table_text, train_total, val_total = _lineage_table(ctx.lineage.sources)
    if train_total == 0 and total_samples > 0:
        val_total = int(total_samples * 0.12)
        train_total = total_samples - val_total

    structure_text = build_structure_text(ctx.manifold_path, is_forex=False, task_key=ctx.task)
    return render_standard_readme(
        StandardReadmeFields(
            common=ReadmeCommon(
                manifold_name=ctx.manifold_name,
                models_markdown=build_models_markdown(ctx.manifold_name, MODELS_META).strip(),
                structure_text=structure_text.strip(),
                kaggle_owner=default_owner(),
            ),
            overview=StandardOverview(
                resolved_desc=ctx.profile.description,
                category=ctx.profile.category,
                total_samples_display=(
                    f"{total_samples:,}" if total_samples > 0 else "Curated Benchmark Split"
                ),
                arch_base=ctx.profile.arch_base,
                resolved_obj=ctx.profile.objective,
            ),
            table_text=table_text.strip(),
            manifest_text=_manifest_text(ctx, train_total, val_total).strip(),
        )
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
    ctx = ensure_context(context_or_path)
    readme_text = _render_forex(ctx) if ctx.task == "forex" else _render_standard(ctx)

    readme_clean = clean_readme(readme_text)
    readme_file = ctx.manifold_path / "README.md"
    readme_file.write_text(readme_clean, encoding="utf-8", newline="\n")
    return readme_file, readme_clean


def _cli_process(manifold_dir: Path, _args: argparse.Namespace) -> str:
    """Generate README.md for one manifold and describe the result."""
    out, _ = generate_readme_md(manifold_dir)
    return f"{out.name} ({out.stat().st_size:,} bytes)"


def main() -> None:
    """CLI entrypoint for README.md generation."""
    parser = manifold_parser(
        "Generate README.md for dataset manifolds", "Generate README.md for all manifolds"
    )
    run_manifold_cli(parser, _cli_process, "Generated README.md")


if __name__ == "__main__":
    main()
