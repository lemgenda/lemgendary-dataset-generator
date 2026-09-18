"""Markdown template rendering engines for LemGendary Dataset Compiler.

Formats dataset overview tables, lineage matrices, model training profiles,
and physical directory manifests with strict Markdown standard compliance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def clean_readme(content: str) -> str:
    """Clean up markdown to avoid multiple consecutive blank lines and trailing whitespace."""
    lines = content.splitlines()
    cleaned: list[str] = []
    prev_empty = False
    for line in lines:
        is_empty = (line.strip() == "")
        if is_empty and prev_empty:
            continue
        cleaned.append(line)
        prev_empty = is_empty
    while cleaned and cleaned[-1].strip() == "":
        cleaned.pop()
    return "\n".join(cleaned) + "\n"


def build_models_markdown(manifold_name: str, models_meta: dict[str, Any]) -> str:
    """Build markdown section detailing applicable models and their SOTA benchmark targets."""
    applicable_models: list[dict[str, Any]] = []
    for _m_key, m_info in models_meta.items():
        if isinstance(m_info, dict) and manifold_name in m_info.get("datasets", []):
            applicable_models.append(m_info)
    if not applicable_models:
        for m_key, m_info in models_meta.items():
            if isinstance(m_info, dict) and m_key.lower() in manifold_name.lower().replace("lemgendized", "").replace("large", ""):
                applicable_models.append(m_info)

    if not applicable_models:
        return "No models are explicitly bound to this dataset in models_metadata.yaml.\n"

    blocks: list[str] = []
    for am in applicable_models:
        model_name = am.get("name", "Unknown Model")
        arch_val = am.get("arch") or am.get("architecture_type") or "Standard Backbone"
        opt_val = am.get("loss", "Unknown")

        model_block = f"### Model: {model_name}\n\n- **Architecture**: {arch_val}\n- **Optimization**: {opt_val}\n"
        sota = am.get("sota_targets", {})
        if sota:
            table_lines = [
                "\n| Metric | Baseline | Advanced | SOTA |",
                "| :--- | :--- | :--- | :--- |",
            ]
            for met, val in sota.items():
                met_name = (
                    met.replace("_", " ")
                    .title()
                    .replace("Psnr", "PSNR")
                    .replace("Ssim", "SSIM")
                    .replace("Lpips", "LPIPS")
                    .replace("Fid", "FID")
                    .replace("Map", "mAP")
                    .replace("Miou", "mIoU")
                )
                if isinstance(val, (int, float)):
                    lower_is_better = any(
                        x in met.lower()
                        for x in ["loss", "lpips", "fid", "drawdown", "mae", "mse", "rank_margin"]
                    )
                    if lower_is_better:
                        base = val * 1.5
                        adv = val * 1.2
                        table_lines.append(f"| **{met_name}** | < {base:.2f} | < {adv:.2f} | **< {val}** |")
                    elif "psnr" in met.lower():
                        base = val * 0.85
                        adv = val * 0.94
                        table_lines.append(f"| **{met_name}** | ~{base:.1f} dB | > {adv:.1f} dB | **> {val:.1f} dB** |")
                    elif "ssim" in met.lower():
                        base = val * 0.88
                        adv = val * 0.95
                        table_lines.append(f"| **{met_name}** | ~{base:.4f} | > {adv:.4f} | **> {val:.4f}** |")
                    else:
                        is_pct = any(x in met.lower() for x in ["acc", "win", "rate"]) or (20.0 < val <= 100.0)
                        if is_pct and val > 10.0:
                            base = val * 0.8
                            adv = val * 0.9
                            table_lines.append(f"| **{met_name}** | ~{base:.1f}% | > {adv:.1f}% | **> {val}%** |")
                        else:
                            base = val * 0.8
                            adv = val * 0.9
                            table_lines.append(f"| **{met_name}** | ~{base:.2f} | > {adv:.2f} | **> {val}** |")
                else:
                    table_lines.append(f"| **{met_name}** | N/A | N/A | **{val}** |")
            model_block += "\n".join(table_lines) + "\n"
        blocks.append(model_block)

    return "\n\n".join(blocks) + "\n"


def build_structure_text(
    output_root: Path,
    is_forex: bool = False,
    task_key: str = "quality",
    targets_desc: str = "Target matrices or masks for training.",
    img_desc: str = "RGB",
    tgt_desc: str = "",
) -> str:
    """Scan directory items and generate a markdown list describing repository components."""
    if not output_root.exists():
        return ""

    desc_map: dict[str, str] = {
        "images": f"Normalized input tensors ({img_desc}, standardized resolution).",
        "labels": "Strict numerical annotation vectors (JSON/TXT format).",
        "targets": f"Clean ground truth tensors ({tgt_desc})." if task_key == "restoration" else targets_desc,
        "shards": "WebDataset `.tar` shards containing serialized manifold data.",
        "forex": "Shards containing serialized manifold data.",
        "dataset_info.yaml": "Manifest metadata for automated PyTorch loaders.",
        "dataset-metadata.json": "Kaggle Frictionless metadata manifest, licensing, and schema column definitions for Parquet feature tensors.",
        "category.txt": "Top-level categorization tag.",
        "classes.txt": "Class labels mapping.",
        "index.json": "Compiled metadata index mapping all dataset samples.",
        "README.md": "This documentation file.",
    }

    structure_lines: list[str] = []
    for item in sorted(output_root.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        name = item.name
        if name.endswith("_colab_training.ipynb"):
            desc = "Auto-generated Google Colab notebook for cloud training."
        elif "_training" in name and name.endswith(".ipynb"):
            desc = "Auto-generated Jupyter notebook for model training."
        elif name.endswith("_usage.ipynb"):
            desc = "Auto-generated notebook demonstrating standalone model inference."
        elif is_forex and name.startswith("ForexUniverse"):
            if name.endswith(".parquet"):
                desc = f"Year-chunked unified Parquet manifold: {name}"
            else:
                desc = f"Year-chunked shard directory: {name}"
        else:
            desc = desc_map.get(name, "Dataset component.")

        if item.is_dir():
            structure_lines.append(f"- **`{name}/`**: {desc}")
        else:
            structure_lines.append(f"- **`{name}`**: {desc}")

    return "\n".join(structure_lines)


def render_forex_readme(
    manifold_name: str,
    category_str: str,
    pairs_display: str,
    tfs_display: str,
    start_date_str: str,
    lookback_bars: int,
    total_samples: int,
    year_table: str,
    models_markdown: str,
    structure_text: str,
) -> str:
    """Render comprehensive Markdown documentation for Forex manifold."""
    slug = manifold_name.lower().replace("_", "-")
    content = f"""# {manifold_name}

> High-fidelity OHLCV temporal manifold for training multi-scale financial prediction models.

## Dataset Overview

- **Category:** {category_str}
- **Acquisition Mode:** MetaTrader 5 Terminal API / Synthetic Multi-Regime Generator
- **Pairs Included:** {pairs_display}
- **Timeframe Rungs:** {tfs_display}
- **Historical Horizon:** {start_date_str} to Present (6-Fold Walk-Forward Matrix with 14-day Embargo)
- **Lookback Window:** {lookback_bars} bars
- **Total Samples:** {total_samples:,}
- **Output Classes:** `SELL` (0), `HOLD` (1), `BUY` (2) + Dual Pip Target Heads (TP/SL)
- **Architecture Base:** Causal TCN + Cross-Timeframe Multi-Head Attention
- **Primary Task:** Predict directional probability (Sell/Hold/Buy) and regress optimal Take-Profit/Stop-Loss boundaries.

## Year-Chunked Shard Breakdown

The dataset is organised by year into unified Apache Parquet files (`ForexUniverseYYYY.parquet`), each containing all pairs and timeframes for that year with Zstandard compression.

{year_table}

## Model Training Profiles

{models_markdown}

## Repository Structure

Standardized directory logic for seamless integration into the **LemGendary Training Suite**.

{structure_text}

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/{slug})
"""
    return clean_readme(content)


def render_standard_readme(
    manifold_name: str,
    resolved_desc: str,
    category: str,
    total_samples_display: str,
    arch_base: str,
    resolved_obj: str,
    table_text: str,
    models_markdown: str,
    structure_text: str,
    manifest_text: str,
) -> str:
    """Render comprehensive Markdown documentation for vision / standard manifold."""
    slug = manifold_name.lower().replace("_", "-")
    content = f"""# {manifold_name}

> {resolved_desc}

## Dataset Overview

- **Category:** {category}
- **Total Samples:** {total_samples_display}
- **Architecture Base:** {arch_base}
- **Primary Task:** {resolved_obj}

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
{table_text}

## Model Training Profiles

{models_markdown}

## Repository Structure

Standardized directory logic for seamless integration into the **LemGendary Training Suite**.

{structure_text}

## Physical Data Manifest

| Folder | Train | Val |
| :--- | :--- | :--- |
{manifest_text}

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/{slug})
"""
    return clean_readme(content)
