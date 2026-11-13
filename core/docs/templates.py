"""Markdown template rendering engines for LemGendary Dataset Compiler.

Formats dataset overview tables, lineage matrices, model training profiles,
and physical directory manifests with strict Markdown standard compliance.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ForexCoverage",
    "ForexReadmeFields",
    "ReadmeCommon",
    "StandardOverview",
    "StandardReadmeFields",
    "StructureDescriptions",
    "build_models_markdown",
    "build_structure_text",
    "clean_readme",
    "render_forex_readme",
    "render_standard_readme",
]

_LOWER_IS_BETTER = ["loss", "lpips", "fid", "drawdown", "mae", "mse", "rank_margin", "nme"]
_METRIC_ACRONYMS = (
    ("Psnr", "PSNR"),
    ("Ssim", "SSIM"),
    ("Lpips", "LPIPS"),
    ("Fid", "FID"),
    ("Map", "mAP"),
    ("Miou", "mIoU"),
    ("Iou", "IoU"),
    ("Ap50 95", "AP50-95"),
    ("Ap50", "AP50"),
    ("Nme", "NME"),
)

_NOTEBOOK_RULES: tuple[tuple[Callable[[str], bool], str], ...] = (
    (
        lambda n: n.endswith("_colab_training.ipynb"),
        "Auto-generated Google Colab notebook for cloud training.",
    ),
    (
        lambda n: n.endswith("_kaggle_training.ipynb"),
        "Auto-generated Kaggle notebook for cloud training.",
    ),
    (
        lambda n: "_training" in n and n.endswith(".ipynb"),
        "Auto-generated Jupyter notebook for model training.",
    ),
    (
        lambda n: n.endswith(("_usage.ipynb", "-usage.ipynb")),
        "Auto-generated notebook demonstrating standalone model inference.",
    ),
)

_CONTAINER_RULES: tuple[tuple[Callable[[str], bool], str], ...] = (
    (lambda n: n.endswith(".tar"), "WebDataset POSIX `.tar` container shard."),
    (lambda n: n.endswith(".mds"), "MosaicML Streaming (`mds`) container shard."),
    (
        lambda n: n.endswith(".bin") and "chunk" in n.lower(),
        "Lightning AI LitData streaming binary chunk.",
    ),
    (lambda n: n.endswith(".cache"), "Ultralytics pre-compiled binary label cache."),
)


def wrap_bullet(text: str, width: int = 80) -> str:
    """Wrap a markdown bullet line (`- ...`) to ``width`` characters."""
    return textwrap.fill(
        text,
        width=width,
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    )


def _wrap_blockquote(text: str, width: int = 80) -> str:
    """Wrap a markdown blockquote (`> ...`) to ``width`` characters."""
    raw = text[2:].strip() if text.startswith("> ") else text.strip()
    return textwrap.fill(
        raw,
        width=width,
        initial_indent="> ",
        subsequent_indent="> ",
        break_long_words=False,
        break_on_hyphens=False,
    )


def wrap_paragraph(text: str, width: int = 80) -> str:
    """Wrap a plain markdown paragraph to ``width`` characters."""
    return textwrap.fill(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


@dataclass
class ReadmeCommon:
    """Sections shared by every README variant."""

    manifold_name: str
    models_markdown: str
    structure_text: str
    kaggle_owner: str
    """Kaggle account that hosts the dataset (linked at the end of the README)."""


@dataclass
class ForexCoverage:
    """Pairs, timeframe rungs and temporal window of a Forex manifold."""

    pairs_display: str
    tfs_display: str
    start_date_str: str
    lookback_bars: int


@dataclass
class ForexReadmeFields:
    """Values rendered into the Forex README template."""

    common: ReadmeCommon
    category_str: str
    coverage: ForexCoverage
    total_samples: int
    year_table: str


@dataclass
class StandardOverview:
    """Dataset Overview values of a vision / standard manifold."""

    resolved_desc: str
    category: str
    total_samples_display: str
    arch_base: str
    resolved_obj: str


@dataclass
class StandardReadmeFields:
    """Values rendered into the vision / standard README template."""

    common: ReadmeCommon
    overview: StandardOverview
    composition_text: str
    manifest_text: str


@dataclass
class StructureDescriptions:
    """Task-dependent descriptions of the images/targets folders."""

    targets_desc: str = "Target matrices or masks for training."
    img_desc: str = "RGB"
    tgt_desc: str = ""


def clean_readme(content: str) -> str:
    """Clean up markdown to avoid multiple consecutive blank lines and trailing whitespace."""
    lines = content.splitlines()
    cleaned: list[str] = []
    prev_empty = False
    for line in lines:
        is_empty = line.strip() == ""
        if is_empty and prev_empty:
            continue
        cleaned.append(line)
        prev_empty = is_empty
    while cleaned and cleaned[-1].strip() == "":
        cleaned.pop()
    return "\n".join(cleaned) + "\n"


def _applicable_models(manifold_name: str, models_meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the model profiles bound to the manifold (by dataset list, else by name)."""
    applicable_models = [
        m_info
        for m_info in models_meta.values()
        if isinstance(m_info, dict) and manifold_name in m_info.get("datasets", [])
    ]
    if applicable_models:
        return applicable_models
    stripped = manifold_name.lower().replace("lemgendized", "").replace("large", "")
    return [
        m_info
        for m_key, m_info in models_meta.items()
        if isinstance(m_info, dict) and m_key.lower() in stripped
    ]


def _metric_label(met: str) -> str:
    """Return the display name of a SOTA metric key."""
    label = met.replace("_", " ").title()
    for old, new in _METRIC_ACRONYMS:
        label = label.replace(old, new)
    return label


def _sota_row(met: str, val: Any) -> str:
    """Return the Baseline/Advanced/SOTA table row of one metric target."""
    met_name = _metric_label(met)
    if not isinstance(val, (int, float)):
        return f"| **{met_name}** | N/A | N/A | **{val}** |"
    met_lower = met.lower()
    if any(x in met_lower for x in _LOWER_IS_BETTER):
        return f"| **{met_name}** | < {val * 1.5:.2f} | < {val * 1.2:.2f} | **< {val}** |"
    if "psnr" in met_lower:
        return (
            f"| **{met_name}** | ~{val * 0.85:.1f} dB | > {val * 0.94:.1f} dB | **> {val:.1f}"
            " dB** |"
        )
    if "ssim" in met_lower:
        return f"| **{met_name}** | ~{val * 0.88:.4f} | > {val * 0.95:.4f} | **> {val:.4f}** |"
    is_pct = any(x in met_lower for x in ["acc", "win", "rate"]) or (20.0 < val <= 100.0)
    if is_pct and val > 10.0:
        return f"| **{met_name}** | ~{val * 0.8:.1f}% | > {val * 0.9:.1f}% | **> {val}%** |"
    return f"| **{met_name}** | ~{val * 0.8:.2f} | > {val * 0.9:.2f} | **> {val}** |"


def _model_block(model: dict[str, Any]) -> str:
    """Render the Markdown block of one model profile with its SOTA table."""
    model_name = model.get("name", "Unknown Model")
    arch_val = model.get("arch") or model.get("architecture_type") or "Standard Backbone"
    opt_val = model.get("loss", "Unknown")

    arch_bullet = wrap_bullet(f"- **Architecture**: {arch_val}")
    opt_bullet = wrap_bullet(f"- **Optimization**: {opt_val}")
    model_block = f"### Model: {model_name}\n\n{arch_bullet}\n{opt_bullet}\n"
    sota = model.get("sota_targets", {})
    if sota:
        table_lines = [
            "\n| Metric | Baseline | Advanced | SOTA |",
            "| :--- | :--- | :--- | :--- |",
        ]
        table_lines += [_sota_row(met, val) for met, val in sota.items()]
        model_block += "\n".join(table_lines) + "\n"
        if model.get("sota_targets_provisional"):
            note = model.get("sota_targets_note") or "no cited source backs these values yet."
            model_block += "\n" + wrap_bullet(f"- **Provisional targets**: {note}") + "\n"
    return model_block


def build_models_markdown(manifold_name: str, models_meta: dict[str, Any]) -> str:
    """Build markdown section detailing applicable models and their SOTA benchmark targets."""
    applicable_models = _applicable_models(manifold_name, models_meta)
    if not applicable_models:
        return "No models are explicitly bound to this dataset in models_metadata.yaml.\n"
    return "\n\n".join(_model_block(am) for am in applicable_models) + "\n"


def _first_rule(rules: tuple[tuple[Callable[[str], bool], str], ...], name: str) -> str | None:
    """Return the description of the first rule matching ``name``."""
    return next((desc for matches, desc in rules if matches(name)), None)


def _describe_structure_entry(name: str, is_forex: bool, desc_map: dict[str, str]) -> str:
    """Describe one root-level entry of the repository structure list."""
    notebook = _first_rule(_NOTEBOOK_RULES, name)
    if notebook is not None:
        return notebook
    if is_forex and name.startswith("ForexUniverse"):
        if name.endswith(".parquet"):
            return f"Year-chunked unified Parquet manifold: {name}"
        return f"Year-chunked shard directory: {name}"
    container = _first_rule(_CONTAINER_RULES, name)
    if container is not None:
        return container
    return desc_map.get(name, "Dataset component.")


def build_structure_text(
    output_root: Path,
    is_forex: bool = False,
    task_key: str = "quality",
    descriptions: StructureDescriptions | None = None,
) -> str:
    """Scan directory items and generate a markdown list describing repository components."""
    if not output_root.exists():
        return ""
    descs = descriptions or StructureDescriptions()

    desc_map: dict[str, str] = {
        "images": f"Normalized input tensors ({descs.img_desc}, standardized resolution).",
        "labels": "Strict numerical annotation vectors (JSON/TXT format).",
        "targets": (
            f"Clean ground truth tensors ({descs.tgt_desc})."
            if task_key == "restoration"
            else descs.targets_desc
        ),
        "shards": (
            "WebDataset shards per split (tar archives named `.data` or `.tar`), one"
            " member group per sample; each split's `shard_offsets.json` records where"
            " every member starts, so readers skip the tar header scan."
        ),
        "forex": "Shards containing serialized manifold data.",
        "dataset_info.yaml": "Manifest metadata for automated PyTorch loaders.",
        "dataset-metadata.json": (
            "Kaggle Frictionless metadata manifest, licensing, and schema column definitions."
        ),
        "category.txt": "Top-level categorization tag.",
        "classes.txt": "Class labels mapping.",
        "index.json": "Compiled metadata index mapping all dataset samples.",
        "README.md": "This documentation file.",
    }

    structure_lines: list[str] = []
    for item in sorted(output_root.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        name = item.name
        desc = _describe_structure_entry(name, is_forex, desc_map)
        label = f"`{name}/`" if item.is_dir() else f"`{name}`"
        structure_lines.append(wrap_bullet(f"- **{label}**: {desc}"))

    return "\n".join(structure_lines)


def render_forex_readme(fields: ForexReadmeFields) -> str:
    """Render comprehensive Markdown documentation for Forex manifold."""
    common = fields.common
    coverage = fields.coverage
    slug = common.manifold_name.lower().replace("_", "-")
    kaggle_url = f"https://www.kaggle.com/datasets/{common.kaggle_owner}/{slug}"
    horizon = (
        f"{coverage.start_date_str} to Present "
        "(6-Fold Walk-Forward Matrix with 14-day Embargo)"
    )
    quote = _wrap_blockquote(
        "> High-fidelity OHLCV temporal manifold for training multi-scale financial "
        "prediction models."
    )
    overview_bullets = "\n".join(
        [
            wrap_bullet(f"- **Category:** {fields.category_str}"),
            wrap_bullet(
                "- **Acquisition Mode:** MetaTrader 5 Terminal API / Synthetic "
                "Multi-Regime Generator"
            ),
            wrap_bullet(f"- **Pairs Included:** {coverage.pairs_display}"),
            wrap_bullet(f"- **Timeframe Rungs:** {coverage.tfs_display}"),
            wrap_bullet(f"- **Historical Horizon:** {horizon}"),
            wrap_bullet(f"- **Lookback Window:** {coverage.lookback_bars} bars"),
            wrap_bullet(f"- **Total Samples:** {fields.total_samples:,}"),
            wrap_bullet(
                "- **Output Classes:** `SELL` (0), `HOLD` (1), `BUY` (2) + Dual Pip "
                "Target Heads (TP/SL)"
            ),
            wrap_bullet(
                "- **Architecture Base:** Causal TCN + Cross-Timeframe Multi-Head Attention"
            ),
            wrap_bullet(
                "- **Primary Task:** Predict directional probability (Sell/Hold/Buy) and "
                "regress optimal Take-Profit/Stop-Loss boundaries."
            ),
        ]
    )
    shard_intro = wrap_paragraph(
        "The dataset is organised by year into unified Apache Parquet files "
        "(`ForexUniverseYYYY.parquet`), each containing all pairs and timeframes "
        "for that year with Zstandard compression."
    )
    repo_intro = wrap_paragraph(
        "Standardized directory logic for seamless integration into the "
        "**LemGendary Training Suite**."
    )
    kaggle_line = (
        "**Kaggle Native Source**: [Access Dataset][kaggle-source]\n\n"
        f"[kaggle-source]:\n  {kaggle_url}"
    )
    content = f"""# {common.manifold_name}

{quote}

## Dataset Overview

{overview_bullets}

## Year-Chunked Shard Breakdown

{shard_intro}

{fields.year_table}

## Model Training Profiles

{common.models_markdown}

## Repository Structure

{repo_intro}

{common.structure_text}

---

{kaggle_line}
"""
    return clean_readme(content)


def render_standard_readme(fields: StandardReadmeFields) -> str:
    """Render comprehensive Markdown documentation for vision / standard manifold."""
    common = fields.common
    overview = fields.overview
    slug = common.manifold_name.lower().replace("_", "-")
    kaggle_url = f"https://www.kaggle.com/datasets/{common.kaggle_owner}/{slug}"
    quote = _wrap_blockquote(f"> {overview.resolved_desc}")
    overview_bullets = "\n".join(
        [
            wrap_bullet(f"- **Category:** {overview.category}"),
            wrap_bullet(f"- **Total Samples:** {overview.total_samples_display}"),
            wrap_bullet(f"- **Architecture Base:** {overview.arch_base}"),
            wrap_bullet(f"- **Primary Task:** {overview.resolved_obj}"),
        ]
    )
    repo_intro = wrap_paragraph(
        "Standardized directory logic for seamless integration into the "
        "**LemGendary Training Suite**."
    )
    kaggle_line = (
        "**Kaggle Native Source**: [Access Dataset][kaggle-source]\n\n"
        f"[kaggle-source]:\n  {kaggle_url}"
    )
    content = f"""# {common.manifold_name}

{quote}

## Dataset Overview

{overview_bullets}

## Composition & Lineage

{fields.composition_text}

## Model Training Profiles

{common.models_markdown}

## Repository Structure

{repo_intro}

{common.structure_text}

## Physical Data Manifest

| Folder | Train | Val |
| :--- | :--- | :--- |
{fields.manifest_text}

---

{kaggle_line}
"""
    return clean_readme(content)
