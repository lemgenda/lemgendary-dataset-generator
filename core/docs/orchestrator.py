"""LemGendary Dataset Documentation Orchestrator.

Coordinates metadata extraction, manifold scanning, manifest serialization,
and Markdown documentation generation for vision and temporal datasets.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any
import yaml

from core.docs.manifests import (
    write_category_txt,
    write_classes_txt,
    write_dataset_info_yaml,
    write_index_json,
    write_kaggle_metadata,
)
from core.docs.metadata import (
    FOREX_COLUMN_FIELDS,
    MANIFEST_CACHE,
    MANIFOLD_TASK_MAP,
    MODELS_META,
    TASK_ARCH_BASE,
    TASK_META,
    UNIFIED_DATA,
    format_source,
)
from core.docs.scanner import scan_forex_manifold
from core.docs.templates import (
    build_models_markdown,
    build_structure_text,
    clean_readme,
    render_forex_readme,
    render_standard_readme,
)


def generate_dataset_docs(
    output_root: str | Path,
    final_index: list[dict[str, Any]] | None = None,
    pascal_name: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> int:
    """Generate all standard documentation and manifest files for a dataset manifold.

    Creates:
    - index.json (if final_index provided)
    - dataset_info.yaml
    - category.txt
    - classes.txt
    - README.md
    - dataset-metadata.json (Kaggle Frictionless metadata)

    Returns:
        int: Total number of samples in the manifold.
    """
    output_root = Path(output_root)
    manifold_name = output_root.name
    if not pascal_name:
        pascal_name = manifold_name

    task: str | None = None
    yaml_path = output_root / "dataset_info.yaml"
    existing_info: dict[str, Any] = {}
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                existing_info = yaml.safe_load(f) or {}
                cached_t = existing_info.get("task", existing_info.get("dataset_type"))
                if cached_t and cached_t != "quality":
                    task = str(cached_t)
        except Exception as exc:
            print(f"[DEBUG] Could not read dataset_info.yaml at {yaml_path}: {exc}")

    if overrides and overrides.get("dataset_type"):
        task = str(overrides["dataset_type"])
    elif overrides and overrides.get("task"):
        task = str(overrides["task"])
    elif final_index and len(final_index) > 0:
        task = str(final_index[0].get("task", task))

    if manifold_name in MANIFOLD_TASK_MAP:
        task = MANIFOLD_TASK_MAP[manifold_name]
    elif not task:
        name_lower = manifold_name.lower()
        if "forex" in name_lower:
            task = "forex"
        elif "authenticity" in name_lower:
            task = "authenticity"
        elif "restoration" in name_lower or any(
            x in name_lower
            for x in ["ffanet", "mirnet", "mprnet", "nafnet", "film", "codeformer"]
        ):
            task = "restoration"
        elif "ultrazoom" in name_lower or "superresolution" in name_lower:
            task = "super-resolution"
        elif "parsenet" in name_lower or "segmentation" in name_lower:
            task = "segmentation"
        elif "retinaface" in name_lower or "yolo" in name_lower or "detection" in name_lower:
            task = "detection"
        elif "upn" in name_lower or "parameter" in name_lower:
            task = "parameter_prediction"
        elif "classification" in name_lower or "nsfw" in name_lower:
            task = "classification"
        else:
            task = "quality"

    task_key = task

    # 1. index.json
    write_index_json(output_root, final_index)

    # 2. Count samples & sources
    sources: dict[str, dict[str, Any]] = {}
    total_samples: int = 0

    if final_index and len(final_index) > 0:
        total_samples = len(final_index)
        for item in final_index:
            actual_src = item.get("source", "")
            if not actual_src or actual_src.lower() in ["old", "unknown", "none", "legacy"]:
                name_parts = item.get("name", "").split("_")
                if len(name_parts) >= 3:
                    actual_src = "_".join(name_parts[1:-1])
                else:
                    actual_src = actual_src if actual_src else "Unknown"
            src = format_source(actual_src)
            if src not in sources:
                sources[src] = {"train": 0, "val": 0, "total": 0}
            sources[src]["total"] += 1
            split = item.get("split", "unknown")
            if split in ["train", "val"]:
                sources[src][split] += 1
    elif task_key == "forex":
        forex_data = scan_forex_manifold(output_root)
        total_samples = sum(forex_data.get("samples_per_year", {}).values())
        sources = {
            "MetaTrader 5 Native Cache": {
                "train": "N/A",
                "val": "N/A",
                "total": total_samples,
            }
        }
        existing_info["forex_scan"] = forex_data
        if not existing_info.get("pairs"):
            existing_info["pairs"] = list(forex_data.get("pairs", []))
        if not existing_info.get("timeframe_rungs"):
            existing_info["timeframe_rungs"] = list(forex_data.get("timeframes", []))
        if not existing_info.get("start_date"):
            existing_info["start_date"] = "2019-01-01"
        if not existing_info.get("lookback_bars"):
            existing_info["lookback_bars"] = 168
        if not existing_info.get("category"):
            existing_info["category"] = "Forex & Financial Time-Series"
        if overrides:
            existing_info.update(overrides)
    elif manifold_name in MANIFEST_CACHE:
        cache_entry = MANIFEST_CACHE[manifold_name]
        total_samples = cache_entry.get("total_samples", 0)
        cached_sources = cache_entry.get("sources", {})
        for raw_src, c_info in cached_sources.items():
            fmt = format_source(raw_src)
            sources[fmt] = {
                "train": c_info.get("train", 0),
                "val": c_info.get("val", 0),
                "total": c_info.get("total", 0),
            }
        if "task" in cache_entry:
            task_key = str(cache_entry["task"])
    elif (output_root / "index.json").exists():
        try:
            with open(output_root / "index.json", "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            total_samples = len(idx_data)
            for item in idx_data:
                actual_src = item.get("source", "")
                src = format_source(actual_src)
                if src not in sources:
                    sources[src] = {"train": 0, "val": 0, "total": 0}
                sources[src]["total"] += 1
                split = item.get("split", "unknown")
                if split in ["train", "val"]:
                    sources[src][split] += 1
        except Exception as exc:
            print(f"[DEBUG] Error tallying index sources: {exc}")
    elif existing_info and "count" in existing_info:
        total_samples = existing_info.get("count", 0)
        orig_sources = existing_info.get("original_sources", [])
        if orig_sources and total_samples > 0:
            per_src = total_samples // len(orig_sources)
            val_pct = 0.12
            for s in orig_sources:
                fmt = format_source(s)
                val_c = int(per_src * val_pct)
                train_c = per_src - val_c
                sources[fmt] = {"train": train_c, "val": val_c, "total": per_src}

    # 3. dataset_info.yaml
    src_keys = list(sources.keys()) if sources else [f"{pascal_name}-source"]
    relative_path = str(Path("..") / "LemGendaryDatasets" / output_root.name)

    info_fields: dict[str, Any] = {
        "count": total_samples if isinstance(total_samples, int) else 0,
        "task": task_key,
        "original_sources": src_keys,
        "path": relative_path,
        "source": f"{pascal_name}-manifold",
        "last_processed": datetime.now().isoformat(),
    }

    if task_key == "forex":
        forex_scan_local = existing_info.get("forex_scan", {})
        if not isinstance(forex_scan_local, dict):
            forex_scan_local = {}

        pairs = (
            existing_info.get("pairs")
            or (overrides.get("pairs") if overrides else None)
            or forex_scan_local.get("pairs", [])
        )
        tfs = (
            existing_info.get("timeframe_rungs")
            or (overrides.get("timeframe_rungs") if overrides else None)
            or forex_scan_local.get("timeframes", [])
        )
        start_date_val = existing_info.get("start_date")
        if start_date_val is None and overrides:
            start_date_val = overrides.get("start_date")
        start_date_str_val = str(start_date_val) if start_date_val is not None else "2019-01-01"

        lookback_val = existing_info.get("lookback_bars")
        if lookback_val is None and overrides:
            lookback_val = overrides.get("lookback_bars")
        lookback_bars_int = int(lookback_val) if lookback_val is not None else 168

        category_val = existing_info.get("category")
        if category_val is None and overrides:
            category_val = overrides.get("category")
        category_name = str(category_val) if category_val is not None else "Forex & Financial Time-Series"

        info_fields["name"] = pascal_name
        info_fields["dataset_type"] = "forex"
        info_fields["category"] = category_name
        info_fields["pairs"] = [str(p) for p in pairs] if pairs else []
        info_fields["timeframe_rungs"] = [int(tf) for tf in tfs] if tfs else []
        info_fields["start_date"] = start_date_str_val
        info_fields["lookback_bars"] = lookback_bars_int
        info_fields["format"] = "parquet"
        info_fields["compression"] = "zstd"

    write_dataset_info_yaml(output_root, info_fields)

    # 4. category.txt
    cat_str = "General Dataset"
    if task_key in TASK_META:
        cat_str = TASK_META[task_key].get("category", "General Dataset")
    elif "detection" in TASK_META:
        cat_str = TASK_META["detection"].get("category", "General Dataset")
    if overrides and overrides.get("category"):
        cat_str = overrides["category"]
    write_category_txt(output_root, cat_str)

    # 5. classes.txt
    write_classes_txt(output_root, task_key)

    # 6. README.md
    m = TASK_META.get(task_key, {})
    if not isinstance(m, dict):
        m = {}

    resolved_desc = m.get("desc", "Dataset manifold.")
    resolved_obj = m.get("obj", "Dataset objective.")
    img_desc = "RGB"
    tgt_desc = ""

    if task_key == "restoration":
        name_lower = manifold_name.lower()
        if "dehazing" in name_lower or "indoor" in name_lower or "outdoor" in name_lower:
            task_noun = "dehazing"
            img_desc = "Hazy RGB images"
            tgt_desc = "Haze-free reference images"
            resolved_obj = "Remove haze from images and restore visual quality."
        elif "deraining" in name_lower:
            task_noun = "deraining"
            img_desc = "Rainy RGB images"
            tgt_desc = "Rain-free reference images"
            resolved_obj = "Remove rain streaks from images and restore visual quality."
        elif "deblurring" in name_lower or "debluring" in name_lower:
            task_noun = "deblurring"
            img_desc = "Blurry RGB images"
            tgt_desc = "Blur-free reference images"
            resolved_obj = "Remove blur from images and restore visual sharpness."
        elif "denoising" in name_lower:
            task_noun = "denoising"
            img_desc = "Noisy RGB images"
            tgt_desc = "Noise-free reference images"
            resolved_obj = "Remove noise from images and restore visual quality."
        elif "exposure" in name_lower or "lowlight" in name_lower:
            task_noun = "exposure correction and low-light enhancement"
            img_desc = "Under/over-exposed RGB images"
            tgt_desc = "Properly exposed reference images"
            resolved_obj = "Correct under/over-exposed images and enhance visual quality."
        elif "film" in name_lower:
            task_noun = "old film restoration"
            img_desc = "Degraded film frame RGB images"
            tgt_desc = "Restored film frame reference images"
            resolved_obj = "Restore degraded vintage film frames (scratches, noise, color fade)."
        else:
            task_noun = "restoration"
            img_desc = "Degraded RGB images"
            tgt_desc = "Clean reference images"
            resolved_obj = "Restore degraded images and enhance visual quality."
        resolved_desc = f"Standardized dataset for image {task_noun} models."

    if task_key == "forex":
        forex_scan = existing_info.get("forex_scan", {})
        if not isinstance(forex_scan, dict):
            forex_scan = {}

        pairs_raw = existing_info.get("pairs", []) or (overrides.get("pairs") if overrides else [])
        if not pairs_raw and isinstance(forex_scan, dict):
            pairs_raw = forex_scan.get("pairs", [])
        pairs_list = [str(p) for p in pairs_raw] if isinstance(pairs_raw, list) else []

        tfs_raw = existing_info.get("timeframe_rungs", []) or (overrides.get("timeframe_rungs") if overrides else [])
        if not tfs_raw and isinstance(forex_scan, dict):
            tfs_raw = forex_scan.get("timeframes", [])
        tfs_list = [int(tf) for tf in tfs_raw] if isinstance(tfs_raw, list) else []

        start_date_raw = existing_info.get("start_date")
        if start_date_raw is None and overrides:
            start_date_raw = overrides.get("start_date")
        start_date_str = str(start_date_raw) if start_date_raw is not None else "2019-01-01"

        lookback_raw = existing_info.get("lookback_bars")
        if lookback_raw is None and overrides:
            lookback_raw = overrides.get("lookback_bars")
        lookback_bars = int(lookback_raw) if lookback_raw is not None else 168

        category_raw = existing_info.get("category")
        if category_raw is None and overrides:
            category_raw = overrides.get("category")
        category_str = str(category_raw) if category_raw is not None else "Forex & Financial Time-Series"

        tf_names = {
            1: "M1 (1min)",
            5: "M5 (5min)",
            15: "M15 (15min)",
            60: "H1 (60min)",
            240: "H4 (240min)",
            1440: "D1 (1440min)",
        }
        tf_labels = [tf_names.get(tf, f"{tf}min") for tf in tfs_list]
        pairs_display = ", ".join(pairs_list) if pairs_list else "All Primary & Secondary FX Pairs"
        tfs_display = (
            ", ".join(tf_labels)
            if tf_labels
            else "M1 (1min), M5 (5min), M15 (15min), H1 (60min), H4 (240min), D1 (1440min)"
        )

        models_markdown = build_models_markdown(manifold_name, MODELS_META)

        details = forex_scan.get("details", []) if isinstance(forex_scan, dict) else []
        if details:
            year_rows: dict[tuple[int, str, int], int] = {}
            for d in details:
                key = (d["year"], d["pair"], d["timeframe"])
                year_rows[key] = year_rows.get(key, 0) + d["count"]
            year_table = "| Year | Pair | Timeframe | Samples |\n"
            year_table += "| :--- | :--- | :--- | :--- |\n"
            for (year, pair, tf), cnt in sorted(year_rows.items()):
                year_table += f"| {year} | {pair} | {tf}min | {cnt:,} |\n"
        else:
            year_table = "| Year | Pair | Timeframe | Samples |\n"
            year_table += "| :--- | :--- | :--- | :--- |\n"
            year_table += "| (dynamic) | (dynamic) | (dynamic) | (dynamic) |\n"

        structure_text = build_structure_text(output_root, is_forex=True, task_key="forex")

        readme = render_forex_readme(
            manifold_name=manifold_name,
            category_str=category_str,
            pairs_display=pairs_display,
            tfs_display=tfs_display,
            start_date_str=start_date_str,
            lookback_bars=lookback_bars,
            total_samples=total_samples,
            year_table=year_table.strip(),
            models_markdown=models_markdown.strip(),
            structure_text=structure_text.strip(),
        )
    else:
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
            table_text = "| **Standard Synthesis** | N/A | N/A | Full Contribution |"

        total_samples_display = f"{total_samples:,}" if isinstance(total_samples, int) else str(total_samples)
        arch_base = TASK_ARCH_BASE.get(task_key, "Deep Convolutional / Transformer Architecture")

        if total_train_all == 0 and isinstance(total_samples, int) and total_samples > 0:
            total_val_all = int(total_samples * 0.12)
            total_train_all = total_samples - total_val_all

        manifest_rows: list[str] = []
        manifest_rows.append(f"| **images** | {total_train_all:,} | {total_val_all:,} |")
        if (output_root / "targets").exists() or task_key in ["restoration", "super-resolution"]:
            manifest_rows.append(f"| **targets** | {total_train_all:,} | {total_val_all:,} |")
        if (output_root / "labels").exists() or task_key in ["detection", "pose", "classification"]:
            manifest_rows.append(f"| **labels** | {total_train_all:,} | {total_val_all:,} |")
        manifest_text = "\n".join(manifest_rows)

        targets_desc = (
            m.get("targets_desc", "Target matrices or masks for training.")
            if isinstance(m, dict)
            else "Target matrices or masks for training."
        )

        structure_text = build_structure_text(
            output_root,
            is_forex=False,
            task_key=task_key,
            targets_desc=targets_desc,
            img_desc=img_desc,
            tgt_desc=tgt_desc,
        )

        models_markdown = build_models_markdown(manifold_name, MODELS_META)
        category = m.get("category", "Dataset") if isinstance(m, dict) else "Dataset"

        readme = render_standard_readme(
            manifold_name=manifold_name,
            resolved_desc=resolved_desc,
            category=category,
            total_samples_display=total_samples_display,
            arch_base=arch_base,
            resolved_obj=resolved_obj,
            table_text=table_text.strip(),
            models_markdown=models_markdown.strip(),
            structure_text=structure_text.strip(),
            manifest_text=manifest_text.strip(),
        )

    readme = clean_readme(readme)
    with open(output_root / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    # 7. dataset-metadata.json
    write_kaggle_metadata(output_root, manifold_name, cat_str, readme, task_key)

    return total_samples


def regenerate_all_docs(datasets_dir: str | Path | None = None) -> None:
    """Regenerate documentation for all manifolds discovered in configuration and workspace."""
    if datasets_dir is None:
        datasets_dir = Path(__file__).resolve().parent.parent.parent / "LemGendaryDatasets"
    datasets_dir = Path(datasets_dir)

    print(f"Scanning manifolds in {datasets_dir}...")

    prefix = UNIFIED_DATA.get("_registry_metadata", {}).get("name_prefix", "LemGendized")
    suffix = UNIFIED_DATA.get("_registry_metadata", {}).get("name_suffix", "Large")
    target_names: set[str] = set()
    for d_key, d_info in UNIFIED_DATA.get("datasets", {}).items():
        t_name = d_info.get("name", d_key)
        target_names.add(f"{prefix}{t_name}{suffix}")

    if datasets_dir.exists():
        for p in datasets_dir.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                target_names.add(p.name)

    count = 0
    for name in sorted(target_names):
        p = datasets_dir / name
        p.mkdir(parents=True, exist_ok=True)
        print(f"Regenerating docs for {name}...")
        try:
            samples = generate_dataset_docs(p, None, name)
            print(f"  Success: {name} -> Total Samples: {samples}")
            count += 1
        except Exception as e:
            print(f"  Error on {name}: {e}")
    print(f"Regeneration complete for {count} manifolds.")


regenerate_all_non_forex = regenerate_all_docs


def main() -> None:
    """Command-line entrypoint for LemGendary Dataset Doc Generator."""
    parser = argparse.ArgumentParser(description="LemGendary Dataset Doc Generator")
    parser.add_argument("--all", action="store_true", help="Regenerate all manifold READMEs")
    parser.add_argument("--manifold", type=str, default=None, help="Specific manifold folder name")
    args = parser.parse_args()

    if args.manifold:
        m_path = Path(__file__).resolve().parent.parent.parent / "LemGendaryDatasets" / args.manifold
        generate_dataset_docs(m_path, None, args.manifold)
    else:
        regenerate_all_docs()
