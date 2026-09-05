import json
import os
from datetime import datetime
from pathlib import Path
import yaml

UNIFIED_DATA = yaml.safe_load(open(Path(__file__).parent / "unified_data.yaml", "r"))
TASK_META = UNIFIED_DATA.get("task_metadata", {})
MODELS_META = UNIFIED_DATA.get("models_metadata", {})
def format_source(name):
    lower_name = name.lower()
    # 2026 Resilience: Explicit SOTA Mappings
    if lower_name == 'celebamask': return 'CelebAMask'
    elif lower_name == 'affectnet': return 'AffectNet'
    elif lower_name == 'wflw': return 'WFLW'
    elif lower_name == 'ffhq': return 'FFHQ'
    elif lower_name == 'helen': return 'Helen'
    elif lower_name in ['ava', 'aadb', 'coco', 'csiq', 'spaq', 'live']: return lower_name.upper()
    elif lower_name == 'koniq10k': return 'KonIQ-10k'
    elif lower_name == 'tid2013': return 'TID2013'
    elif lower_name == 'laion': return 'LAION'
    elif lower_name == 'laion-5b': return 'LAION-5B'
    elif 'smartphone' in lower_name: return 'SIDD (Smartphone Image Denoising)'
    elif 'dnd' in lower_name or 'nam' in lower_name: return 'DND & NAM Noise Data'
    elif '9-classes' in lower_name: return '9-Classes Noisy Image Dataset'
    elif 'multi-noises' in lower_name: return 'Multi-Noise Synthetic Dataset'
    elif 'salt-and-pepper' in lower_name: return 'Salt-and-Pepper Noise'
    elif 'iso-levels' in lower_name: return 'Multiple ISO Denoising Dataset'
    elif 'gopro' in lower_name: return 'GoPro Deblurring Dataset'
    elif 'hideblur' in lower_name: return 'HiDeBlur Dataset'
    elif 'realblur' in lower_name: return 'RealBlur Dataset'
    else: return name.replace('-', ' ').title()

def generate_dataset_docs(output_root, final_index, pascal_name):
    """
    Generates index.json, dataset_info.yaml, category.txt, classes.txt, and README.md
    for the compiled dataset manifold.
    """
    output_root = Path(output_root)
    
    # Support for Forex datasets that don't use index.json
    task = "quality"
    if final_index:
        task = final_index[0]["task"]
    elif (output_root / "dataset_info.yaml").exists():
        with open(output_root / "dataset_info.yaml", "r") as f:
            d_info = yaml.safe_load(f)
            task = d_info.get("task", d_info.get("dataset_type", "quality"))
            
    if "authenticity" in str(output_root).lower():
        task = "authenticity"

    # 1. Generate index.json (Only if we have an index)
    if final_index:
        with open(output_root / "index.json", "w", encoding="utf-8") as f:
            json.dump(final_index, f, indent=2)

        # 2. Generate dataset_info.yaml
        sources_set = sorted(list(set(x.get("source", "unknown") for x in final_index)))
        yaml_content = f"""count: {len(final_index)}
task: {task}
original_sources:
{chr(10).join(f"- {s}" for s in sources_set)}
path: {str(output_root.resolve())}
source: {pascal_name}-manifold
last_processed: '{datetime.now().isoformat()}'
"""
        with open(output_root / "dataset_info.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)

    # 3. Generate category.txt
    cat_str = TASK_META.get(task, TASK_META["detection"])["category"]
    with open(output_root / "category.txt", "w", encoding="utf-8") as f:
        f.write(f"{cat_str}\n")

    # 4. Generate classes.txt
    with open(output_root / "classes.txt", "w", encoding="utf-8") as f:
        if task == "forex":
            f.write("Sell\nHold\nBuy\n")
        else:
            class_name = "face" if task == "pose" else task
            f.write(f"{class_name}\n")

    # 5. Generate README.md
    tasks = {}
    sources = {}
    total_samples = 0
    
    if final_index:
        total_samples = len(final_index)
        for item in final_index:
            tasks[item["task"]] = tasks.get(item["task"], 0) + 1

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
    elif task == "forex":
        total_samples = "Auto-Synced OHLCV Tensors"
        sources["MetaTrader 5 Native Cache"] = {"train": "N/A", "val": "N/A", "total": "Dynamic"}

    m = TASK_META.get(task, TASK_META.get("detection", {}))
    resolved_desc = m.get('desc', 'Dataset manifold.')
    resolved_obj = m.get('obj', 'Dataset objective.')
    img_desc = "RGB"
    tgt_desc = ""

    if task == "quality":
        pass
    elif task == "restoration":
        name_lower = output_root.name.lower()
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

    desc_map = {
        "images": f"Normalized input tensors ({img_desc}, standardized resolution).",
        "labels": "Strict numerical annotation vectors (JSON/TXT format).",
        "targets": f"Clean ground truth tensors ({tgt_desc})." if task == "restoration" else m.get('targets_desc', "Target matrices or masks for training."),
        "shards": "WebDataset `.tar` shards containing serialized manifold data.",
        "forex": "Shards containing serialized manifold data.",
        "dataset_info.yaml": "Manifest metadata for automated PyTorch loaders.",
        "category.txt": "Top-level categorization tag.",
        "classes.txt": "Class labels mapping.",
        "index.json": "Compiled metadata index mapping all dataset samples.",
        "README.md": "This documentation file."
    }


    structure_lines = []
    if output_root.exists():
        for item in sorted(output_root.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            name = item.name
            if name.endswith("_colab_training.ipynb"):
                desc = "Auto-generated Google Colab notebook for cloud training."
            elif "_training" in name and name.endswith(".ipynb"):
                desc = "Auto-generated Jupyter notebook for model training."
            else:
                desc = desc_map.get(name, "Dataset component.")
                
            if item.is_dir():
                structure_lines.append(f"- **`{name}/`**: {desc}")
            else:
                structure_lines.append(f"- **`{name}`**: {desc}")
                
    structure_text = "\n".join(structure_lines)

    # Find models that use this dataset
    dataset_name = output_root.name
    applicable_models = []
    for m_key, m_info in MODELS_META.items():
        if dataset_name in m_info.get("datasets", []):
            applicable_models.append(m_info)
    
    # Generate SOTA Tables
    models_markdown = ""
    for am in applicable_models:
        models_markdown += f"### Model: {am.get('name', 'Unknown Model')}\n\n"
        models_markdown += f"- **Architecture**: {am.get('arch', 'Unknown')}\n"
        models_markdown += f"- **Optimization**: {am.get('loss', 'Unknown')}\n\n"
        
        sota = am.get('sota_targets', {})
        if sota:
            models_markdown += "| Metric | Baseline | Advanced | SOTA |\n"
            models_markdown += "| :--- | :--- | :--- | :--- |\n"
            for met, val in sota.items():
                met_name = met.replace('_', ' ').title().replace('Psnr', 'PSNR').replace('Ssim', 'SSIM').replace('Lpips', 'LPIPS').replace('Fid', 'FID').replace('Map', 'mAP')
                if isinstance(val, (int, float)):
                    # Heuristic for extrapolation
                    lower_is_better = any(x in met.lower() for x in ['loss', 'lpips', 'fid', 'drawdown', 'mae', 'mse'])
                    if lower_is_better:
                        base = val * 2.0
                        adv = val * 1.5
                        models_markdown += f"| **{met_name}** | ~{base:.2f} | < {adv:.2f} | **< {val}** |\n"
                    else:
                        is_pct = any(x in met.lower() for x in ['acc', 'win', 'rate']) or (val > 20.0 and val <= 100.0)
                        if is_pct and val > 10.0:
                            base = val * 0.8
                            adv = val * 0.9
                            models_markdown += f"| **{met_name}** | ~{base:.1f}% | > {adv:.1f}% | **> {val}%** |\n"
                        else:
                            base = val * 0.8
                            adv = val * 0.9
                            models_markdown += f"| **{met_name}** | ~{base:.2f} | > {adv:.2f} | **> {val}** |\n"
                else:
                    models_markdown += f"| **{met_name}** | N/A | N/A | **{val}** |\n"
            models_markdown += "\n"

    if not models_markdown:
        models_markdown = "No models are explicitly bound to this dataset in unified_models_v2.yaml.\n"

    if task == "forex":
        yaml_path = output_root / "dataset_info.yaml"
        d_info = {}
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                d_info = yaml.safe_load(f)
        pairs_list = d_info.get("pairs", [])
        tfs_list = d_info.get("timeframe_rungs", [])
        start_date_str = d_info.get("start_date", "2019-01-01")
        lookback_bars = d_info.get("lookback_bars", 168)
        
        tf_names = {1: 'M1 (1min)', 5: 'M5 (5min)', 15: 'M15 (15min)', 60: 'H1 (60min)', 240: 'H4 (240min)', 1440: 'D1 (1440min)'}
        tf_labels = [tf_names.get(tf, f'{tf}min') for tf in tfs_list]
        
        models_block = models_markdown.strip()
        readme = f"""# {output_root.name}

> {m.get('desc', 'High-fidelity temporal manifold.')}

## Dataset Overview

- **Category:** {m.get('category', 'Financial Time-Series')}
- **Acquisition Mode:** MetaTrader 5 Terminal API / Synthetic Multi-Regime Generator
- **Pairs Included:** {', '.join(pairs_list)}
- **Timeframe Rungs:** {', '.join(tf_labels)}
- **Historical Horizon:** {start_date_str} to Present (6-Fold Walk-Forward Matrix with 14-day Embargo)
- **Lookback Window:** {lookback_bars} bars
- **Total Samples:** [Computed Dynamically During Training]
- **Output Classes:** `SELL` (0), `HOLD` (1), `BUY` (2) + Dual Pip Target Heads (TP/SL)
- **Primary Task:** {m.get('obj', 'Predict directional probability.')}

## Composition & Lineage

This manifold is dynamically assembled from the following temporal specifications:

- **Currency Pairs**: {', '.join(pairs_list)}
- **Timeframes (Minutes)**: {', '.join(tf_labels)}
- **Historical Horizon**: {start_date_str} to Present
- **Chronology Strategy**: 6-Fold Walk-Forward Matrix
- **Fold Embargo**: 14 Days

## Model Training Profiles

{models_block}

## Repository Structure

Standardized directory logic for seamless integration into the **LemGendary Training Suite**.

{structure_text}

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/{output_root.name.lower().replace('_', '-')})
"""
    else:
        models_block = models_markdown.strip()
        table_rows = []
        for src, counts in sorted(sources.items(), key=lambda x: str(x[1].get('total', 0)), reverse=True):
            table_rows.append(f"| **{src}** | {counts['train']} | {counts['val']} | {counts['total']} samples |")
        table_text = "\n".join(table_rows)

        readme = f"""# {output_root.name}

> {resolved_desc}

## Dataset Overview

- **Category:** {m.get('category', 'Dataset')}
- **Total Samples:** {total_samples}
- **Primary Task:** {resolved_obj}

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
{table_text}

## Model Training Profiles

{models_block}

## Repository Structure

Standardized directory logic for seamless integration into the **LemGendary Training Suite**.

{structure_text}

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/{output_root.name.lower().replace('_', '-')})
"""

    with open(output_root / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)
