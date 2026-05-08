import os
import json
import sqlite3
import yaml
from pathlib import Path
from datetime import datetime

# Path to the manifold
MANIFOLD_PATH = Path("c:/Development/python/model-training/LemGendaryDatasets/LemGendizedFilmRestorerLarge")
DB_PATH = MANIFOLD_PATH / "manifold_registry.db"
INDEX_PATH = MANIFOLD_PATH / "index.json"
README_PATH = MANIFOLD_PATH / "README.md"
YAML_PATH = Path("c:/Development/python/model-training/lemgendary-datasets/unified_data.yaml")

def format_source(name):
    lower_name = name.lower()
    if lower_name == 'celebamask': return 'CelebAMask'
    elif lower_name == 'affectnet': return 'AffectNet'
    elif lower_name == 'ffhq': return 'FFHQ'
    elif lower_name in ['ava', 'aadb', 'coco', 'csiq', 'spaq', 'live']: return lower_name.upper()
    elif lower_name == 'koniq10k': return 'KonIQ-10k'
    elif lower_name == 'tid2013': return 'TID2013'
    elif lower_name == 'laion': return 'LAION'
    elif 'smartphone' in lower_name: return 'SIDD (Smartphone Image Denoising)'
    elif 'dnd' in lower_name or 'nam' in lower_name: return 'DND & NAM Noise Data'
    elif 'gopro' in lower_name: return 'GoPro Deblurring Dataset'
    elif 'old-film' in lower_name: return 'Old Film Restoration'
    elif 'old-photos' in lower_name: return 'Old Photos'
    elif 'vintage-photo' in lower_name: return 'Vintage Photo Restoration'
    elif 'vintage-degraded' in lower_name: return 'Vintage Degraded Synthetic'
    elif 'photo-restoration' in lower_name: return 'Photo Restoration (Sureshmud)'
    else: return name.replace('-', ' ').title()

def repair():
    print(f"[REPAIR] Commencing surgical repair of {MANIFOLD_PATH.name}...")
    
    # 1. Scan disk
    samples = []
    for split in ["train", "val"]:
        img_dir = MANIFOLD_PATH / "images" / split
        if not img_dir.exists(): continue
        
        for entry in os.scandir(img_dir):
            if entry.is_file():
                name = entry.name.split(".")[0]
                # name format: FilmRestorer_source_idx
                parts = name.split("_")
                if len(parts) >= 3:
                    source = "_".join(parts[1:-1])
                    samples.append({
                        "name": name,
                        "source": source,
                        "split": split,
                        "task": "restoration"
                    })

    print(f"[OK] Scanned {len(samples)} samples from disk.")
    if not samples:
        print("[ERROR] No samples found on disk!")
        return

    # 2. Update Registry
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM registry")
    entries = [
        (s["name"], s["source"], s["task"], s["split"], None, 1.0, None, None, None, None)
        for s in samples
    ]
    conn.executemany("""
        INSERT OR IGNORE INTO registry (name, source, task, split, hash, nima_score, caption, style_tag, clip_latent, img_bytes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, entries)
    conn.commit()
    print(f"[OK] Registry updated with {len(entries)} entries.")

    # 3. Generate index.json
    final_index = []
    cursor = conn.execute("SELECT * FROM registry")
    for row in cursor:
        final_index.append({
            "id": row[0], "name": row[1], "source": row[2], "task": row[3], "split": row[4],
            "hash": row[5], "nima_score": row[6], "caption": row[7], "style_tag": row[8], "cluster_id": row[11]
        })
    
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(final_index, f, indent=2)
    print(f"[OK] index.json updated.")

    # 4. Generate README
    sources = {}
    for item in final_index:
        src_name = format_source(item["source"])
        if src_name not in sources:
            sources[src_name] = {"train": 0, "val": 0, "total": 0}
        sources[src_name]["total"] += 1
        if item["split"] in ["train", "val"]:
            sources[src_name][item["split"]] += 1

    readme = f"""# {MANIFOLD_PATH.name}

> Standardized dataset for image restoration, denoising, and enhancement models.

## 📊 Dataset Overview
- **Category:** Image Restoration
- **Total Samples:** {len(final_index):,}
- **Architecture Base:** UNet-based restoration architectures with residual learning
- **Primary Task:** Restore degraded images and enhance visual quality.

## 🧬 Composition & Lineage
This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
"""
    for src, counts in sorted(sources.items(), key=lambda x: x[1]["total"], reverse=True):
        readme += f"| **{src}** | {counts['train']:,} | {counts['val']:,} | {counts['total']:,} samples |\n"

    readme += f"""
## 🎯 Model Training Profile
- **Target Architectures**: NafNet, MirNet, MprNet, FfaNet, MultiTaskRestorer
- **Optimization Strategy**: L1 Loss, SSIM Loss, Charbonnier Loss

### Benchmark Metrics [SOTA]
| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **PSNR** | ~28.0 dB | > 31.0 dB | **> 33.0 dB** |
| **SSIM** | ~0.8000  | > 0.8800  | **> 0.9200**  |
| **LPIPS**| ~0.1500  | < 0.1200  | **< 0.0800**  |
| **FID**  | ~15.00   | < 12.00   | **< 8.00**    |

## 📂 Repository Structure
Standardized directory logic for seamless integration into the **LemGendary Training Suite**.

- **`images/`**: Normalized input tensors (RGB, standardized resolution).
- **`labels/`**: Strict numerical annotation vectors (JSON/TXT format).
- **`targets/`**: ACTIVELY DEPLOYED (High-Resolution ground truth).
- **`dataset_info.yaml`**: Manifest metadata for automated PyTorch loaders.

---
**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/{MANIFOLD_PATH.name.lower().replace('_', '-')})
"""
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"[OK] README.md updated with proper source counts.")

if __name__ == "__main__":
    repair()
