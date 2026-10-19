#!/usr/bin/env python3
"""
Authoritative Metadata Compiler for LemGendizedAIHelperLemGend.
Reads ai_helper_train.jsonl and ai_helper_val.jsonl directly from disk,
computes verified sample counts and category distributions, and regenerates:
  1. dataset_info.yaml
  2. index.json
  3. dataset-metadata.json
  4. README.md
"""

import json
import os
from pathlib import Path
import re
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[3]
dataset_dir = str(BASE_DIR / "LemGendaryDatasets" / "LemGendizedAIHelperLemGend")
train_path = os.path.join(dataset_dir, "ai_helper_train.jsonl")
val_path = os.path.join(dataset_dir, "ai_helper_val.jsonl")
info_path = os.path.join(dataset_dir, "dataset_info.yaml")
meta_path = os.path.join(dataset_dir, "dataset-metadata.json")
index_path = os.path.join(dataset_dir, "index.json")
readme_path = os.path.join(dataset_dir, "README.md")

# 1. Read records directly from disk
train_records = []
with open(train_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            train_records.append(json.loads(line))

val_records = []
with open(val_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            val_records.append(json.loads(line))

total_samples = len(train_records) + len(val_records)
train_count = len(train_records)
val_count = len(val_records)

print(
    f"Loaded {train_count} train records and {val_count} val records (total: {total_samples})."
)

train_cats = Counter(r["category"] for r in train_records)
val_cats = Counter(r["category"] for r in val_records)
all_cats = Counter(r["category"] for r in train_records + val_records)

# 2. Update dataset_info.yaml
info_content = f"""---
name: LemGendizedAIHelperLemGend
count: {total_samples}
task: ai_helper
category: Technical Documentation & AI Assistant Training
canonical_format: jsonl
original_sources:
- LemGendary AI Documentation Hub
- General AI Training Knowledge Reference
- Ecosystem Architecture and Current State Manifest
- Master Operations Manuals (GUI, CLI, API)
- Dataset Compiler and Environment Manager Whitepapers
- Training Suite and Training Pathology Documentation
- Model-Family Architecture Whitepapers
- Canonical Glossary and AI Helper Troubleshooting Knowledge Base
- Curated AI LemGend Instruction and Validation Examples
path: ..\\LemGendaryDatasets\\LemGendizedAIHelperLemGend
source: LemGendizedAIHelperLemGend-manifold
last_processed: '2026-10-17T12:00:00.000000+00:00'
train_samples: {train_count}
val_samples: {val_count}
"""

with open(info_path, "w", encoding="utf-8") as f:
    f.write(info_content)
print(f"Updated {info_path}")

# 3. Generate index.json
index_data = []
current_id = 0
for r in train_records:
    prov = r.get("provenance", {})
    index_data.append(
        {
            "id": current_id,
            "name": f"AIHelper_train_{current_id:06d}",
            "record_id": r["id"],
            "category": r["category"],
            "source_doc_id": prov.get(
                "source_doc_id", "LemGendary-AI-Documentation"
            ),
            "canonical_url": prov.get("canonical_url", ""),
            "task": "ai_helper",
            "split": "train",
            "hash": prov.get("content_sha256", "e3b0c44298fc1c14")[:16],
            "nima_score": 1.0,
            "caption": None,
            "style_tag": None,
            "cluster_id": -1,
        }
    )
    current_id += 1

val_id = 0
for r in val_records:
    prov = r.get("provenance", {})
    index_data.append(
        {
            "id": current_id,
            "name": f"AIHelper_val_{val_id:06d}",
            "record_id": r["id"],
            "category": r["category"],
            "source_doc_id": prov.get(
                "source_doc_id", "LemGendary-AI-Documentation"
            ),
            "canonical_url": prov.get("canonical_url", ""),
            "task": "ai_helper",
            "split": "val",
            "hash": prov.get("content_sha256", "e3b0c44298fc1c14")[:16],
            "nima_score": 1.0,
            "caption": None,
            "style_tag": None,
            "cluster_id": -1,
        }
    )
    current_id += 1
    val_id += 1

with open(index_path, "w", encoding="utf-8") as f:
    json.dump(index_data, f, indent=2)
print(f"Updated {index_path} with {len(index_data)} entries")

# 4. Update README.md
with open(readme_path, "r", encoding="utf-8") as f:
    readme_text = f.read()

# Replace sample count badges and text
readme_text = re.sub(
    r"\*\*Total Samples:\*\*\s*\d+\s*\(\d+\s*train\s*/\s*\d+\s*val\)",
    f"**Total Samples:** {total_samples} ({train_count} train / {val_count} val)",
    readme_text,
)
readme_text = re.sub(
    r"\*\*Total Samples\*\*:\s*\d+", f"**Total Samples**: {total_samples}", readme_text
)
readme_text = re.sub(
    r"\*\*Training Samples\*\*:\s*\d+",
    f"**Training Samples**: {train_count}",
    readme_text,
)
readme_text = re.sub(
    r"\*\*Validation Samples\*\*:\s*\d+",
    f"**Validation Samples**: {val_count}",
    readme_text,
)
readme_text = re.sub(
    r"`ai_helper_train\.jsonl`\*\*:\s*Supervised instruction training split \(\d+ dialogue records\)",
    f"`ai_helper_train.jsonl`**: Supervised instruction training split ({train_count} dialogue records)",
    readme_text,
)
readme_text = re.sub(
    r"`ai_helper_val\.jsonl`\*\*:\s*Held-out evaluation benchmark split \(\d+ dialogue records\)",
    f"`ai_helper_val.jsonl`**: Held-out evaluation benchmark split ({val_count} dialogue records)",
    readme_text,
)
readme_text = re.sub(
    r"`index\.json`\*\*:\s*Master sample inventory mapping all \d+ records",
    f"`index.json`**: Master sample inventory mapping all {total_samples} records",
    readme_text,
)

# Rebuild Task Category Distribution table
table_lines = [
    "| Task Category | Train | Val | Total Contribution |",
    "| :--- | :--- | :--- | :--- |",
]
for cat in sorted(all_cats.keys()):
    t_cnt = train_cats.get(cat, 0)
    v_cnt = val_cats.get(cat, 0)
    tot = all_cats[cat]
    table_lines.append(f"| **`{cat}`** | {t_cnt} | {v_cnt} | {tot} samples |")
table_lines.append(
    f"| **Total Corpus** | **{train_count}** | **{val_count}** | **{total_samples} samples** |"
)
new_table = "\n".join(table_lines)

table_pattern = re.compile(
    r"\|\s*Task Category\s*\|\s*Train\s*\|\s*Val\s*\|\s*Total Contribution\s*\|[\s\S]*?\|\s*\*\*Total Corpus\*\*[^\n]*\|",
    re.MULTILINE,
)
if table_pattern.search(readme_text):
    readme_text = table_pattern.sub(new_table, readme_text)

# Replace Physical Data Manifest table
manifest_table_lines = [
    "| File | Partition | Samples | Format |",
    "| :--- | :--- | :--- | :--- |",
    f"| **`ai_helper_train.jsonl`** | Train | {train_count} | JSONL (OpenAI / ChatML message schema) |",
    f"| **`ai_helper_val.jsonl`** | Val | {val_count} | JSONL (OpenAI / ChatML message schema) |",
]
manifest_table = "\n".join(manifest_table_lines)
manifest_pattern = re.compile(
    r"\|\s*File\s*\|\s*Partition\s*\|\s*Samples\s*\|\s*Format\s*\|[\s\S]*?ai_helper_val\.jsonl[^\n]*\|",
    re.MULTILINE,
)
if manifest_pattern.search(readme_text):
    readme_text = manifest_pattern.sub(manifest_table, readme_text)

# Ensure zero mention of Rain Streaks
readme_text = readme_text.replace("Rain Streaks Dataset", "LemGendary AI Documentation Hub")
readme_text = re.sub(r"release `v16\.9\.\d+-STABLE`", "release `v16.9.27-STABLE`", readme_text)

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_text)
print(f"Updated {readme_path}")

# 5. Update dataset-metadata.json using canonical Kaggle metadata generator
import sys
module_root = BASE_DIR / "lemgendary-datasets"
if str(module_root) not in sys.path:
    sys.path.insert(0, str(module_root))

from core.docs.generators.kaggle_metadata import generate_kaggle_metadata

meta_data = generate_kaggle_metadata(Path(dataset_dir))
print(
    f"Updated {meta_path} (License: {meta_data['licenses']}, Resources: {len(meta_data.get('resources', []))})"
)
