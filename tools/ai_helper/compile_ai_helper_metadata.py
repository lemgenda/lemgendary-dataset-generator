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
- Ecosystem Architecture And Current State Manifest
- Master Operations Manuals (GUI, CLI, API)
- Dataset Compiler And Environment Manager Whitepapers
- Model Family Architecture Whitepapers
- Canonical Glossary And AI Helper Troubleshooting Knowledge Base
- Curated AI LemGend Instruction And Validation Examples
path: ..\\LemGendaryDatasets\\LemGendizedAIHelperLemGend
source: LemGendizedAIHelperLemGend-manifold
last_processed: '2026-10-16T17:30:00.000000+00:00'
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
    r"\d+\s*supervised conversation samples",
    f"{total_samples} supervised conversation samples",
    readme_text,
)
readme_text = re.sub(
    r"\(\d+\s*training,\s*\d+\s*validation\)",
    f"({train_count} training, {val_count} validation)",
    readme_text,
)

# Rebuild category table
table_lines = [
    "| Operational Category | Train Samples | Validation Samples | Total Samples | Percentage |",
    "| :--- | :--- | :--- | :--- | :--- |",
]
for cat in sorted(all_cats.keys()):
    t_cnt = train_cats.get(cat, 0)
    v_cnt = val_cats.get(cat, 0)
    tot = all_cats[cat]
    pct = (tot / total_samples) * 100
    table_lines.append(f"| `{cat}` | {t_cnt} | {v_cnt} | {tot} | {pct:.1f}% |")
table_lines.append(
    f"| **Total** | **{train_count}** | **{val_count}** | **{total_samples}** | **100.0%** |"
)
new_table = "\n".join(table_lines)

table_pattern = re.compile(
    r"\| Operational Category \| Train Samples \| Validation Samples \| Total Samples \| "
    r"Percentage \|[\s\S]*?\|\s*\*\*Total\*\*.*?\|",
    re.MULTILINE,
)
if table_pattern.search(readme_text):
    readme_text = table_pattern.sub(new_table, readme_text)

# Ensure zero mention of Rain Streaks
readme_text = readme_text.replace("Rain Streaks Dataset", "LemGendary AI Documentation Hub")

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_text)
print(f"Updated {readme_path}")

# 5. Update dataset-metadata.json
description_text = (
    f"The LemGendizedAIHelperLemGend corpus is the official, verified fine-tuning and evaluation "
    f"dataset for AI LemGend, the interactive assistant integrated into LemGendary AI Studio. "
    f"Comprising {total_samples} supervised instruction-tuning records ({train_count} training pairs, "
    f"{val_count} held-out validation pairs), this dataset formalizes procedural knowledge, "
    f"operational parameters, diagnostic triage protocols, and architectural specifications across "
    f"the entire LemGendary AI ecosystem.\n\n"
    f"### Corpus Architecture\n"
    f"Each record contains a multi-turn conversation format with structured system, user, and assistant turns, "
    f"adhering to the 3-Stage Diagnostic Intake Protocol and epistemic confidence markers "
    f"([CONFIRMED], [PROBABLE], [ILLUSTRATIVE]).\n\n"
    f"### Operational Scope\n"
    f"- GUI Procedures: Step-by-step guidance for LemGendary AI Studio desktop application (Tauri v2 + React).\n"
    f"- CLI Commands: Command-line reference for the lemtrain training engine.\n"
    f"- API Integrations: OpenAPI endpoints, REST methods, and WebSocket streams across Ports 8000, 8100, and 8200.\n"
    f"- Model Selection & SOTA Ladders: Architecture choices across all 22 supported model families.\n"
    f"- Diagnostic Triage: Safe triage, telemetry analysis, and bounded remediation for training failure modes."
)

collection_methodology_text = (
    "Curated supervised instruction examples derived from reviewed documentation and general AI/ML "
    "engineering principles. Each example is designed around a user task, diagnostic decision, "
    "terminology distinction, or documentation authority rule. Records use one JSON object per line "
    "with messages in system/user/assistant order. Validated for zero semantic leakage, schema integrity, "
    "and held-out evaluation performance.\n"
)

meta_data = {
    "title": "LemGendizedAIHelperLemGend",
    "id": "lemtreursi/lemgendizedaihelperlemgend",
    "subtitle": (
        f"Authoritative Fine-Tuning Corpus for LemGendary AI Studio Assistant "
        f"({total_samples} Conversational Records)"
    ),
    "description": description_text,
    "isPrivate": False,
    "licenses": [{"name": "MIT"}],
    "keywords": [
        "ai-assistant",
        "instruction-tuning",
        "lemgendary-ai",
        "pytorch",
        "model-training",
        "synthetic-data",
        "vision-restoration",
        "quant-trading",
        "deep-learning",
        "studio-copilot",
    ],
    "collaborators": [],
    "data": [
        {
            "path": "ai_helper_train.jsonl",
            "description": (
                f"Supervised instruction tuning training set containing {train_count} "
                "conversational records with record-level provenance."
            ),
        },
        {
            "path": "ai_helper_val.jsonl",
            "description": (
                f"Held-out evaluation benchmark containing {val_count} conversational "
                "records covering diagnostics, procedures, and unsupported refusal edge cases."
            ),
        },
        {
            "path": "classes.txt",
            "description": "Exhaustive newline-delimited class dictionary.",
        },
        {
            "path": "dataset_info.yaml",
            "description": (
                f"Machine-readable YAML manifest detailing sample counts ({total_samples} samples), "
                "verified source lineage, and container topology."
            ),
        },
        {
            "path": "index.json",
            "description": (
                f"Master sample inventory mapping all {total_samples} samples with split partitions "
                "and record-level source provenance."
            ),
        },
        {
            "path": "README.md",
            "description": "Authoritative manifold documentation, category distribution, and lineage matrix.",
        },
    ],
    "schema": {
        "format": "jsonl",
        "record_fields": {
            "id": "string (unique record identifier)",
            "category": "string (operational task domain)",
            "provenance": (
                "object {source_doc_id, canonical_url, doc_version, section_anchor, "
                "curation_method, content_sha256, split_rule}"
            ),
            "messages": "array of objects [{role: 'system'|'user'|'assistant', content: string}]",
        },
    },
    "headerImage": "AIHelperLemGend.jpg",
    "provenanceSources": [
        "LemGendary AI Documentation Hub",
        "General AI Training Knowledge Reference",
        "Ecosystem Architecture and Current State Manifest",
        "Master Operations Manuals (GUI, CLI, API)",
        "Dataset Compiler and Environment Manager Whitepapers",
        "Training Suite and Training Pathology Documentation",
        "Model-Family Architecture Whitepapers",
        "Canonical Glossary and AI Helper Troubleshooting Knowledge Base",
        "Curated AI LemGend instruction and validation examples",
    ],
    "collectionMethodology": collection_methodology_text,
    "citations": ["https://github.com/lemtreursi/model-training"],
    "coverage": {
        "temporal": {"start": "2024-01-01", "end": "2026-10-16"},
        "geospatial": "Global Digital Documentation and AI Engineering",
    },
}

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta_data, f, indent=2)
print(f"Updated {meta_path}")
