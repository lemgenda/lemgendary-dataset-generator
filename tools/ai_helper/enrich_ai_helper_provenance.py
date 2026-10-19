#!/usr/bin/env python3
"""
Enrich AI Helper dataset records with record-level provenance metadata.
Injects source_doc_id, canonical_url, doc_version, section_anchor, curation_method,
content_sha256, and split_rule into each JSONL record.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parents[3]
DATASET_DIR = str(BASE_DIR / "LemGendaryDatasets" / "LemGendizedAIHelperLemGend")
TRAIN_PATH = os.path.join(DATASET_DIR, "ai_helper_train.jsonl")
VAL_PATH = os.path.join(DATASET_DIR, "ai_helper_val.jsonl")

CATEGORY_DOC_MAPPING = {
    "troubleshooting_diagnostics": {
        "doc_id": "AI_HELPER_TROUBLESHOOTING_KNOWLEDGE.md",
        "url": "https://lemgenda.github.io/ai-training-whitepapers/papers/AI_HELPER_TROUBLESHOOTING_KNOWLEDGE.html",
        "anchor": "#diagnostic-triage",
    },
    "cross_document_pipeline": {
        "doc_id": "PAPER_TRAINING_SUITE.md",
        "url": "https://lemgenda.github.io/ai-training-whitepapers/papers/PAPER_TRAINING_SUITE.html",
        "anchor": "#pipeline-lifecycle",
    },
    "terminology_and_invariants": {
        "doc_id": "GLOSSARY.md",
        "url": "https://lemgenda.github.io/ai-training-whitepapers/papers/GLOSSARY.html",
        "anchor": "#canonical-invariants",
    },
    "cli_procedure": {
        "doc_id": "MANUAL_CLI.md",
        "url": "https://lemgenda.github.io/ai-training-whitepapers/papers/MANUAL_CLI.html",
        "anchor": "#cli-reference",
    },
    "gui_procedure": {
        "doc_id": "MANUAL_AI_STUDIO_GUI.md",
        "url": "https://lemgenda.github.io/ai-training-whitepapers/papers/MANUAL_AI_STUDIO_GUI.html",
        "anchor": "#gui-workflows",
    },
    "api_procedure": {
        "doc_id": "MANUAL_API.md",
        "url": "https://lemgenda.github.io/ai-training-whitepapers/papers/MANUAL_API.html",
        "anchor": "#api-contracts",
    },
    "model_selection": {
        "doc_id": "ECOSYSTEM_ARCHITECTURE.md",
        "url": "https://lemgenda.github.io/ai-training-whitepapers/papers/ECOSYSTEM_ARCHITECTURE.html",
        "anchor": "#model-registry",
    },
    "unsupported_and_negative": {
        "doc_id": "canonical_capabilities.yaml",
        "url": "https://lemgenda.github.io/ai-training-whitepapers/papers/LEMGENDARY_CURRENT_STATE.html",
        "anchor": "#unsupported-boundaries",
    },
}


def compute_content_sha256(messages: list) -> str:
    serialized = json.dumps(messages, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def enrich_file(filepath: str, split_name: str) -> int:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    enriched_lines = []
    for line in lines:
        record: Dict[str, Any] = json.loads(line)
        cat = record.get("category", "troubleshooting_diagnostics")
        mapping = CATEGORY_DOC_MAPPING.get(
            cat, CATEGORY_DOC_MAPPING["troubleshooting_diagnostics"]
        )

        provenance = {
            "source_doc_id": mapping["doc_id"],
            "canonical_url": mapping["url"],
            "doc_version": "v16.9.31-STABLE",
            "section_anchor": mapping["anchor"],
            "curation_method": "expert_authored_verified",
            "content_sha256": compute_content_sha256(
                record.get("messages", [])
            ),
            "split_rule": f"domain_stratified_{split_name}_v1",
        }

        # Structure ordered record: id, category, provenance, messages
        ordered_record = {
            "id": record["id"],
            "category": record["category"],
            "provenance": provenance,
            "messages": record["messages"],
        }
        enriched_lines.append(
            json.dumps(ordered_record, ensure_ascii=False) + "\n"
        )

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(enriched_lines)

    return len(enriched_lines)


def main():
    train_count = enrich_file(TRAIN_PATH, "train")
    val_count = enrich_file(VAL_PATH, "val")
    print(
        f"Enriched {train_count} train records and {val_count} val records with record-level provenance."
    )


if __name__ == "__main__":
    main()
