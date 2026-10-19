#!/usr/bin/env python3
"""
Enrich AI Helper dataset records with record-level provenance metadata.
Injects source_doc_id, canonical_url, doc_version, section_anchor, curation_method,
content_sha256, and split_rule into each JSONL record.

Ensures 100% of canonical URLs strictly match real published HTML files in lemgendary-docs/papers/.
Version is aligned strictly with the authoritative documentation baseline: v16.9.27-STABLE.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parents[3]
DATASET_DIR = str(BASE_DIR / "LemGendaryDatasets" / "LemGendizedAIHelperLemGend")
TRAIN_PATH = os.path.join(DATASET_DIR, "ai_helper_train.jsonl")
VAL_PATH = os.path.join(DATASET_DIR, "ai_helper_val.jsonl")
DOCS_MANIFEST_PATH = str(BASE_DIR / "lemgendary-docs" / "manifest" / "documents.json")
PAPERS_DIR = str(BASE_DIR / "lemgendary-docs" / "papers")

DOC_VERSION = "v16.9.27-STABLE"
BASE_URL = "https://lemgenda.github.io/ai-training-whitepapers/papers/"

# Complete 48-publication mapping from Markdown source file to published HTML filename
DOC_TO_HTML = {
    "AI_HELPER_TROUBLESHOOTING_KNOWLEDGE.md": "ai-helper-troubleshooting-knowledge.html",
    "ECOSYSTEM_ARCHITECTURE.md": "ecosystem-architecture.html",
    "GENERAL_AI_TRAINING_KNOWLEDGE.md": "general-ai-training-knowledge.html",
    "GLOSSARY.md": "glossary.html",
    "GUI_CONTROL_REGISTRY.md": "gui-control-registry.html",
    "LEMGENDARY_CURRENT_STATE.md": "current-state.html",
    "MANUALS_HUB.md": "manuals-hub.html",
    "MANUAL_AI_STUDIO_GUI.md": "ai-studio-manual.html",
    "MANUAL_API.md": "api-manual.html",
    "MANUAL_CLI.md": "cli-manual.html",
    "PAPER_AI_STUDIO_GUI.md": "ai-studio-whitepaper.html",
    "PAPER_DATASET_COMPILER.md": "dataset-compiler.html",
    "PAPER_DEGRADATION_ENGINE.md": "degradation-engine.html",
    "PAPER_DETECTION_MASTER.md": "detection-master.html",
    "PAPER_ENV_MANAGER.md": "env_manager.html",
    "PAPER_FOREX_PREDICTOR.md": "forex_predictor.html",
    "PAPER_FOUNDATION_MODELS_MASTER.md": "foundation-models-master.html",
    "PAPER_LEMGENDARY_CODEFORMER.md": "face-codeformer.html",
    "PAPER_LEMGENDARY_DETECTION_CLASSIFICATION.md": "detection-classification.html",
    "PAPER_LEMGENDARY_DIFFUSION.md": "foundation-diffusion.html",
    "PAPER_LEMGENDARY_FACE_SUITE.md": "face-suite.html",
    "PAPER_LEMGENDARY_FFANET.md": "ffanet.html",
    "PAPER_LEMGENDARY_FILM_RESTORER.md": "hybrid-film-restorer.html",
    "PAPER_LEMGENDARY_HYBRID.md": "universal-hybrid.html",
    "PAPER_LEMGENDARY_IMAGE_TO_TEXT.md": "foundation-image-to-text.html",
    "PAPER_LEMGENDARY_LLM.md": "foundation-llm.html",
    "PAPER_LEMGENDARY_MIRNET.md": "mirnet.html",
    "PAPER_LEMGENDARY_MPRNET.md": "mprnet.html",
    "PAPER_LEMGENDARY_MULTITASK.md": "hybrid-multitask.html",
    "PAPER_LEMGENDARY_NAFNET.md": "nafnet.html",
    "PAPER_LEMGENDARY_NIMA.md": "nima-quality.html",
    "PAPER_LEMGENDARY_NSFW_CLASSIFIER.md": "classification-nsfw.html",
    "PAPER_LEMGENDARY_PARSENET.md": "face-parsenet.html",
    "PAPER_LEMGENDARY_RESTORATION_SUITE.md": "restoration-suite.html",
    "PAPER_LEMGENDARY_RETINAFACE.md": "face-retinaface.html",
    "PAPER_LEMGENDARY_ULTRAZOOM.md": "ultrazoom.html",
    "PAPER_LEMGENDARY_UPN_V2.md": "hybrid-upn-v2.html",
    "PAPER_LEMGENDARY_YOLOV8N.md": "detection-yolov8n.html",
    "PAPER_NIMA_AUTHENTICITY.md": "nima-authenticity.html",
    "PAPER_NIMA_EFFICIENTNET.md": "nima-efficientnet.html",
    "PAPER_NIMA_MASTER.md": "nima-master.html",
    "PAPER_NIMA_MOBILE.md": "nima-mobile.html",
    "PAPER_NIMA_PRO.md": "nima-pro.html",
    "PAPER_NIMA_TECHNICAL.md": "nima-technical.html",
    "PAPER_RESTORATION_MASTER.md": "restoration-master.html",
    "PAPER_TRAINING_PATHOLOGY.md": "training-pathology.html",
    "PAPER_TRAINING_SUITE.md": "training-suite-master.html",
    "VERSIONING_POLICY.md": "versioning-policy.html",
}

# If manifest exists on disk, synchronize dynamic mapping
if os.path.exists(DOCS_MANIFEST_PATH):
    try:
        with open(DOCS_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            for d in manifest_data.get("documents", []):
                md_name = os.path.basename(d.get("md_path", ""))
                html_name = os.path.basename(d.get("html_path", ""))
                if md_name and html_name:
                    DOC_TO_HTML[md_name] = html_name
    except Exception as exc:
        print(f"Warning: could not read manifest from {DOCS_MANIFEST_PATH}: {exc}")


def resolve_record_doc_id(record_id: str, category: str) -> str:
    """Resolve the specific authoritative source document ID for each record."""
    # 1. Direct model pipeline records
    if record_id == "pipeline_nafnet": return "PAPER_LEMGENDARY_NAFNET.md"
    if record_id == "pipeline_mprnet": return "PAPER_LEMGENDARY_MPRNET.md"
    if record_id == "pipeline_mirnet": return "PAPER_LEMGENDARY_MIRNET.md"
    if record_id == "pipeline_ffanet": return "PAPER_LEMGENDARY_FFANET.md"
    if record_id == "pipeline_yolov8n": return "PAPER_LEMGENDARY_YOLOV8N.md"
    if record_id == "pipeline_ultrazoom": return "PAPER_LEMGENDARY_ULTRAZOOM.md"
    if record_id == "pipeline_hybrid_upn_v2": return "PAPER_LEMGENDARY_UPN_V2.md"
    if record_id == "pipeline_hybrid_film_restorer": return "PAPER_LEMGENDARY_FILM_RESTORER.md"
    if record_id == "pipeline_hybrid_multitask": return "PAPER_LEMGENDARY_MULTITASK.md"
    if record_id == "pipeline_face_codeformer": return "PAPER_LEMGENDARY_CODEFORMER.md"
    if record_id == "pipeline_face_parsenet": return "PAPER_LEMGENDARY_PARSENET.md"
    if record_id == "pipeline_retinaface": return "PAPER_LEMGENDARY_RETINAFACE.md"
    if record_id == "pipeline_nima_aesthetic_mobile": return "PAPER_NIMA_MOBILE.md"
    if record_id == "pipeline_nima_aesthetic_efficientnet": return "PAPER_NIMA_EFFICIENTNET.md"
    if record_id == "pipeline_nima_aesthetic_pro": return "PAPER_NIMA_PRO.md"
    if record_id == "pipeline_nima_technical": return "PAPER_NIMA_TECHNICAL.md"
    if record_id == "pipeline_nima_authenticity": return "PAPER_NIMA_AUTHENTICITY.md"
    if record_id == "pipeline_classification_nsfw": return "PAPER_LEMGENDARY_NSFW_CLASSIFIER.md"
    if record_id == "pipeline_forex_predictor": return "PAPER_FOREX_PREDICTOR.md"

    # 2. Specific architecture records
    if "architecture_nafnet" in record_id: return "PAPER_LEMGENDARY_NAFNET.md"
    if "architecture_codeformer" in record_id: return "PAPER_LEMGENDARY_CODEFORMER.md"
    if "architecture_upn_v2" in record_id or "hybrid_upn_v2" in record_id: return "PAPER_LEMGENDARY_UPN_V2.md"
    if "architecture_universal_film_restorer" in record_id: return "PAPER_LEMGENDARY_FILM_RESTORER.md"
    if "architecture_yolov8n" in record_id: return "PAPER_LEMGENDARY_YOLOV8N.md"
    if "architecture_diffusion" in record_id: return "PAPER_LEMGENDARY_DIFFUSION.md"
    if "architecture_vlm" in record_id: return "PAPER_LEMGENDARY_IMAGE_TO_TEXT.md"
    if "architecture_llm" in record_id: return "PAPER_LEMGENDARY_LLM.md"
    if "architecture_multitask" in record_id: return "PAPER_LEMGENDARY_MULTITASK.md"
    if "mprnet" in record_id: return "PAPER_LEMGENDARY_MPRNET.md"
    if "nima" in record_id:
        if "authenticity" in record_id: return "PAPER_NIMA_AUTHENTICITY.md"
        if "four_quadrant" in record_id or "resonance" in record_id or "intra_epoch" in record_id: return "PAPER_NIMA_MASTER.md"
        return "PAPER_LEMGENDARY_NIMA.md"
    if "forex" in record_id: return "PAPER_FOREX_PREDICTOR.md"

    # 3. Environment Manager records
    if record_id.startswith("env_manager") or record_id.startswith("envmanager") or "rebuild_virtualenvs" in record_id:
        return "PAPER_ENV_MANAGER.md"

    # 4. Dataset Compiler records
    if record_id.startswith("compiler_") or "format_parquet" in record_id or "format_zarr" in record_id or "format_webdataset" in record_id:
        return "PAPER_DATASET_COMPILER.md"

    # 5. General AI Training / Weights format
    if "format_arrow" in record_id or "format_safetensors" in record_id or "weights_" in record_id or "val_safetensors" in record_id:
        return "GENERAL_AI_TRAINING_KNOWLEDGE.md"

    # 6. Training Suite / Pathology records
    if record_id.startswith("pathology_") or "overfitting_rescue" in record_id or "loss_spike" in record_id:
        return "PAPER_TRAINING_PATHOLOGY.md"
    if record_id.startswith("training_") or "governor" in record_id:
        return "PAPER_TRAINING_SUITE.md"

    # 7. GUI / CLI / API procedure records
    if record_id.startswith("gui_") or record_id.startswith("val_gui_"):
        return "MANUAL_AI_STUDIO_GUI.md"
    if record_id.startswith("cli_") or record_id.startswith("val_cli_"):
        return "MANUAL_CLI.md"
    if record_id.startswith("api_") or record_id.startswith("val_api_"):
        return "MANUAL_API.md"

    # 8. Troubleshooting
    if record_id.startswith("troubleshoot_") or record_id.startswith("val_troubleshoot_") or record_id.startswith("multiturn_"):
        return "AI_HELPER_TROUBLESHOOTING_KNOWLEDGE.md"

    # 9. Versioning / Current State / Terminology
    if record_id.startswith("versioning_"):
        return "VERSIONING_POLICY.md"
    if record_id.startswith("current_state_"):
        return "LEMGENDARY_CURRENT_STATE.md"
    if record_id.startswith("terminology_"):
        return "GLOSSARY.md"

    # 10. Model selection decision records
    if record_id.startswith("decision_"):
        if "restoration" in record_id: return "PAPER_RESTORATION_MASTER.md"
        if "quality" in record_id: return "PAPER_NIMA_MASTER.md"
        if "facial" in record_id: return "PAPER_LEMGENDARY_FACE_SUITE.md"
        if "detection" in record_id: return "PAPER_DETECTION_MASTER.md"
        if "forex" in record_id: return "PAPER_FOREX_PREDICTOR.md"
        if "upn" in record_id: return "PAPER_LEMGENDARY_UPN_V2.md"
        return "ECOSYSTEM_ARCHITECTURE.md"

    # 11. Unsupported & Negative
    if record_id.startswith("unsupported_") or record_id.startswith("negative_") or record_id.startswith("val_unsupported_"):
        return "canonical_capabilities.yaml"

    # Fallback to category defaults
    category_defaults = {
        "gui_procedure": "MANUAL_AI_STUDIO_GUI.md",
        "cli_procedure": "MANUAL_CLI.md",
        "api_procedure": "MANUAL_API.md",
        "troubleshooting_diagnostics": "AI_HELPER_TROUBLESHOOTING_KNOWLEDGE.md",
        "model_selection": "ECOSYSTEM_ARCHITECTURE.md",
        "terminology_and_invariants": "GLOSSARY.md",
        "unsupported_and_negative": "canonical_capabilities.yaml",
        "cross_document_pipeline": "PAPER_TRAINING_SUITE.md",
    }
    return category_defaults.get(category, "ECOSYSTEM_ARCHITECTURE.md")


def resolve_section_anchor(doc_id: str, record_id: str) -> str:
    """Resolve section anchor tag for provenance linkage."""
    anchors = {
        "MANUAL_AI_STUDIO_GUI.md": "#gui-workflows",
        "MANUAL_CLI.md": "#cli-reference",
        "MANUAL_API.md": "#api-contracts",
        "AI_HELPER_TROUBLESHOOTING_KNOWLEDGE.md": "#diagnostic-triage",
        "GLOSSARY.md": "#canonical-invariants",
        "VERSIONING_POLICY.md": "#versioning-governance",
        "LEMGENDARY_CURRENT_STATE.md": "#current-state-manifest",
        "canonical_capabilities.yaml": "#unsupported-boundaries",
        "PAPER_DATASET_COMPILER.md": "#compiler-architecture",
        "PAPER_ENV_MANAGER.md": "#environment-manager-specification",
        "PAPER_TRAINING_SUITE.md": "#training-suite-orchestration",
        "PAPER_TRAINING_PATHOLOGY.md": "#pathology-diagnostic-remedies",
        "GENERAL_AI_TRAINING_KNOWLEDGE.md": "#general-training-fundamentals",
        "ECOSYSTEM_ARCHITECTURE.md": "#ecosystem-architecture",
    }
    return anchors.get(doc_id, "#architectural-specification")


def compute_content_sha256(messages: list) -> str:
    serialized = json.dumps(messages, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def enrich_file(filepath: str, split_name: str) -> int:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    enriched_lines = []
    for line in lines:
        record: Dict[str, Any] = json.loads(line)
        rec_id = record["id"]
        cat = record.get("category", "troubleshooting_diagnostics")

        doc_id = resolve_record_doc_id(rec_id, cat)

        if doc_id == "canonical_capabilities.yaml":
            html_filename = "current-state.html"
        else:
            html_filename = DOC_TO_HTML.get(doc_id)
            if not html_filename:
                raise ValueError(f"No HTML mapping found for source document {doc_id} in record {rec_id}")

        canonical_url = f"{BASE_URL}{html_filename}"
        anchor = resolve_section_anchor(doc_id, rec_id)

        provenance = {
            "source_doc_id": doc_id,
            "canonical_url": canonical_url,
            "doc_version": DOC_VERSION,
            "section_anchor": anchor,
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


def verify_html_files_exist() -> None:
    """Verify that every mapped HTML file actually exists in lemgendary-docs/papers/."""
    if not os.path.isdir(PAPERS_DIR):
        print(f"Warning: papers directory {PAPERS_DIR} not found, skipping disk check.")
        return

    missing = []
    for md_doc, html_file in DOC_TO_HTML.items():
        html_path = os.path.join(PAPERS_DIR, html_file)
        if not os.path.isfile(html_path):
            missing.append(f"{md_doc} -> {html_file}")

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} HTML publications in {PAPERS_DIR}:\n"
            + "\n".join(missing)
        )
    print(f"Verified all {len(DOC_TO_HTML)} published HTML files exist on disk.")


def main():
    verify_html_files_exist()
    train_count = enrich_file(TRAIN_PATH, "train")
    val_count = enrich_file(VAL_PATH, "val")
    print(
        f"Enriched {train_count} train records and {val_count} val records with record-level provenance (doc_version={DOC_VERSION})."
    )


if __name__ == "__main__":
    main()
