"""Controlled Domain Taxonomy and Authority Classification for LLM Tooling.

Enforces domain categorization, token boundary matching, and provenance tiering
to guarantee precision during instruction corpus synthesis.
"""

from __future__ import annotations

import re

# (category, doc_tokens, title_tokens)
CATEGORY_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("troubleshooting_diagnostic", ("trouble",), ("diagnostic", "failure")),
    ("gui_procedure", ("gui", "control"), ("interface",)),
    ("cli_command", ("cli",), ("command", "flag")),
    ("api_procedure", ("api",), ("endpoint", "rest")),
    ("terminology_concept", ("glossary",), ("terminology", "definition")),
    ("model_pipeline", ("paper",), ("architecture", "mathematical")),
)

CONTROLLED_DOMAINS: tuple[str, ...] = (
    "general_knowledge",
    "gui_procedure",
    "cli_command",
    "api_procedure",
    "troubleshooting_diagnostic",
    "model_pipeline",
    "terminology_concept",
)


def tokenize_text(text: str) -> set[str]:
    """Tokenize text into lowercase alphanumeric words for boundary-clean matching."""
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def classify_category(doc_id: str, section_title: str) -> str:
    """Classify knowledge chunk into domain categories using token boundaries."""
    doc_tokens = tokenize_text(doc_id)
    title_tokens = tokenize_text(section_title)
    for category, doc_keys, title_keys in CATEGORY_RULES:
        if any(k in doc_tokens for k in doc_keys) or any(k in title_tokens for k in title_keys):
            return category
    return "general_knowledge"


def resolve_authority_tier(doc_id: str) -> str:
    """Classify the document into an authoritative provenance tier."""
    lower = doc_id.lower()
    if any(k in lower for k in ("manifest", "canonical", "unified_data", "unified_models")):
        return "tier_1_manifest"
    if any(k in lower for k in ("manual", "cli", "api", "operations")):
        return "tier_2_operational_manual"
    if any(k in lower for k in ("paper", "architecture", "whitepaper", "specification")):
        return "tier_3_whitepaper"
    return "tier_4_educational"
