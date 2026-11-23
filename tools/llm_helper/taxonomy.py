"""Controlled Domain Taxonomy and Authority Classification for LLM Tooling.

Enforces domain categorization, token boundary matching, provenance tiering,
and SSOT sidecar port lookups from ``core.ecosystem``.
"""

from __future__ import annotations

import functools
import re
from typing import Any

from core import ecosystem

HARDWARE_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "gpu": "NVIDIA GeForce GTX 1650",
        "vram_mb": 4096,
        "backend": "cuda",
        "turing_no_tensor_cores": True,
    },
    {
        "gpu": "NVIDIA GeForce RTX 3060",
        "vram_mb": 12288,
        "backend": "cuda",
        "turing_no_tensor_cores": False,
    },
    {
        "gpu": "NVIDIA Tesla T4",
        "vram_mb": 15360,
        "backend": "cuda",
        "turing_no_tensor_cores": False,
    },
    {
        "gpu": "Google Cloud TPU v5e-8",
        "vram_mb": 131072,
        "backend": "xla",
        "turing_no_tensor_cores": False,
    },
    {
        "gpu": "CPU Fallback (No CUDA)",
        "vram_mb": 0,
        "backend": "cpu",
        "turing_no_tensor_cores": False,
    },
)

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
    "boundary_and_negative",
)


@functools.lru_cache(maxsize=1)
def sidecar_ports() -> tuple[int, int, int]:
    """Return ``(env_port, compiler_port, training_port)`` from ``ecosystem.json``."""
    return (
        ecosystem.service_port(ecosystem.ENV_MANAGER_SERVICE),
        ecosystem.service_port(ecosystem.DATASETS_SERVICE),
        ecosystem.service_port(ecosystem.TRAINING_SERVICE),
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
