"""
LemGendary Dataset Compiler — Audit and Ground-Truth Loaders.

Phase 2 of the 2026 modernization roadmap.

Submodules:
    ground_truth   — dataset-format loaders (Phase 1.5.5, retained)
    vision_audit   — VisionAuditor (resolution, black-frame, NIMA gate)
    dedup          — ExactHasher, PerceptualHasher
    hardlinks      — hardlink fraction audit
    reject_log     — RejectLog (buffered writes to manifold_registry.db)
"""

from __future__ import annotations

__all__ = [
    "ExactHasher",
    "HardlinkAuditResult",
    "PerceptualHasher",
    "RejectLog",
    "VisionAuditor",
    "audit_hardlinks",
    "dedup",
    "ground_truth",
    "hardlinks",
    "reject_log",
    "vision_audit",
]

from . import dedup, ground_truth, hardlinks, reject_log, vision_audit
from .dedup import ExactHasher, PerceptualHasher
from .hardlinks import HardlinkAuditResult, audit_hardlinks
from .reject_log import RejectLog
from .vision_audit import VisionAuditor
