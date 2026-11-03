"""
LemGendary Dataset Compiler — Services Layer
============================================
Encapsulates all domain workflows into dedicated single-responsibility services
(a function for a single operation, a class for a family of operations),
allowing in-process programmatic invocation from the unified CLI, tests,
and external applications without relying on subprocess shelling.
"""

from __future__ import annotations

from .audit_service import audit_manifold
from .compiler_service import run_compile
from .degrade_service import run_degrade
from .doc_service import DocService
from .generation_service import run_generation
from .migration_service import MigrationService
from .sync_service import SyncService

__all__ = [
    "DocService",
    "MigrationService",
    "SyncService",
    "audit_manifold",
    "run_compile",
    "run_degrade",
    "run_generation",
]
