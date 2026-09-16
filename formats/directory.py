"""
Canonical directory writer.

Single responsibility: write a Sample into the standard
`images/`, `labels/`, `targets/`, `masks/` layout, preserving hardlinks
where possible.

Phase 4 extracts the write logic currently inline in
`compiler_core.process_image`. Until then, this module is a stub.
"""

from __future__ import annotations

# Phase 4 will define: class DirectoryWriter(Writer)