"""
JPEG compression artifacts.

Single responsibility: re-encode through JPEG at a configurable quality to
introduce block and ringing artifacts, then decode back to an array.

Phase 6.
"""

from __future__ import annotations