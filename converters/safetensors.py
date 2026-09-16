"""SafeTensors metadata reader. Copied verbatim from compiler_core.py in Phase 1.4."""

from __future__ import annotations

from safetensors import safe_open


def parse_safetensors(st_path):
    metadata = {}
    try:
        with safe_open(st_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
    except Exception:
        pass
    return metadata