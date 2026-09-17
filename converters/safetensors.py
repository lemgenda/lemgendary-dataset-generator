"""SafeTensors metadata reader. Extracted from compiler_core.py in Phase 1.4."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from safetensors import safe_open

logger = logging.getLogger(__name__)


def parse_safetensors(st_path: str | Path) -> dict[str, Any]:
    """Read metadata from a .safetensors file's header.

    Returns an empty dict if the file has no metadata or fails to open.
    The compiler uses this for Kohya / Civitai tags embedded in model
    metadata; missing metadata is not an error.
    """
    metadata: dict[str, Any] = {}
    try:
        with safe_open(str(st_path), framework="pt", device="cpu") as f:
            meta = f.metadata()
            if meta is not None:
                metadata = dict(meta)
    except (OSError, RuntimeError) as exc:
        # Corrupt or non-safetensors file → log diagnostic and treat metadata as absent.
        logger.debug("Failed reading safetensors metadata from %s: %s", st_path, exc)
    return metadata