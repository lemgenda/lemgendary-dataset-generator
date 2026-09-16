"""Content hashing. Extracted from compiler_core.py in Phase 1.5.5."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_hash(img_or_path, no_hash: bool = False):
    """Return the MD5 hex digest of a path, bytes, or PIL.Image.

    Returns None if ``no_hash`` is True. The no_hash flag is retained as an
    explicit parameter rather than reaching into a module-global args object
    (which was the pre-1.5.5 behavior in compiler_core.args).
    """
    if no_hash:
        return None
    if isinstance(img_or_path, (str, Path)):
        h = hashlib.md5()
        with open(img_or_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    elif isinstance(img_or_path, bytes):
        return hashlib.md5(img_or_path).hexdigest()
    else:
        # PIL.Image or numpy array with .tobytes()
        return hashlib.md5(img_or_path.tobytes()).hexdigest()