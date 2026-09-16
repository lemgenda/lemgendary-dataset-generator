"""HTTP helpers. Extracted from compiler_core.py in Phase 1.5.5."""

from __future__ import annotations

import os
import time

import requests


def download_image(url: str, dest_path: str, session=None) -> bool:
    """Lazy downloader with exponential backoff and image content-type validation."""
    if os.path.exists(dest_path):
        return True

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = (session or requests).get(url, headers=headers, timeout=15, stream=True)
            if r.status_code == 200:
                content_type = r.headers.get("Content-Type", "")
                if "image" not in content_type and "octet-stream" not in content_type:
                    return False
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
            elif r.status_code == 404:
                return False
        except Exception:
            if attempt == max_retries - 1:
                return False
            time.sleep(2 ** attempt)
    return False