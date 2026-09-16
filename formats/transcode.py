"""
Image transcoding layer.

Single responsibility: convert in-memory images to the target storage
format (WebP q=92 default, WebP-lossless for masks, or pass-through for
`--image-format keep`).

Populated in Phase 3, not Phase 4. Transcoding runs before container
write so every downstream writer receives already-encoded bytes.
"""

from __future__ import annotations

# Phase 3 will define:
#     class ImageTranscoder:
#         def __init__(self, policy: ImageFormatPolicy): ...
#         def transcode(self, img: Image.Image, kind: str) -> tuple[bytes, str]: ...