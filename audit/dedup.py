"""
Duplicate detection.

Two independent hashers:

    ExactHasher       — MD5 of raw bytes. Catches byte-identical duplicates.
    PerceptualHasher  — pHash + dHash (64-bit each). Catches visual near-dups
                        within a configurable Hamming radius.

Both are stateless per-instance; instances are safe to share across threads.

Extracted from compiler_core.compute_hash and expanded in Phase 2 of the
2026 modernization roadmap.

The perceptual hashes are computed with stdlib numpy + scipy only — no
external imagehash dependency is required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct


class ExactHasher:
    """MD5 of raw file content. Preserves compatibility with pre-Phase-2
    registry hashes (which were also MD5)."""

    def __init__(self, no_hash: bool = False) -> None:
        self.no_hash = no_hash

    def hash(self, src: str | Path | bytes | Image.Image) -> str | None:
        if self.no_hash:
            return None

        if isinstance(src, (str, Path)):
            h = hashlib.md5()
            with open(src, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()

        if isinstance(src, bytes):
            return hashlib.md5(src).hexdigest()

        # PIL.Image or ndarray
        return hashlib.md5(src.tobytes()).hexdigest()


class PerceptualHasher:
    """pHash (DCT-based) + dHash (difference-based).

    Both produce a hex string of 16 chars (64 bits). Comparison is done via
    `hamming()`; two images are near-duplicates when EITHER pHash OR dHash
    distance is <= `near_dup_radius`.
    """

    def __init__(
        self,
        hash_size: int = 8,
        highfreq_factor: int = 4,
        near_dup_radius: int = 5,
        no_hash: bool = False,
    ) -> None:
        self.hash_size = hash_size
        self.highfreq_factor = highfreq_factor
        self.near_dup_radius = near_dup_radius
        self.no_hash = no_hash

    # ── Public API ──────────────────────────────────────────────────────────
    def hash(self, img: Image.Image) -> tuple[str, str] | None:
        """Return (phash_hex, dhash_hex), or None when no_hash is set."""
        if self.no_hash:
            return None
        return self._phash(img), self._dhash(img)

    @staticmethod
    def hamming(a: str, b: str) -> int:
        """Hamming distance between two equal-length hex strings."""
        ba = bytes.fromhex(a)
        bb = bytes.fromhex(b)
        arr_a = np.frombuffer(ba, dtype=np.uint8)
        arr_b = np.frombuffer(bb, dtype=np.uint8)
        return int(np.unpackbits(arr_a ^ arr_b).sum())

    def is_near_dup(self, h1: tuple[str, str], h2: tuple[str, str]) -> bool:
        """True when either pHash or dHash distance is within the radius."""
        p = self.hamming(h1[0], h2[0])
        d = self.hamming(h1[1], h2[1])
        return p <= self.near_dup_radius or d <= self.near_dup_radius

    # ── Internal hashes ─────────────────────────────────────────────────────
    def _phash(self, img: Image.Image) -> str:
        size = self.hash_size * self.highfreq_factor
        small = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        pixels = np.asarray(small, dtype=np.float32)
        # 2D DCT
        dct_rows = dct(pixels, axis=0, norm="ortho")
        dct_full = dct(dct_rows, axis=1, norm="ortho")
        # Take low-frequency block, drop DC term via median
        low = dct_full[: self.hash_size, : self.hash_size]
        med = float(np.median(low))
        bits = (low > med).flatten()
        packed = np.packbits(bits)
        return packed.tobytes().hex()

    def _dhash(self, img: Image.Image) -> str:
        # Resize to (hash_size) rows x (hash_size+1) columns
        small = img.convert("L").resize(
            (self.hash_size + 1, self.hash_size),
            Image.Resampling.LANCZOS,
        )
        pixels = np.asarray(small, dtype=np.int16)
        # Horizontal gradient
        diff = pixels[:, 1:] > pixels[:, :-1]
        packed = np.packbits(diff.flatten())
        return packed.tobytes().hex()