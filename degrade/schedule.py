"""Deterministic degradation synthesis with configurable severity schedules."""

from __future__ import annotations

import hashlib
import io
import math
import random
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageFilter


def sample_seed(global_seed: int, sample_key: str) -> int:
    """Derive a deterministic 64-bit integer seed from ``global_seed`` and ``sample_key``."""
    digest = hashlib.sha256(f"{global_seed}:{sample_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class DegradationSchedule:
    """Configurable, deterministic degradation schedule for visual samples.

    Supports ``uniform``, ``linear``, and ``cosine`` severity schedules over
    ``[min_severity, max_severity]`` parameterized by normalized progress ``t in [0, 1]``.
    """

    global_seed: int = 42
    mode: str = "uniform"
    min_severity: float = 0.1
    max_severity: float = 0.9

    def severity_for(self, sample_key: str, progress: float = 0.5) -> float:
        """Compute deterministic severity in ``[min_severity, max_severity]``."""
        t = min(1.0, max(0.0, float(progress)))
        rng = random.Random(sample_seed(self.global_seed, sample_key))
        lo = float(self.min_severity)
        hi = float(self.max_severity)
        if self.mode == "linear":
            center = lo + (hi - lo) * t
            jitter = rng.uniform(-0.05, 0.05) * (hi - lo)
            return round(min(hi, max(lo, center + jitter)), 4)
        if self.mode == "cosine":
            w = 0.5 * (1.0 - math.cos(math.pi * t))
            center = lo + (hi - lo) * w
            jitter = rng.uniform(-0.05, 0.05) * (hi - lo)
            return round(min(hi, max(lo, center + jitter)), 4)
        return round(rng.uniform(lo, hi), 4)

    def apply(
        self,
        image_bytes: bytes,
        sample_key: str,
        progress: float = 0.5,
        output_format: str = "WEBP",
    ) -> tuple[bytes, dict[str, Any]]:
        """Apply deterministic blur + lossy compression to ``image_bytes``.

        Returns ``(degraded_bytes, degradation_record)`` for the sample's JSON sidecar.
        """
        seed = sample_seed(self.global_seed, sample_key)
        sev = self.severity_for(sample_key, progress=progress)
        radius = round(0.3 + sev * 1.7, 3)
        quality = round(95 - sev * 55)

        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB")
            degraded = rgb.filter(ImageFilter.GaussianBlur(radius=radius))
            buf = io.BytesIO()
            degraded.save(buf, format=output_format, quality=quality)

        record = {
            "seed": seed,
            "schedule_mode": self.mode,
            "severity": sev,
            "ops": [
                {"op": "gaussian_blur", "radius": radius},
                {"op": "lossy_encode", "format": output_format, "quality": quality},
            ],
        }
        return buf.getvalue(), record
