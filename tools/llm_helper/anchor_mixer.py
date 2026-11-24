"""Universal Anchor Mixer for LLM Fine-Tuning Corpora.

Prevents catastrophic forgetting during specialized domain fine-tuning by blending
a calibrated proportion of general reasoning, mathematics, and coding anchors.
"""

from __future__ import annotations

import random
from typing import Any

from .quality_validator import QualityValidator


class AnchorMixer:
    """Blends general reasoning and safety anchor records into domain splits."""

    def __init__(self, mix_ratio: float = 0.0, seed: int = 42) -> None:
        self.mix_ratio = max(0.0, min(0.5, mix_ratio))
        self.rng = random.Random(seed)
        self.validator = QualityValidator()

    def blend_records(
        self,
        domain_records: list[dict[str, Any]],
        anchor_records: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Blend anchor records into domain records according to mix_ratio."""
        if not anchor_records or self.mix_ratio <= 0.0:
            return list(domain_records)

        target_anchors = int(len(domain_records) * self.mix_ratio)
        if target_anchors == 0:
            return list(domain_records)

        sampled_anchors = (
            self.rng.sample(anchor_records, min(target_anchors, len(anchor_records)))
            if len(anchor_records) >= target_anchors
            else list(anchor_records)
        )

        # Validate anchors before blending to guarantee zero corruption
        clean_anchors = [
            rec for rec in sampled_anchors if len(self.validator.validate_record(rec)) == 0
        ]

        combined = list(domain_records) + clean_anchors
        self.rng.shuffle(combined)
        return combined

    def compute_anchor_count(self, total_domain_records: int) -> int:
        """Calculate target number of anchor samples given total domain records."""
        return int(total_domain_records * self.mix_ratio)
