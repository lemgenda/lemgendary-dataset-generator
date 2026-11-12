"""Collect samples rejected by a shard writer and report them.

Converters record each corrupt sample here instead of writing it. Every
rejection is logged; when a sink such as ``audit.reject_log.RejectLog().record``
is supplied the rejection is also persisted to the registry ``reject_log`` table.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = ["CORRUPT_IMAGE_CODE", "RejectSink", "RejectTally"]

CORRUPT_IMAGE_CODE = "corrupt_image"


# (source, sample_name, reject_code, reason), e.g. ``RejectLog(db).record``.
RejectSink = Callable[[str, str, str, str], None]


@dataclass
class RejectTally:
    """Rejected sample keys of one conversion, with their reasons."""

    source: str
    sink: RejectSink | None = None
    entries: list[tuple[str, str]] = field(default_factory=list)

    def record(self, key: str, reason: str) -> None:
        """Log and remember one rejected sample."""
        logger.warning("Rejected sample %s from %s: %s", key, self.source, reason)
        self.entries.append((key, reason))
        if self.sink is not None:
            self.sink(self.source, key, CORRUPT_IMAGE_CODE, reason)

    @property
    def keys(self) -> set[str]:
        """Return the keys of every rejected sample."""
        return {key for key, _ in self.entries}

    def report(self) -> None:
        """Print how many samples were rejected (nothing when none were)."""
        if self.entries:
            print(
                f"[REJECT] {len(self.entries)} corrupt samples from {self.source} were not"
                " written; see the log for each reason."
            )
