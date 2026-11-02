"""Compile outcome bookkeeping and the post-compile loose-image purge.

``manifold_compile.process_dataset`` reports one ``ManifoldOutcome`` per
manifold it handled. ``finalize_compile`` turns those outcomes into the
process exit code (a compile that produced nothing fails) and runs the
optional loose-image purge only after every manifold compiled successfully.

Kept free of torch / heavy imports so it can be tested in isolation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from formats.directory import purge_loose_layout


@dataclass(frozen=True)
class ManifoldOutcome:
    """What a compile run produced for a single manifold."""

    model_key: str
    output_root: Path
    samples: int
    # Packed (non-directory) containers written successfully in this run.
    containers_written: tuple[str, ...] = ()
    # False for manifolds that are not sample-indexed (forex); those raise on failure.
    sample_based: bool = True

    @property
    def compiled(self) -> bool:
        return (not self.sample_based) or self.samples > 0


def finalize_compile(
    outcomes: Sequence[ManifoldOutcome],
    purge_loose: bool,
    purge: Callable[[Path], list[Path]] = purge_loose_layout,
) -> int:
    """Return the compile exit code and purge loose images when it is safe.

    * No outcome, or any manifold with zero samples -> exit 1, nothing purged.
    * With ``purge_loose``, a manifold's loose layout is removed only when a
      packed container was written for it in this run; otherwise the loose
      layout is the only copy and is kept.
    """
    if not outcomes:
        print("[FAILED] No manifold was compiled: the requested model is not in unified_data.yaml "
              "or produced no work.")
        return 1

    failed = [o.model_key for o in outcomes if not o.compiled]
    if failed:
        print(f"[FAILED] Compile produced no samples for: {', '.join(failed)}. Loose images were not purged.")
        return 1

    if purge_loose:
        for outcome in outcomes:
            if not outcome.sample_based:
                continue
            if not outcome.containers_written:
                print(f"[JANITOR] Keeping loose images for {outcome.model_key}: no packed container was "
                      "written, so the directory layout is the only copy.")
                continue
            removed = purge(outcome.output_root)
            print(f"[JANITOR] Purged loose layout for {outcome.model_key} "
                  f"({', '.join(p.name for p in removed) or 'nothing to remove'}); "
                  f"containers: {', '.join(outcome.containers_written)}.")
    return 0
