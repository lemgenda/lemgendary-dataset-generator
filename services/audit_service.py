"""
AuditService — In-process manifold quality and hardlink auditing service.

Single Responsibility: Coordinate and execute image quality, dimensions,
corruptions, deduplication, and NTFS hardlink fraction audits.
Resolves the missing audit_cli.py and unifies audit functionality.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from rich.console import Console
from rich.table import Table

from audit.hardlinks import HardlinkAuditResult, audit_hardlinks
from audit.vision_audit import VisionAuditor

console = Console()
logger = logging.getLogger(__name__)


_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


@dataclass
class SampleAuditTally:
    """Pass count and per-code rejection counts of a vision sample audit."""

    audited: int = 0
    passed: int = 0
    rejections: dict[str, int] = field(default_factory=dict)

    def reject(self, code: str) -> None:
        """Count one rejection under ``code``."""
        self.rejections[code] = self.rejections.get(code, 0) + 1

    @property
    def pass_rate(self) -> float:
        """Percentage of audited samples that passed."""
        return (self.passed / max(1, self.audited)) * 100


def _collect_images(manifold_path: Path, sample: int | None) -> list[Path]:
    """Return image files under ``images/`` (or the manifold root), capped at ``sample``."""
    img_dir = manifold_path / "images"
    if not img_dir.exists():
        img_dir = manifold_path

    found_images: list[Path] = []
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
            found_images.append(p)
            if sample is not None and len(found_images) >= sample:
                break
    return found_images


def _audit_one(auditor: VisionAuditor, img_p: Path) -> str | None:
    """Audit one image; return its rejection code, or None when it passes."""
    try:
        with open(img_p, "rb") as f:
            header_bytes = f.read(16)
        header_res = auditor.verify_header(header_bytes)
        if not header_res.valid:
            return header_res.code or "ERR_HEADER_INVALID"

        with Image.open(img_p) as img:
            result = auditor.audit_image(img, task="quality", slug=img_p.stem)
            return None if result.valid else (result.code or "UNKNOWN")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        logger.debug("Audit decode failed for %s: %s", img_p, exc)
        return "ERR_DECODE_FAILED"


def _audit_samples(images: list[Path]) -> SampleAuditTally:
    """Run the vision auditor over ``images`` and tally the outcome."""
    auditor = VisionAuditor()
    tally = SampleAuditTally(audited=len(images))
    for img_p in images:
        code = _audit_one(auditor, img_p)
        if code is None:
            tally.passed += 1
        else:
            tally.reject(code)
    return tally


def _build_report(
    manifold_path: Path, hardlink_res: HardlinkAuditResult, tally: SampleAuditTally
) -> dict[str, Any]:
    """Return the structured (JSON) audit report."""
    return {
        "manifold": manifold_path.name,
        "path": str(manifold_path),
        "hardlinks": {
            "total_files": hardlink_res.total_files,
            "hardlinked_files": hardlink_res.hardlinked_files,
            "hardlink_pct": round(hardlink_res.hardlink_pct, 4),
            "total_bytes": hardlink_res.total_bytes,
            "hardlinked_bytes": hardlink_res.hardlinked_bytes,
            "verdict": hardlink_res.verdict,
        },
        "vision_audit": {
            "samples_audited": tally.audited,
            "passed": tally.passed,
            "pass_rate_pct": round(tally.pass_rate, 2),
            "rejections": tally.rejections,
        },
    }


def _render_table(
    manifold_path: Path, hardlink_res: HardlinkAuditResult, tally: SampleAuditTally
) -> Table:
    """Return the Rich table shown for a human-readable audit."""
    table = Table(title=f"Manifold Audit: {manifold_path.name}")
    table.add_column("Audit Metric", style="bold cyan")
    table.add_column("Measurement / Value", style="white")

    table.add_row("Total Files Checked", f"{hardlink_res.total_files:,}")
    table.add_row(
        "Hardlink Deduplication",
        f"{hardlink_res.hardlinked_files:,} ({hardlink_res.hardlink_pct:.1f}%) [Verdict:"
        f" {hardlink_res.verdict}]",
    )
    table.add_row(
        "Hardlinked Data",
        f"{hardlink_res.hardlinked_bytes / (1024**3):.2f} GB /"
        f" {hardlink_res.total_bytes / (1024**3):.2f} GB",
    )
    table.add_row("Audited Image Samples", f"{tally.audited:,}")
    table.add_row(
        "Sample Pass Rate", f"{tally.pass_rate:.1f}% ({tally.passed}/{tally.audited})"
    )

    if tally.rejections:
        table.add_row(
            "Rejection Codes", ", ".join(f"{k}: {v}" for k, v in tally.rejections.items())
        )
    else:
        table.add_row("Rejection Codes", "None (Clean)")
    return table


class AuditService:
    """Encapsulates manifold validation and diagnostic auditing."""

    @staticmethod
    def audit_manifold(
        manifold_path: Path,
        sample: int | None = None,
        as_json: bool = False,
    ) -> int:
        """Run comprehensive audit on a compiled manifold directory."""
        if not manifold_path.exists():
            console.print(f"[bold red]Manifold path does not exist:[/bold red] {manifold_path}")
            return 1

        # 1. Hardlink Audit
        hardlink_res = audit_hardlinks(manifold_path)

        # 2. Image Sample Audit
        tally = _audit_samples(_collect_images(manifold_path, sample))

        if as_json:
            print(json.dumps(_build_report(manifold_path, hardlink_res, tally), indent=2))
        else:
            console.print(_render_table(manifold_path, hardlink_res, tally))
        return 0 if hardlink_res.verdict != "BLOCK" else 1
