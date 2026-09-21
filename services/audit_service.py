"""
AuditService — In-process manifold quality and hardlink auditing service.

Single Responsibility: Coordinate and execute image quality, dimensions,
corruptions, deduplication, and NTFS hardlink fraction audits.
Resolves the missing audit_cli.py and unifies audit functionality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from audit.hardlinks import audit_hardlinks
from audit.vision_audit import VisionAuditor

console = Console()


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
        auditor = VisionAuditor()
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        img_dir = manifold_path / "images"
        if not img_dir.exists():
            img_dir = manifold_path

        found_images: list[Path] = []
        for p in img_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in image_extensions:
                found_images.append(p)
                if sample is not None and len(found_images) >= sample:
                    break

        audited_count = len(found_images)
        passed_count = 0
        rejection_breakdown: dict[str, int] = {}

        for img_p in found_images:
            try:
                with open(img_p, "rb") as f:
                    header_bytes = f.read(16)
                header_res = auditor.verify_header(header_bytes)
                if not header_res.valid:
                    rejection_breakdown[header_res.code or "ERR_HEADER_INVALID"] = rejection_breakdown.get(header_res.code or "ERR_HEADER_INVALID", 0) + 1
                    continue

                from PIL import Image
                with Image.open(img_p) as img:
                    result = auditor.audit_image(img, task="quality", slug=img_p.stem)
                    if result.valid:
                        passed_count += 1
                    else:
                        code = result.code or "UNKNOWN"
                        rejection_breakdown[code] = rejection_breakdown.get(code, 0) + 1
            except Exception:
                code = "ERR_DECODE_FAILED"
                rejection_breakdown[code] = rejection_breakdown.get(code, 0) + 1

        pass_rate = (passed_count / max(1, audited_count)) * 100

        # Structured result
        report: dict[str, Any] = {
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
                "samples_audited": audited_count,
                "passed": passed_count,
                "pass_rate_pct": round(pass_rate, 2),
                "rejections": rejection_breakdown,
            },
        }

        if as_json:
            print(json.dumps(report, indent=2))
            return 0 if hardlink_res.verdict != "BLOCK" else 1

        # Render Rich Table
        table = Table(title=f"Manifold Audit: {manifold_path.name}")
        table.add_column("Audit Metric", style="bold cyan")
        table.add_column("Measurement / Value", style="white")

        table.add_row("Total Files Checked", f"{hardlink_res.total_files:,}")
        table.add_row(
            "Hardlink Deduplication",
            f"{hardlink_res.hardlinked_files:,} ({hardlink_res.hardlink_pct:.1f}%) [Verdict: {hardlink_res.verdict}]"
        )
        table.add_row("Hardlinked Data", f"{hardlink_res.hardlinked_bytes / (1024**3):.2f} GB / {hardlink_res.total_bytes / (1024**3):.2f} GB")
        table.add_row("Audited Image Samples", f"{audited_count:,}")
        table.add_row("Sample Pass Rate", f"{pass_rate:.1f}% ({passed_count}/{audited_count})")

        if rejection_breakdown:
            table.add_row("Rejection Codes", ", ".join(f"{k}: {v}" for k, v in rejection_breakdown.items()))
        else:
            table.add_row("Rejection Codes", "None (Clean)")

        console.print(table)
        return 0 if hardlink_res.verdict != "BLOCK" else 1
