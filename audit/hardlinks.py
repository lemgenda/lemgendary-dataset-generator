"""
Hardlink fraction audit.

Single responsibility: report what proportion of files under a directory are
NTFS/POSIX hardlinks (st_nlink > 1). Used as a pre-flight gate before any
container-format write — MDS / LitData / FFCV inline bytes, which destroys
hardlink dedup, so a high hardlink fraction means the container would
duplicate terabytes of data.

Phase 2 of the 2026 modernization roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class HardlinkAuditResult:
    total_files: int
    hardlinked_files: int
    hardlink_pct: float
    total_bytes: int
    hardlinked_bytes: int

    @property
    def verdict(self) -> str:
        """Tiered gate used by Phase 4's container-write pre-flight.

        < 5%   PROCEED — negligible duplication cost
        5-25%  WARN    — container may still be net positive for shuffle/resume
        > 25%  BLOCK   — catastrophic duplication; keep directory layout
        """
        if self.hardlink_pct < 5.0:
            return "PROCEED"
        if self.hardlink_pct <= 25.0:
            return "WARN"
        return "BLOCK"

    def summary(self) -> str:
        return (
            f"{self.hardlinked_files}/{self.total_files} files hardlinked "
            f"({self.hardlink_pct:.1f}%, {self.hardlinked_bytes / 1024**3:.2f} GB) "
            f"-> {self.verdict}"
        )


def audit_hardlinks(root: str | Path, recursive: bool = True) -> HardlinkAuditResult:
    """Walk a directory and count files with st_nlink > 1.

    Uses os.scandir + DirEntry.stat() to avoid a second syscall per file.
    """
    root_path = Path(root)
    total = 0
    linked = 0
    total_bytes = 0
    linked_bytes = 0

    if not root_path.exists():
        return HardlinkAuditResult(0, 0, 0.0, 0, 0)

    stack: list[Path] = [root_path]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue

        for entry in entries:
            try:
                if entry.is_dir():
                    if recursive:
                        stack.append(entry)
                    continue
                if not entry.is_file():
                    continue
                st = entry.stat()
            except OSError:
                continue

            total += 1
            total_bytes += st.st_size
            if st.st_nlink > 1:
                linked += 1
                linked_bytes += st.st_size

    pct = (linked / total * 100.0) if total else 0.0
    return HardlinkAuditResult(
        total_files=total,
        hardlinked_files=linked,
        hardlink_pct=pct,
        total_bytes=total_bytes,
        hardlinked_bytes=linked_bytes,
    )
