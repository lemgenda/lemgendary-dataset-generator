"""Composition & Lineage section of the standard manifold README.

Follows dataset-card practice (Hugging Face dataset cards, Kaggle dataset
documentation, "Datasheets for Datasets"): the real split counts are reported
once for the whole manifold, the sources are listed with their upstream links,
license note and citations, and per-source counts appear only when the sample
index records them exactly. A count that is not recorded is shown as such,
never estimated.
"""

from __future__ import annotations

from typing import Any

from core.docs.context import ManifoldContext
from core.docs.context_types import SplitCounts
from core.docs.templates import wrap_bullet, wrap_paragraph

NOT_RECORDED = "not recorded"

# (ref scheme, public URL prefix); refs of other schemes are not public pages.
_REF_URLS = (
    ("kaggle://", "https://www.kaggle.com/datasets/"),
    ("hf://", "https://huggingface.co/datasets/"),
    ("gh://", "https://github.com/"),
)


def format_count(value: int | None) -> str:
    """Thousands-separated count, or ``not recorded`` when the manifold has no record."""
    return f"{value:,}" if isinstance(value, int) else NOT_RECORDED


def ref_url(ref: str) -> str | None:
    """Public page of a registry ref (Kaggle, Hugging Face, GitHub, http), else ``None``."""
    if ref.startswith(("https://", "http://")):
        return ref
    for scheme, prefix in _REF_URLS:
        if ref.startswith(scheme):
            path = ref[len(scheme):].split(":", 1)[0].strip("/")
            return f"{prefix}{path}" if path else None
    return None


def _per_source_counted(sources: dict[str, dict[str, int | None]]) -> bool:
    """True when every source row carries exact train, val and total counts."""
    return bool(sources) and all(
        isinstance(counts.get(key), int)
        for counts in sources.values()
        for key in ("train", "val", "total")
    )


def _display_name(src: str) -> str:
    """Source name trimmed for the table column (36 characters at most)."""
    disp_src = src.split(" (", 1)[0] if len(src) > 36 and " (" in src else src
    if len(disp_src) > 36:
        disp_src = disp_src[:33].rstrip() + "..."
    return disp_src


def _split_table(splits: SplitCounts, total_samples: int) -> str:
    """The split counts of the whole manifold, stated once."""
    return "\n".join(
        [
            "| Split | Samples |",
            "| :--- | :--- |",
            f"| Train | {format_count(splits.train)} |",
            f"| Val | {format_count(splits.val)} |",
            f"| Total | {format_count(total_samples or None)} |",
        ]
    )


def _source_table(sources: dict[str, dict[str, int | None]]) -> str:
    """Exact per-source rows, largest source first."""
    rows = [
        "| Source Dataset | Train | Val | Total |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for src, counts in sorted(
        sources.items(), key=lambda kv: kv[1].get("total") or 0, reverse=True
    ):
        rows.append(
            f"| **{_display_name(src)}** | {format_count(counts.get('train'))} |"
            f" {format_count(counts.get('val'))} | {format_count(counts.get('total'))} |"
        )
    return "\n".join(rows)


def _source_list(names: list[str]) -> str:
    """The sources as a bullet list (no per-source counts)."""
    return "\n".join(wrap_bullet(f"- {name}") for name in names)


def _reference_lines(entry: dict[str, Any], citations: list[str]) -> list[str]:
    """Upstream links, the license note and the citations."""
    refs = [r.get("ref", "") if isinstance(r, dict) else str(r) for r in entry.get("refs", [])]
    urls = list(dict.fromkeys(u for u in (ref_url(str(r)) for r in refs) if u))
    lines: list[str] = []
    if urls:
        lines += ["", "Upstream sources:", ""]
        lines += [wrap_bullet(f"- <{u}>") for u in urls]
    if citations:
        lines += ["", "Citations:", ""]
        lines += [wrap_bullet(f"- <{c}>" if c.startswith("http") else f"- {c}") for c in citations]
    return lines


def build_composition_text(ctx: ManifoldContext) -> str:
    """Markdown body of the Composition & Lineage section."""
    lineage = ctx.lineage
    sources = lineage.sources
    parts = [_split_table(lineage.splits, lineage.total_samples), ""]
    if _per_source_counted(sources):
        parts += [
            wrap_paragraph(
                "Per-source counts, taken exactly from the sample keys and source tags "
                "in index.json:"
            ),
            "",
            _source_table(sources),
        ]
    elif sources:
        parts += [
            wrap_paragraph(
                "Sources merged into this manifold. The sample index does not record "
                "which source each sample came from, so no per-source counts are given."
            ),
            "",
            _source_list(list(sources)),
        ]
    else:
        parts.append(wrap_paragraph("No source breakdown is recorded for this manifold."))
    parts += [
        "",
        wrap_paragraph(
            f"The manifold is published under {ctx.listing.license_name}. Each upstream "
            "source keeps its own license and terms of use; check them on the source "
            "pages before reuse."
        ),
    ]
    parts += _reference_lines(ctx.spec.unified_entry, lineage.citations)
    return "\n".join(parts)
