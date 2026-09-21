"""
LemGendary Dataset Compiler — Regenerate manifolds.md

Rebuilds manifolds.md from:
  - unified_data.yaml (structure, refs)
  - dataset_info.yaml inside each manifold folder (task, category)
  - manifold_registry.db or disk scan (authoritative sample counts)
  - models_metadata.yaml (bound models)

Scope: regenerates the master matrix and per-manifold spec sections.

Usage:
    python regenerate_manifolds_md.py
    python regenerate_manifolds_md.py --check    # dry-run, print diff summary
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_YAML = ROOT / "unified_data.yaml"
MODELS_META_YAML = ROOT / "models" / "models_metadata.yaml"
if not MODELS_META_YAML.exists():
    MODELS_META_YAML = ROOT / "models_metadata.yaml"
OUTPUT_MD = ROOT / "manifolds.md"


# ─── Paths & Registry ───────────────────────────────────────────────────────
def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _out_parent(reg: dict) -> Path:
    folder = reg.get("_registry_metadata", {}).get(
        "output_folder_name", "../LemGendaryDatasets"
    )
    return (ROOT / folder).resolve()


# ─── Count Strategies ───────────────────────────────────────────────────────
def _count_from_registry(manifold_path: Path) -> int | None:
    """Prefer registry DB if present."""
    for db_name in ("manifold_registry.db",):
        db = manifold_path / db_name
        if not db.exists():
            continue
        try:
            conn = sqlite3.connect(str(db))
            cur = conn.execute("SELECT COUNT(*) FROM registry")
            n = cur.fetchone()[0]
            conn.close()
            return int(n)
        except Exception as exc:
            logger.debug("Could not read count from registry %s: %s", db, exc)
    # Legacy location
    legacy = ROOT / ".cache" / f"registry_{manifold_path.name}.db"
    if legacy.exists():
        try:
            conn = sqlite3.connect(str(legacy))
            cur = conn.execute("SELECT COUNT(*) FROM registry")
            n = cur.fetchone()[0]
            conn.close()
            return int(n)
        except Exception as exc:
            logger.debug("Could not read count from legacy registry %s: %s", legacy, exc)
    return None


def _count_from_disk(manifold_path: Path) -> int:
    """Count samples directly. Handles both image manifolds and Forex parquet."""
    # Forex: sum rows across *.parquet directly in folder
    pq_files = list(manifold_path.glob("*.parquet"))
    if pq_files:
        total = 0
        try:
            import pyarrow.parquet as pq
            for pqf in pq_files:
                try:
                    meta = pq.read_metadata(str(pqf))
                    total += meta.num_rows
                except Exception as exc:
                    logger.debug("Could not read parquet metadata for %s: %s", pqf, exc)
        except ImportError as exc:
            logger.debug("pyarrow unavailable for counting parquet rows: %s", exc)
        if total:
            return total

    # Image manifolds: count files under images/ across all splits
    total = 0
    for split in ("train", "val", "test"):
        d = manifold_path / "images" / split
        if d.exists():
            try:
                total += sum(1 for f in d.iterdir() if f.is_file())
            except OSError as exc:
                logger.debug("Failed scanning split directory %s: %s", d, exc)
    return total


def _sample_count(manifold_path: Path) -> int:
    n = _count_from_registry(manifold_path)
    if n is not None and n > 0:
        return n
    return _count_from_disk(manifold_path)


# ─── Per-Manifold Metadata ──────────────────────────────────────────────────
def _manifold_info(manifold_path: Path) -> dict:
    info = _load_yaml(manifold_path / "dataset_info.yaml")
    return {
        "task": info.get("task", "unknown"),
        "category": info.get("category", ""),
        "original_sources": info.get("original_sources", []),
    }


# ─── Table Builders ─────────────────────────────────────────────────────────
def _format_domain(task: str) -> str:
    if task == "forex":
        return "Financial & Time-Series"
    if task in ("diffusion", "vlm"):
        return "Image Generation & Multimodal"
    return "Image Manipulation & Restoration"


def _format_storage(manifold_path: Path) -> str:
    if list(manifold_path.glob("*.parquet")):
        return "Parquet (Annual Shards)"
    if (manifold_path / "mds").exists():
        return "MDS + Directory"
    if (manifold_path / "litdata").exists():
        return "LitData + Directory"
    if (manifold_path / "shards").exists():
        return "WebDataset + Directory"
    if (manifold_path / "targets").exists():
        return "Directory Pair (images/, targets/)"
    return "Directory Pair (images/, labels/)"


def _bound_models(models_meta: dict, manifold_name: str) -> list[str]:
    out = []
    for m_key, m_info in models_meta.items():
        if not isinstance(m_info, dict):
            continue
        if manifold_name in m_info.get("datasets", []):
            out.append(m_key)
    return out


# ─── Master Matrix ──────────────────────────────────────────────────────────
def _build_matrix(reg: dict, models_meta: dict) -> list[dict]:
    out = _out_parent(reg)
    if not out.exists():
        return []

    rows = []
    for folder in sorted(out.iterdir()):
        if not folder.is_dir():
            continue
        if not folder.name.startswith("LemGendized"):
            continue

        info = _manifold_info(folder)
        count = _sample_count(folder)

        # Match registry entry by name
        dataset_key = None
        kaggle_ref = ""
        for key, entry in reg.get("datasets", {}).items():
            slug = entry.get("name", "")
            if folder.name == f"LemGendized{slug}Large" or folder.name == f"LemGendized{slug}":
                dataset_key = key
                kaggle_ref = entry.get("kaggle_ref", "")
                break

        rows.append({
            "name": folder.name,
            "domain": _format_domain(info["task"]),
            "task": info["task"],
            "samples": count,
            "format": _format_storage(folder),
            "sources": ", ".join(info["original_sources"][:3]) or "—",
            "models": ", ".join(_bound_models(models_meta, folder.name)) or "—",
            "kaggle_ref": kaggle_ref or "—",
            "path": folder,
        })
    return rows


# ─── Markdown Rendering ─────────────────────────────────────────────────────
def _render(rows: list[dict], reg: dict) -> str:
    lines: list[str] = []
    lines.append("# LemGendary Dataset Manifolds Directory & Lineage Matrix")
    lines.append("")
    lines.append(
        "This document provides a comprehensive operational index, physical file "
        "structure taxonomy, sample distributions, lineage tracking, and model "
        "bindings for all production dataset manifolds in the **LemGendary Dataset Suite**."
    )
    lines.append("")
    lines.append(f"*Auto-generated by `regenerate_manifolds_md.py` on "
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.*")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Master Manifolds Matrix")
    lines.append("")
    lines.append(
        "| Manifold Name | Domain | Task | Samples | Format | Upstream Sources | Bound Model(s) | Kaggle Reference |"
    )
    lines.append("| :--- | :--- | :--- | ---: | :--- | :--- | :--- | :--- |")
    for r in rows:
        lines.append(
            f"| `{r['name']}` | {r['domain']} | `{r['task']}` | {r['samples']:,} | "
            f"{r['format']} | {r['sources']} | {r['models']} | `{r['kaggle_ref']}` |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Manifold Specifications")
    lines.append("")

    for i, r in enumerate(rows, 1):
        lines.append(f"### 2.{i} {r['name']}")
        lines.append("")
        lines.append(f"- **Domain**: {r['domain']}")
        lines.append(f"- **Task Category**: `{r['task']}`")
        lines.append(f"- **Total Samples**: {r['samples']:,}")
        lines.append(f"- **Storage Format**: {r['format']}")
        if r["kaggle_ref"] != "—":
            lines.append(f"- **Kaggle Reference**: `{r['kaggle_ref']}`")
        lines.append("")
        # Directory layout
        lines.append("```text")
        lines.append(f"{r['name']}/")
        for child in sorted(p.name for p in r["path"].iterdir() if p.is_dir()):
            lines.append(f"├── {child}/")
        for child in sorted(p.name for p in r["path"].iterdir() if p.is_file()):
            lines.append(f"├── {child}")
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*End of manifolds directory.*")
    lines.append("")
    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────────────
def main(args: list[str] | argparse.Namespace | None = None) -> int:
    if isinstance(args, argparse.Namespace):
        parsed = args
    else:
        parser = argparse.ArgumentParser(description="Regenerate manifolds.md")
        parser.add_argument("--check", action="store_true",
                            help="Print summary, do not write the file")
        parsed = parser.parse_args(args)

    reg = _load_yaml(REGISTRY_YAML)
    if not reg:
        print(f"[ERROR] Could not load registry: {REGISTRY_YAML}")
        return 1

    models_meta = _load_yaml(MODELS_META_YAML).get("models_metadata", {})
    rows = _build_matrix(reg, models_meta)

    if not rows:
        print("[WARN] No manifolds found on disk.")
        return 0

    content = _render(rows, reg)

    if parsed.check:
        print(f"[CHECK] Would write {len(content)} bytes, {len(rows)} manifolds.")
        for r in rows:
            print(f"  {r['name']:<52} {r['samples']:>12,}  ({r['task']})")
        return 0

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Wrote {OUTPUT_MD} ({len(rows)} manifolds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())