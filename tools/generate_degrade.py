"""
LemGendary Dataset Compiler — Batch Degradation Manifold Synthesizer.

Phase 6 of the 2026 modernization roadmap.

Synthesizes derived restoration and enhancement manifolds from clean source images
by applying composable degradation profiles, recording exact per-sample quantitative
parameter descriptors into labels/<split>/<name>.json, and logging provenance to
the manifold registry database.

Usage:
    python generate_degrade.py --source raw-sets/div2k --profile motion-blur+iso-noise \
                               --intensity medium --output LemGendizedNafNetDebluringSynthetic \
                               --pairs 50000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal, cast

_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "tools" else Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image
from tqdm import tqdm

from degrade import DynamicDegrader, parse_profile
from formats.transcode import ImageTranscoder
from core.config_schema import ImageFormatPolicy


_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})


def _discover_source_images(source_path: Path) -> list[Path]:
    """Find all image files within the source directory recursively."""
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")

    files: list[Path] = []
    # If source is a compiled manifold, prioritize images/ or targets/
    search_dirs = [source_path / "targets", source_path / "images", source_path]
    target_dir = source_path
    for candidate in search_dirs:
        if candidate.exists() and candidate.is_dir():
            target_dir = candidate
            break

    for root, _, filenames in os.walk(target_dir):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in _IMAGE_EXTS:
                files.append(Path(root) / f)

    files.sort()
    return files


def _init_synthetic_registry(db_path: Path) -> None:
    """Initialize manifold_registry.db schema for synthetic manifold."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                source TEXT NOT NULL,
                task TEXT NOT NULL,
                split TEXT NOT NULL,
                hash TEXT,
                perceptual_hash TEXT,
                img_format TEXT,
                img_size_bytes INTEGER,
                target_size_bytes INTEGER,
                mask_size_bytes INTEGER DEFAULT 0,
                is_hardlinked INTEGER DEFAULT 0,
                reject_code TEXT,
                audit_trail TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY,
                sample_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()


def process_single_sample(
    idx: int,
    src_file: Path,
    split: str,
    output_root: Path,
    degrader: DynamicDegrader,
    base_seed: int,
    transcoder: ImageTranscoder,
) -> dict[str, Any]:
    """Process, degrade, and write a single clean-degraded pair with parameter labels."""
    sample_seed = (base_seed * 1000003 + idx) % (2**31 - 1)
    stem = src_file.stem
    sample_name = f"{stem}_synth_{idx:06d}"

    # Load clean ground truth
    with Image.open(src_file) as clean_img:
        if clean_img.mode != "RGB":
            clean_img = clean_img.convert("RGB")
        clean_img.load()

    # Apply degradation directly as PIL image
    degraded_img, meta = degrader.degrade_pil(clean_img, sample_seed=sample_seed)

    # Encode images via transcoder
    deg_bytes, deg_fmt = transcoder.encode(degraded_img, kind="image")
    tgt_bytes, tgt_fmt = transcoder.encode(clean_img, kind="target")

    deg_ext = transcoder.extension_for(deg_fmt)
    tgt_ext = transcoder.extension_for(tgt_fmt)

    # Destination paths
    img_out = output_root / "images" / split / f"{sample_name}{deg_ext}"
    tgt_out = output_root / "targets" / split / f"{sample_name}{tgt_ext}"
    lbl_out = output_root / "labels" / split / f"{sample_name}.json"

    # Write files atomically
    img_out.write_bytes(deg_bytes)
    tgt_out.write_bytes(tgt_bytes)

    # Label payload detailing exact quantitative parameters
    label_payload = {
        "sample_name": sample_name,
        "source_file": src_file.name,
        "split": split,
        "seed": sample_seed,
        "profile": meta.get("profile", "unknown"),
        "degradations": meta.get("degradations", []),
        "input_dimensions": [clean_img.height, clean_img.width],
        "image_format": deg_fmt,
        "target_format": tgt_fmt,
    }
    lbl_out.write_text(json.dumps(label_payload, indent=2), encoding="utf-8")

    return {
        "name": sample_name,
        "source": src_file.name,
        "task": "restoration",
        "split": split,
        "img_format": deg_fmt,
        "img_size": len(deg_bytes),
        "target_size": len(tgt_bytes),
        "label_payload": label_payload,
    }


def synthesize_manifold(
    source: str,
    output: str,
    profile_expr: str = "motion-blur+iso-noise",
    intensity: str = "medium",
    pairs: int | None = None,
    val_split: float = 0.12,
    seed: int = 42,
    image_format: Literal["webp", "jpeg", "png", "keep"] = "webp",
    workers: int | None = None,
    dry_run: bool = False,
) -> int:
    """Main entry point for compiler-time manifold degradation synthesis."""
    source_path = Path(source).resolve()
    if not source_path.exists():
        # Check relative to datasets repository
        alt_path = Path("../LemGendaryDatasets") / source
        if alt_path.exists():
            source_path = alt_path.resolve()

    print(f"[DEGRADE] Discovering source images in {source_path}...")
    source_files = _discover_source_images(source_path)
    total_found = len(source_files)
    print(f"[DEGRADE] Discovered {total_found:,} candidate source images.")

    if total_found == 0:
        print(f"[ERROR] No image files found in {source_path}.")
        return 1

    if pairs is not None and pairs > 0:
        source_files = source_files[:pairs]
    total_samples = len(source_files)

    # Resolve output directory
    output_path = Path(output)
    if not output_path.is_absolute() and not str(output).startswith("."):
        output_path = Path("../LemGendaryDatasets") / output
    output_path = output_path.resolve()

    print(f"[DEGRADE] Target Manifold: {output_path}")
    print(f"[DEGRADE] Profile: {profile_expr} (Intensity: {intensity}, Base Seed: {seed})")
    print(f"[DEGRADE] Target Samples: {total_samples:,} (Val Split: {val_split:.1%})")

    if dry_run:
        print("[DEGRADE] Dry-run completed. No files were written.")
        return 0

    # Build directories
    for split in ("train", "val"):
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "targets" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Initialize Registry DB
    db_path = output_path / "manifold_registry.db"
    _init_synthetic_registry(db_path)

    # Setup profile & transcoder
    composite_prof = parse_profile(profile_expr, intensity=intensity)
    degrader = DynamicDegrader(composite_prof, seed=seed)
    policy = ImageFormatPolicy(format=image_format, quality=92, target_quality=95)
    transcoder = ImageTranscoder(policy)


    num_workers = workers or min(16, os.cpu_count() or 4)
    rng = random.Random(seed)

    print(f"[DEGRADE] Synthesizing {total_samples:,} pairs using {num_workers} parallel workers...")
    start_time = time.time()
    records: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for idx, src_file in enumerate(source_files):
            split = "val" if rng.random() < val_split else "train"
            fut = executor.submit(
                process_single_sample,
                idx,
                src_file,
                split,
                output_path,
                degrader,
                seed,
                transcoder,
            )
            futures.append(fut)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Synthesizing"):
            try:
                res = fut.result()
                records.append(res)
            except Exception as e:
                print(f"\n[WARNING] Failed to synthesize sample: {e}")

    # Commit metadata to registry in batch
    with sqlite3.connect(db_path) as conn:
        for r in records:
            conn.execute(
                """
                INSERT OR REPLACE INTO samples (
                    name, source, task, split, img_format, img_size_bytes, target_size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (r["name"], r["source"], r["task"], r["split"], r["img_format"], r["img_size"], r["target_size"]),
            )
            conn.execute(
                """
                INSERT INTO audit_events (sample_name, event_type, event_data)
                VALUES (?, ?, ?)
                """,
                (r["name"], "degradation_synthesis", json.dumps(r["label_payload"])),
            )
        conn.commit()

    # Generate metadata package (dataset_info.yaml, README.md)
    elapsed = time.time() - start_time
    total_written = len(records)

    info_yaml = {
        "dataset_name": output_path.name,
        "task": "restoration",
        "dataset_type": "vision_paired",
        "synthetic": True,
        "base_source": str(source_path.name),
        "degradation_profile": profile_expr,
        "intensity": intensity,
        "seed": seed,
        "total_samples": total_written,
        "train_samples": sum(1 for r in records if r["split"] == "train"),
        "val_samples": sum(1 for r in records if r["split"] == "val"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_path / "dataset_info.yaml").write_text(
        json.dumps(info_yaml, indent=2), encoding="utf-8"
    )

    readme_content = f"""# {output_path.name}

> Synthesized Restoration Dataset Manifold generated by LemGendary Dataset Compiler (Phase 6).

## Dataset Summary

- **Task**: Image Restoration & Enhancement
- **Base Ground Truth**: `{source_path.name}`
- **Degradation Profile**: `{profile_expr}`
- **Intensity**: `{intensity}`
- **Base Seed**: `{seed}`
- **Total Samples**: {total_written:,} (Train: {info_yaml['train_samples']:,}, Val: {info_yaml['val_samples']:,})
- **Format**: WebP (Images: q=92, Targets: q=95)

## Parameter Supervision

Every sample in `images/` has its clean ground truth in `targets/` and exact parameter descriptor in `labels/<split>/<name>.json`.
"""
    (output_path / "README.md").write_text(readme_content, encoding="utf-8")
    (output_path / "classes.txt").write_text("restoration\n", encoding="utf-8")

    print(
        f"[DEGRADE] Successfully synthesized {total_written:,} pairs into {output_path.name} "
        f"in {elapsed:.1f}s ({total_written / max(0.1, elapsed):.1f} pairs/sec)."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="LemGendary Dataset Compiler — Degradation Synthesizer")
    parser.add_argument("--source", "-s", required=True, help="Clean source dataset directory or manifold name")
    parser.add_argument("--output", "-o", required=True, help="Target manifold name or directory")
    parser.add_argument("--profile", "-p", default="motion-blur+iso-noise", help="Degradation profile expression or preset")
    parser.add_argument("--intensity", default="medium", choices=["low", "medium", "high"], help="Degradation intensity")
    parser.add_argument("--pairs", type=int, default=None, help="Max pairs to generate")
    parser.add_argument("--val-split", type=float, default=0.12, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic generation")
    parser.add_argument("--image-format", default="webp", choices=["webp", "jpeg", "png"], help="Output format")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Worker thread count")
    parser.add_argument("--dry-run", action="store_true", help="Preview plan without writing")

    args = parser.parse_args()
    fmt = cast(Literal["webp", "jpeg", "png", "keep"], args.image_format)
    code = synthesize_manifold(
        source=args.source,
        output=args.output,
        profile_expr=args.profile,
        intensity=args.intensity,
        pairs=args.pairs,
        val_split=args.val_split,
        seed=args.seed,
        image_format=fmt,
        workers=args.workers,
        dry_run=args.dry_run,
    )

    sys.exit(code)


if __name__ == "__main__":
    main()
