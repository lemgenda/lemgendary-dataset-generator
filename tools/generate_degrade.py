"""
LemGendary Dataset Compiler — Batch Degradation Manifold Synthesizer.

Phase 6 of the 2026 modernization roadmap.

Synthesizes derived restoration and enhancement manifolds from clean source images
by applying composable degradation profiles, recording exact per-sample quantitative
parameter descriptors into labels/<split>/<name>.json, and logging provenance to
the manifold registry database.

Usage:
    python -m tools.generate_degrade --source raw-sets/div2k --profile motion-blur+iso-noise \
                               --intensity medium --output LemGendizedNafNetDebluringSynthetic \
                               --pairs 50000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, Unpack, cast

from PIL import Image
from tqdm import tqdm

from core.config_schema import ImageFormatPolicy
from degrade import DynamicDegrader, parse_profile
from formats.transcode import ImageTranscoder
from tools.generate_degrade_registry import (
    ManifoldDescription,
    commit_records,
    init_synthetic_registry,
    write_metadata,
)

__all__ = [
    "DEFAULT_IMAGE_FORMAT",
    "DEFAULT_INTENSITY",
    "DEFAULT_PROFILE",
    "DEFAULT_SEED",
    "DEFAULT_VAL_SPLIT",
    "ImageFormat",
    "SampleContext",
    "SampleJob",
    "SynthesisOptions",
    "main",
    "process_single_sample",
    "synthesize_manifold",
]

logger = logging.getLogger(__name__)

ImageFormat = Literal["webp", "jpeg", "png", "keep"]

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


@dataclass(frozen=True)
class SampleContext:
    """Shared, read-only state every sample worker needs."""

    output_root: Path
    degrader: DynamicDegrader
    base_seed: int
    transcoder: ImageTranscoder


@dataclass(frozen=True)
class SampleJob:
    """One source image to degrade: its index, file and target split."""

    idx: int
    src_file: Path
    split: str


class SynthesisOptions(TypedDict, total=False):
    """Optional settings for :func:`synthesize_manifold` (defaults: ``DEFAULT_*``)."""

    profile_expr: str
    intensity: str
    pairs: int | None
    val_split: float
    seed: int
    image_format: ImageFormat
    workers: int | None
    dry_run: bool


DEFAULT_PROFILE = "motion-blur+iso-noise"
DEFAULT_INTENSITY = "medium"
DEFAULT_VAL_SPLIT = 0.12
DEFAULT_SEED = 42
DEFAULT_IMAGE_FORMAT: ImageFormat = "webp"


def _label_payload(
    job: SampleJob, sample_seed: int, meta: dict[str, Any], clean_img: Image.Image
) -> dict[str, Any]:
    """Return the label skeleton (formats are filled in after encoding)."""
    return {
        "sample_name": f"{job.src_file.stem}_synth_{job.idx:06d}",
        "source_file": job.src_file.name,
        "split": job.split,
        "seed": sample_seed,
        "profile": meta.get("profile", "unknown"),
        "degradations": meta.get("degradations", []),
        "input_dimensions": [clean_img.height, clean_img.width],
    }


def process_single_sample(job: SampleJob, ctx: SampleContext) -> dict[str, Any]:
    """Process, degrade, and write a single clean-degraded pair with parameter labels."""
    sample_seed = (ctx.base_seed * 1000003 + job.idx) % (2**31 - 1)

    # Load clean ground truth
    with Image.open(job.src_file) as clean_img:
        if clean_img.mode != "RGB":
            clean_img = clean_img.convert("RGB")
        clean_img.load()

    # Apply degradation directly as PIL image
    degraded_img, meta = ctx.degrader.degrade_pil(clean_img, sample_seed=sample_seed)
    label_payload = _label_payload(job, sample_seed, meta, clean_img)
    sample_name = label_payload["sample_name"]

    # Encode images via transcoder
    deg_bytes, deg_fmt = ctx.transcoder.encode(degraded_img, kind="image")
    tgt_bytes, tgt_fmt = ctx.transcoder.encode(clean_img, kind="target")

    # Write files atomically
    root = ctx.output_root
    deg_name = f"{sample_name}{ctx.transcoder.extension_for(deg_fmt)}"
    tgt_name = f"{sample_name}{ctx.transcoder.extension_for(tgt_fmt)}"
    (root / "images" / job.split / deg_name).write_bytes(deg_bytes)
    (root / "targets" / job.split / tgt_name).write_bytes(tgt_bytes)

    # Label payload detailing exact quantitative parameters
    label_payload["image_format"] = deg_fmt
    label_payload["target_format"] = tgt_fmt
    (root / "labels" / job.split / f"{sample_name}.json").write_text(
        json.dumps(label_payload, indent=2), encoding="utf-8"
    )

    return {
        "name": sample_name,
        "source": job.src_file.name,
        "task": "restoration",
        "split": job.split,
        "img_format": deg_fmt,
        "img_size": len(deg_bytes),
        "target_size": len(tgt_bytes),
        "label_payload": label_payload,
    }


def _resolve_source_path(source: str) -> Path:
    """Resolve ``source``, falling back to the sibling LemGendaryDatasets folder."""
    source_path = Path(source).resolve()
    if not source_path.exists():
        # Check relative to datasets repository
        alt_path = Path("../LemGendaryDatasets") / source
        if alt_path.exists():
            source_path = alt_path.resolve()
    return source_path


def _resolve_output_path(output: str) -> Path:
    """Resolve the output manifold directory (bare names go under LemGendaryDatasets)."""
    output_path = Path(output)
    if not output_path.is_absolute() and not str(output).startswith("."):
        output_path = Path("../LemGendaryDatasets") / output
    return output_path.resolve()


def _prepare_output_dirs(output_path: Path) -> None:
    """Create the images/targets/labels split folders."""
    for split in ("train", "val"):
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "targets" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)


def _build_context(desc: ManifoldDescription, image_format: ImageFormat) -> SampleContext:
    """Set up the degradation profile and transcoder shared by all samples."""
    composite_prof = parse_profile(desc.profile_expr, intensity=desc.intensity)
    degrader = DynamicDegrader(composite_prof, seed=desc.seed)
    policy = ImageFormatPolicy(format=image_format, quality=92, target_quality=95)
    return SampleContext(desc.output_path, degrader, desc.seed, ImageTranscoder(policy))


def _synthesize_all(
    ctx: SampleContext,
    source_files: list[Path],
    val_split: float,
    workers: int | None,
) -> list[dict[str, Any]]:
    """Degrade every source file on a thread pool; return the successful records."""
    num_workers = workers or min(16, os.cpu_count() or 4)
    rng = random.Random(ctx.base_seed)

    print(
        f"[DEGRADE] Synthesizing {len(source_files):,} pairs using {num_workers} parallel"
        " workers..."
    )
    records: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for idx, src_file in enumerate(source_files):
            split = "val" if rng.random() < val_split else "train"
            futures.append(
                executor.submit(process_single_sample, SampleJob(idx, src_file, split), ctx)
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Synthesizing"):
            try:
                records.append(fut.result())
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                logger.warning("Failed to synthesize sample: %s", e)
                print(f"\n[WARNING] Failed to synthesize sample: {e}")
    return records


def synthesize_manifold(
    source: str,
    output: str,
    **options: Unpack[SynthesisOptions],
) -> int:
    """Main entry point for compiler-time manifold degradation synthesis.

    ``options`` defaults: ``profile_expr="motion-blur+iso-noise"``,
    ``intensity="medium"``, ``pairs=None`` (all), ``val_split=0.12``,
    ``seed=42``, ``image_format="webp"``, ``workers=None`` (auto) and
    ``dry_run=False``.
    """
    source_path = _resolve_source_path(source)

    print(f"[DEGRADE] Discovering source images in {source_path}...")
    source_files = _discover_source_images(source_path)
    print(f"[DEGRADE] Discovered {len(source_files):,} candidate source images.")

    if not source_files:
        print(f"[ERROR] No image files found in {source_path}.")
        return 1

    pairs = options.get("pairs")
    if pairs is not None and pairs > 0:
        source_files = source_files[:pairs]
    val_split = options.get("val_split", DEFAULT_VAL_SPLIT)

    desc = ManifoldDescription(
        output_path=_resolve_output_path(output),
        source_path=source_path,
        profile_expr=options.get("profile_expr", DEFAULT_PROFILE),
        intensity=options.get("intensity", DEFAULT_INTENSITY),
        seed=options.get("seed", DEFAULT_SEED),
    )

    print(f"[DEGRADE] Target Manifold: {desc.output_path}")
    print(
        f"[DEGRADE] Profile: {desc.profile_expr} (Intensity: {desc.intensity}, Base Seed:"
        f" {desc.seed})"
    )
    print(f"[DEGRADE] Target Samples: {len(source_files):,} (Val Split: {val_split:.1%})")

    if options.get("dry_run", False):
        print("[DEGRADE] Dry-run completed. No files were written.")
        return 0

    _prepare_output_dirs(desc.output_path)
    db_path = desc.output_path / "manifold_registry.db"
    init_synthetic_registry(db_path)
    ctx = _build_context(desc, options.get("image_format", DEFAULT_IMAGE_FORMAT))

    start_time = time.time()
    records = _synthesize_all(ctx, source_files, val_split, options.get("workers"))
    commit_records(db_path, records)

    elapsed = time.time() - start_time
    total_written = write_metadata(desc, records)

    print(
        f"[DEGRADE] Successfully synthesized {total_written:,} pairs into {desc.output_path.name} "
        f"in {elapsed:.1f}s ({total_written / max(0.1, elapsed):.1f} pairs/sec)."
    )
    return 0


def main() -> None:
    """Synthesise a degraded manifold from a clean source (CLI entry point)."""
    parser = argparse.ArgumentParser(
        description="LemGendary Dataset Compiler — Degradation Synthesizer"
    )
    parser.add_argument(
        "--source", "-s", required=True, help="Clean source dataset directory or manifold name"
    )
    parser.add_argument("--output", "-o", required=True, help="Target manifold name or directory")
    parser.add_argument(
        "--profile",
        "-p",
        default="motion-blur+iso-noise",
        help="Degradation profile expression or preset",
    )
    parser.add_argument(
        "--intensity",
        default="medium",
        choices=["low", "medium", "high"],
        help="Degradation intensity",
    )
    parser.add_argument("--pairs", type=int, default=None, help="Max pairs to generate")
    parser.add_argument("--val-split", type=float, default=0.12, help="Validation split ratio")
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for deterministic generation"
    )
    parser.add_argument(
        "--image-format", default="webp", choices=["webp", "jpeg", "png"], help="Output format"
    )
    parser.add_argument("--workers", "-w", type=int, default=None, help="Worker thread count")
    parser.add_argument("--dry-run", action="store_true", help="Preview plan without writing")

    args = parser.parse_args()
    fmt = cast(ImageFormat, args.image_format)
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
