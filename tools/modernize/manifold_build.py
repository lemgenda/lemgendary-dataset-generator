"""Create modern manifold folders: copy structure/metadata and transcode images."""

from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image

from core.config_schema import ImageFormatPolicy
from formats.transcode import ImageTranscoder
from tools.modernize.common import IMAGE_EXTS
from tools.modernize.manifold_scan import (
    _has_manifold_data,
    _is_manifold_fully_modernized,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)

ImageKind = Literal["image", "target", "mask"]
TranscodeTask = tuple[Path, Path, ImageKind]


@dataclass
class _FilePlan:
    """Work needed to bring a modern manifold folder up to date with its source."""

    transcode_tasks: list[TranscodeTask] = field(default_factory=list)
    copy_tasks: list[tuple[Path, Path]] = field(default_factory=list)
    skipped_copies: int = 0
    skipped_images: int = 0

    def add_transcode(self, src_file: Path, target_file: Path, kind: ImageKind) -> None:
        """Plan a transcode unless a non-empty output already exists."""
        if target_file.exists() and target_file.stat().st_size > 0:
            self.skipped_images += 1
        else:
            self.transcode_tasks.append((src_file, target_file, kind))

    def add_copy(self, src_file: Path, target_file: Path) -> None:
        """Plan a copy unless a same-size copy already exists."""
        if target_file.exists() and target_file.stat().st_size == src_file.stat().st_size:
            self.skipped_copies += 1
        else:
            self.copy_tasks.append((src_file, target_file))


def _image_kind(rel_file: Path) -> ImageKind:
    if "masks" in rel_file.parts:
        return "mask"
    if "targets" in rel_file.parts:
        return "target"
    return "image"


def _output_ext(kind: ImageKind, policy: ImageFormatPolicy, transcoder: ImageTranscoder) -> str:
    if kind == "mask" and policy.mask_format == "webp-lossless":
        return ".webp"
    return ".webp" if policy.format == "webp" else transcoder.extension_for(policy.format)


def _plan_files(src_dir: Path, dst_dir: Path, transcoder: ImageTranscoder | None) -> _FilePlan:
    """Mirror the folder tree of ``src_dir`` under ``dst_dir`` and plan the file work.

    Manifold images are planned as transcodes with ``transcoder``, or as plain
    copies when it is None; every other file is copied.
    """
    plan = _FilePlan()
    for root, _dirs, files in os.walk(src_dir):
        rel_dir = Path(root).relative_to(src_dir)
        target_sub = dst_dir / rel_dir
        target_sub.mkdir(parents=True, exist_ok=True)
        for fname in files:
            src_file = Path(root) / fname
            rel_file = rel_dir / fname
            is_image = src_file.suffix.lower() in IMAGE_EXTS and any(
                part in rel_file.parts for part in ("images", "targets", "masks")
            )
            if is_image and transcoder is not None:
                kind = _image_kind(rel_file)
                ext = _output_ext(kind, transcoder.policy, transcoder)
                plan.add_transcode(src_file, target_sub / (src_file.stem + ext), kind)
            else:
                plan.add_copy(src_file, target_sub / fname)
    return plan


def _run_copies(plan: _FilePlan, dst_dir: Path) -> None:
    if plan.copy_tasks:
        print(f"[COPY] Copying {len(plan.copy_tasks)} structure, label, and metadata file(s)...")
        copy_iter = plan.copy_tasks
        if tqdm is not None and len(plan.copy_tasks) > 50:
            copy_iter = tqdm(
                plan.copy_tasks,
                desc=f"  -> Copying metadata [{dst_dir.name}]",
                unit="file",
                ncols=88,
                mininterval=0.5,
            )
        for s_f, d_f in copy_iter:
            try:
                shutil.copy2(s_f, d_f)
            except OSError as exc:
                logger.warning("Copy failed for %s -> %s: %s", s_f, d_f, exc)
    elif plan.skipped_copies > 0:
        print(
            f"[COPY] All {plan.skipped_copies} structure, label, and metadata file(s) already in"
            " place."
        )


def _transcode_one(transcoder: ImageTranscoder, task: TranscodeTask) -> bool:
    """Transcode one image; on failure fall back to a plain copy and return False."""
    s_p, d_p, k = task
    try:
        with Image.open(s_p) as img:
            data, _ = transcoder.encode(img, kind=k)
        with open(d_p, "wb") as f_out:
            f_out.write(data)
        return True
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Transcode failure for %s, falling back to copy: %s", s_p, exc)
        try:
            shutil.copy2(s_p, d_p)
        except OSError as copy_exc:
            logger.debug("Fallback copy2 failed for %s -> %s: %s", s_p, d_p, copy_exc)
        return False


def _run_transcodes(
    plan: _FilePlan, dst_dir: Path, transcoder: ImageTranscoder, max_workers: int
) -> None:
    fmt_upper = transcoder.policy.format.upper()
    tasks = plan.transcode_tasks
    if not tasks:
        if plan.skipped_images > 0:
            print(
                f"[TRANSCODE] All {plan.skipped_images} image(s) already converted to"
                f" {fmt_upper} in {dst_dir.name}."
            )
        return
    print(f"[TRANSCODE] Modernizing {len(tasks)} image(s) to {fmt_upper} in {dst_dir.name}...")

    def _worker(task: TranscodeTask) -> bool:
        return _transcode_one(transcoder, task)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_iter = executor.map(_worker, tasks)
        if tqdm is not None and len(tasks) > 20:
            name_disp = dst_dir.name if len(dst_dir.name) <= 24 else f"{dst_dir.name[:21]}..."
            results_iter = tqdm(
                results_iter,
                total=len(tasks),
                desc=f"  -> Transcode [{name_disp}]",
                unit="img",
                dynamic_ncols=True,
                bar_format=(
                    "{l_bar}{bar:15}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                ),
                mininterval=0.5,
            )
        results = list(results_iter)
    successes = sum(1 for r in results if r)
    print(f"[TRANSCODE] Completed: {successes}/{len(tasks)} images converted to {fmt_upper}.")


def _prepare_target(dst_dir: Path) -> bool:
    """Create ``dst_dir`` (reporting a resume of partial data); False on failure."""
    is_resume = dst_dir.exists() and _has_manifold_data(dst_dir)
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[HALT] Failed creating directory {dst_dir.name}: {e}")
        return False
    if is_resume:
        print(
            f"[RESUME] Target manifold exists with partial data: {dst_dir.name}. Checking"
            " for missing files..."
        )
    else:
        print(f"[CREATE] Created modern target directory: {dst_dir.name}")
    return True


def _create_modernized_batch(
    selected: list[dict],
    policy: ImageFormatPolicy,
    skip_transcode: bool,
    max_workers: int = 16,
) -> tuple[bool, list[dict]]:
    """Create modernized manifold directories alongside the legacy manifolds.

    The legacy manifold folders are preserved untouched.
    Returns (success, created_list).
    """
    created: list[dict] = []
    transcoder = ImageTranscoder(policy)
    image_transcoder = transcoder if not skip_transcode and policy.format != "keep" else None

    for item in selected:
        src_dir: Path = item["current_path"]
        dst_dir: Path = item["target_path"]

        if dst_dir.exists() and _is_manifold_fully_modernized(dst_dir):
            print(f"[SKIP] Target manifold already fully modernized: {dst_dir.name}")
            created.append(item)
            continue
        if not _prepare_target(dst_dir):
            return False, created

        plan = _plan_files(src_dir, dst_dir, image_transcoder)
        if plan.skipped_copies > 0 or plan.skipped_images > 0:
            print(
                f"[RESUME] Found existing files: {plan.skipped_images} image(s),"
                f" {plan.skipped_copies} metadata file(s) already complete."
            )
        _run_copies(plan, dst_dir)
        _run_transcodes(plan, dst_dir, transcoder, max_workers)
        created.append(item)
    return True, created
