"""Compile the datasets in unified_data.yaml into manifolds.

Run as ``python -m core.manifold_compile``.

This module is the entry point and the per-manifold orchestration; the
steps live in:

  - core.compile_setup      CLI parsing, pre-flight, worker pool
  - core.compile_forex      forex (MT5) manifolds
  - core.compile_registry   resumption scans, orphan adoption, registry inserts
  - core.compile_sources    ref resolution, annotations, image scans, downloads
  - core.compile_tasks      worker task planning and NSFW balancing
  - core.compile_passes     PASS 1 extraction, style clustering, PASS 2 index
  - core.compile_notebooks  training notebooks from the training suite, i.e.
                            ``from training.notebooks import generate_training_notebook``
"""

import argparse
import sys
from pathlib import Path
from typing import Any, cast

from core.compile_forex import compile_forex_manifold, is_forex
from core.compile_notebooks import (
    get_training_suite_notebook_generators,
    write_manifold_notebooks,
)
from core.compile_outcome import ManifoldOutcome, finalize_compile
from core.compile_passes import (
    ExtractionRun,
    cluster_styles,
    run_extraction,
    write_containers,
    write_index,
)
from core.compile_registry import (
    adopt_orphans,
    load_existing_names,
    open_registry,
    scan_physical_index,
)
from core.compile_setup import (
    CompileSettings,
    install_interrupt_handler,
    make_executor,
    parse_compile_args,
    prepare_compile,
)
from core.compile_sources import load_annotations, resolve_ref, scan_images
from core.compile_tasks import (
    CocoAnnData,
    ManifoldContext,
    MatlabAnnData,
    ParquetAnnData,
    TaskItem,
    balance_nsfw,
    image_tasks,
    shard_tasks,
)
from core.compiler_core import (
    DATASETS_META,
    OUT_PARENT,
    ExistingSamples,
    detect_task,
    remove_empty_dirs,
)
from core.doc_generator import generate_dataset_docs
from core.task_taxonomy import taxonomy
from runtime.environment import load_torch

# torch is loaded after the runtime env vars are set (see runtime.environment).
torch = load_torch()

__all__ = [
    "CocoAnnData",
    "MatlabAnnData",
    "ParquetAnnData",
    "TaskItem",
    "finish_manifold",
    "parse_compile_args",
    "process_dataset",
]

# Above this many tasks a CPU-only run drops labeling and vetting.
_CPU_GUARD_TASKS = 50000


def _get_training_suite_notebook_generators():
    """Return the notebook generators from lemgendary-training-suite (Single Source of Truth).

    The training suite owns notebook generation; this is equivalent to::

        from training.notebooks import generate_training_notebook, generate_colab_training_notebook

    with the sibling checkout put on ``sys.path`` first (see
    ``core.compile_notebooks.get_training_suite_notebook_generators``).
    """
    return get_training_suite_notebook_generators()


def _create_layout(output_root: Path, model_config: dict[str, Any], task: str) -> None:
    """Create the images/ (and labels/, masks/ or targets/) split folders of a new manifold."""
    if output_root.exists() or is_forex(model_config):
        return
    subdirs = ["images"]
    layout_dir = taxonomy().layout_dir(task)
    if layout_dir:
        subdirs.append(layout_dir)
    for subdir in subdirs:
        for s in ["train", "val"]:
            (output_root / subdir / s).mkdir(parents=True, exist_ok=True)


def _plan_tasks(ctx: ManifoldContext, affixes: tuple[str, str]) -> list[TaskItem]:
    """Plan the worker tasks of every source ref; NSFW sources are ratio-balanced."""
    sfw_tasks: list[TaskItem] = []
    nsfw_tasks: list[TaskItem] = []
    for ref_entry in cast("list[dict[str, Any]]", ctx.model_config.get("refs", [])):
        src = resolve_ref(ref_entry, affixes)
        if src is None:
            continue
        ann = load_annotations(src.dataset)
        images, is_virtual = scan_images(src, ann)
        if is_virtual:
            tasks = shard_tasks(ctx, src, ann)
        else:
            tasks = image_tasks(ctx, src, ann, images)
            print(f"   -> [{src.slug}] Discovered {len(images)} source tensors.")
        (nsfw_tasks if src.tag == "nsfw" else sfw_tasks).extend(tasks)

    nsfw_tasks = balance_nsfw(sfw_tasks, nsfw_tasks,
                              float(ctx.model_config.get("nsfw_ratio", 0)))
    return sfw_tasks + nsfw_tasks


def _apply_cpu_guard(args: argparse.Namespace, task_count: int) -> None:
    """Switch a huge CPU-only run to High-Speed Mode (no labeling, no vetting)."""
    if torch.cuda.is_available() or task_count <= _CPU_GUARD_TASKS:
        return
    if not args.no_labeling or not args.no_vetting:
        print(
            f"[CPU-GUARD] Massive dataset ({task_count} items) on CPU. Auto-enabling"
            " High-Speed Mode.",
            flush=True,
        )
        args.no_labeling = True
        args.no_vetting = True


def _pass_one_label(args: argparse.Namespace, task: str) -> str:
    """The PASS 1 progress-bar label."""
    if not args.no_vetting and task in ["quality", "classification"]:
        return "[PASS 1] Extraction & Vetting"
    return "[PASS 1] Extraction & Processing"


def _compile_samples(args: argparse.Namespace, settings: CompileSettings,
                     ctx: ManifoldContext) -> ManifoldOutcome:
    """Compile a sample-based manifold: plan, PASS 1, clustering, PASS 2, extras."""
    pascal_name, output_root = ctx.pascal_name, ctx.output_root
    db_path = output_root / "manifold_registry.db"
    conn = open_registry(db_path, pascal_name)
    ctx.existing.names.update(load_existing_names(conn, db_path, pascal_name))
    ctx.existing.on_disk.update(scan_physical_index(output_root, pascal_name))
    executor = make_executor(args, settings, ctx.existing.on_disk)
    try:
        adopt_orphans(conn, ctx.existing.names, ctx.existing.on_disk, ctx.task)

        affixes = (settings.prefix_str, settings.suffix_str)
        all_tasks = _plan_tasks(ctx, affixes)
        existing_count = len(ctx.existing.names)
        if not all_tasks:
            print(f"[NOTICE] No tasks found for {pascal_name}. Manifold is fully processed.")
            return ManifoldOutcome(ctx.model_key, output_root, samples=existing_count)

        print(f"[MANIFOLD] Found {len(all_tasks)} items needing processing (after disk-skip).")

        compiled_bytes = 0
        _apply_cpu_guard(args, len(all_tasks))
        if not args.finalize:
            run = ExtractionRun(conn, executor, settings.max_workers, settings.max_gb,
                                args.no_hash, batch_size=100 if args.no_vetting else 50)
            compiled_bytes = run_extraction(run, all_tasks, existing_count,
                                            _pass_one_label(args, ctx.task))

        conn.commit()

        compiled_gb = compiled_bytes / (1024**3)
        if compiled_gb < settings.min_gb:
            print(
                f"[WARNING] Compiled set size ({compiled_gb:.2f}GB) is below the minimum manifold"
                f" constraint ({settings.min_gb:.2f}GB)."
            )

        return finish_manifold(args, ctx, conn, affixes)
    finally:
        # Runs on success, error and Ctrl+C alike: no SQLite WAL or pool is left open.
        executor.shutdown(wait=True, cancel_futures=True)
        conn.close()


def finish_manifold(args: argparse.Namespace, ctx: ManifoldContext, conn: Any,
                    affixes: tuple[str, str]) -> ManifoldOutcome:
    """Cluster styles, write the index, docs, extra containers and notebooks.

    The manifold's task id goes into dataset_info.yaml as ``task_type``.
    """
    pascal_name, output_root = ctx.pascal_name, ctx.output_root
    cluster_styles(conn)
    final_index = write_index(conn, output_root, affixes)
    remove_empty_dirs(output_root)
    generate_dataset_docs(output_root, final_index, pascal_name,
                          overrides={"task_type": ctx.task})
    containers_written = write_containers(args, output_root)
    write_manifold_notebooks(ctx.model_key, pascal_name, output_root)

    print(
        f"[SUCCESS] v5.0 Ascension Complete: {len(final_index)} samples compiled for"
        f" {pascal_name}."
    )
    return ManifoldOutcome(ctx.model_key, output_root, samples=len(final_index),
                           containers_written=containers_written)


def _compile_manifold(args: argparse.Namespace, settings: CompileSettings,
                      model_key: str, model_config: dict[str, Any]) -> ManifoldOutcome:
    """Compile one registry entry into its manifold folder."""
    task = detect_task(model_key)
    pascal_name = model_config.get("name", model_key.replace("_", ""))
    full_name = f"{settings.prefix_str}{pascal_name}{settings.suffix_str}"
    output_root = OUT_PARENT / full_name
    _create_layout(output_root, model_config, task)

    print(f"\n[SOTA v5.0] Commencing compilation for {pascal_name} -> {output_root.name}...")

    # ─── FOREX MANIFOLD ────────────────────────────────────────────────────
    if is_forex(model_config):
        return compile_forex_manifold(model_key, model_config, output_root, full_name, task)

    # ─── NON-FOREX MANIFOLDS ──────────────────────────────────────────────
    ctx = ManifoldContext(model_key, model_config, task, pascal_name, output_root,
                          ExistingSamples())
    return _compile_samples(args, settings, ctx)


def process_dataset(parsed_args: argparse.Namespace | None = None) -> list[ManifoldOutcome]:
    """Compile every requested manifold and report what each one produced.

    ``--cleanup`` is not handled here: the loose-image purge runs in
    ``finalize_compile`` only after a successful compile.
    """
    args = parsed_args if parsed_args is not None else parse_compile_args()
    install_interrupt_handler()
    settings = prepare_compile(args)

    outcomes: list[ManifoldOutcome] = []
    for model_key, model_config in DATASETS_META.items():
        if args.model and model_key != args.model:
            continue
        outcomes.append(_compile_manifold(args, settings, model_key, model_config))
    return outcomes


if __name__ == '__main__':
    compile_args = parse_compile_args()
    sys.exit(finalize_compile(process_dataset(compile_args), purge_loose=compile_args.cleanup))
