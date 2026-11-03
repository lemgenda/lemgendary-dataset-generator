"""Modernization queue: ordering, dry-run planning, conversion dispatch and the run loop."""

from __future__ import annotations

import shutil
from typing import Any

from tools.modernize.kaggle_push import push_manifold_to_kaggle
from tools.modernize.loose_convert import (
    convert_loose_directory_to_mds,
    convert_loose_directory_to_webdataset,
)
from tools.modernize.mds_transcode import transcode_mds_to_webdataset
from tools.modernize.queue_items import ManifoldQueueItem, queue_status_labels
from tools.modernize.queue_options import (
    CleanupOptions,
    ConvertOptions,
    QueueOptions,
    resolve_queue_options,
)
from tools.stream_zip_to_container import stream_zip_to_mds, stream_zip_to_webdataset

_RULE = "=" * 90


# ── Conversion dispatch ─────────────────────────────────────────────────────
def _convert_from_zip(it: ManifoldQueueItem, fmt: str, options: ConvertOptions) -> bool:
    """Stream the item's legacy zip archive into MDS or WebDataset shards."""
    assert it.zip_path is not None, f"zip_path must not be None for zip streaming of {it.key}"
    cleanup = options.cleanup
    legacy_dir = it.legacy_dir if cleanup.delete_legacy_dir else None
    if fmt == "mds":
        print(f"[ACTION] Streaming legacy zip {it.zip_path.name} directly into MDS shards...")
        return stream_zip_to_mds(
            zip_path=it.zip_path,
            target_dir=it.target_dir,
            mds_shard_size_bytes=options.mds_shard_size_bytes,
            delete_zip=cleanup.delete_zip,
            legacy_dir=legacy_dir,
            transcode_webp=True,
            max_workers=options.workers,
        )
    # Default: webdataset (and fallback for other formats via zip)
    if fmt not in ("webdataset",):
        print(
            f"[NOTICE] Format '{fmt}' does not have a direct zip streaming path. Falling back"
            " to WebDataset streaming."
        )
    print(f"[ACTION] Streaming legacy zip {it.zip_path.name} into WebDataset shards...")
    return stream_zip_to_webdataset(
        zip_path=it.zip_path,
        target_dir=it.target_dir,
        shard_size_samples=options.shard_size,
        delete_zip=cleanup.delete_zip,
        legacy_dir=legacy_dir,
        transcode_webp=True,
        max_workers=options.workers,
    )


def _convert_from_loose(it: ManifoldQueueItem, fmt: str, options: ConvertOptions) -> bool:
    """Pack the item's loose files (or legacy folder) into MDS or WebDataset shards."""
    src = it.target_dir if it.has_loose_files else it.legacy_dir
    if src is None or not src.exists():
        print(f"[NOTICE] No suitable source found for {it.key}. Skipping.")
        return False

    container = "MDS" if fmt == "mds" else "WebDataset"
    if fmt not in ("mds", "webdataset", "directory"):
        print(
            f"[NOTICE] Format '{fmt}' loose-to-container not yet implemented in queue. Packing as"
            " WebDataset."
        )
    # it.target_dir is always set by load_queue_manifest; assert to narrow the type.
    assert it.target_dir is not None, (
        f"target_dir must not be None for loose {container} pack of {it.key}"
    )
    print(f"[ACTION] Packing loose directory {src.name} into {container} shards...")
    flags: dict[str, Any] = {
        "transcode_webp": True,
        "max_workers": options.workers,
        "purge_loose": options.cleanup.purge_loose if it.has_loose_files else False,
    }
    if fmt == "mds":
        ok = convert_loose_directory_to_mds(
            src, it.target_dir, mds_shard_size_bytes=options.mds_shard_size_bytes, **flags
        )
    else:
        ok = convert_loose_directory_to_webdataset(
            src, it.target_dir, shard_size_samples=options.shard_size, **flags
        )
    if ok and not it.has_loose_files and options.cleanup.delete_legacy_dir and it.legacy_dir:
        print(f"[CLEANUP] Purging legacy directory {it.legacy_dir.name}...")
        shutil.rmtree(it.legacy_dir, ignore_errors=True)
    return ok


def _convert_item(it: ManifoldQueueItem, options: ConvertOptions) -> bool:
    """Dispatch a single manifold item to the correct conversion function."""
    fmt = it.canonical_format.lower()
    if it.zip_path:
        return _convert_from_zip(it, fmt, options)

    if it.has_mds and fmt in ("webdataset", "directory"):
        assert it.target_dir is not None, (
            f"target_dir must not be None for MDS transcode of {it.key}"
        )
        print(
            f"[ACTION] Transcoding legacy MDS shards in {it.target_dir.name} to WebDataset (.data)"
            " shards..."
        )
        return transcode_mds_to_webdataset(
            target_dir=it.target_dir,
            shard_size_samples=options.shard_size,
            shard_ext="data",
            purge_mds=True,
        )
    return _convert_from_loose(it, fmt, options)


# ── Selection and ordering ──────────────────────────────────────────────────
def select_queue_items(
    base_queue: list[ManifoldQueueItem], filter_keys: list[str]
) -> list[ManifoldQueueItem]:
    """Select the queue items named by ``filter_keys``, preserving queue order.

    Each key may be a 1-based queue index, an exact dataset key / name /
    modernized folder, or (only when nothing matches exactly) a key substring.
    Exact matches win so that a scoped run never spills onto sibling datasets
    whose keys merely contain the requested key.
    """
    selected: list[ManifoldQueueItem] = []
    for k in filter_keys:
        if k.isdigit():
            idx = int(k)
            if 1 <= idx <= len(base_queue):
                selected.append(base_queue[idx - 1])
            else:
                print(f"[WARN] Index #{idx} out of range (1..{len(base_queue)}).")
            continue
        norm_k = k.lower().replace("-", "_")
        exact = [
            it for it in base_queue
            if norm_k in (it.key.lower(), it.name.lower(), it.modernized_folder.lower())
        ]
        if exact:
            selected.extend(exact)
            continue
        partial = [it for it in base_queue if norm_k in it.key.lower()]
        if not partial:
            print(f"[WARN] No manifold matches '{k}'.")
        selected.extend(partial)
    return [it for it in base_queue if it in selected]


def _matches_priority(it: ManifoldQueueItem, priority_key: str) -> bool:
    norm_pk = priority_key.lower().replace("-", "_")
    return norm_pk in (it.key.lower(), it.name.lower()) or norm_pk in it.key.lower()


def _queue_sort_key(
    it: ManifoldQueueItem, priority_keys: list[str] | None
) -> tuple[int, int, str]:
    """Order: priority keys first, then containerized, zip, loose, MDS, the rest."""
    for p_idx, pk in enumerate(priority_keys or []):
        if _matches_priority(it, pk):
            return (0, p_idx, it.key)
    state_ranks = (
        (it.is_containerized, (1, 0)),
        (it.zip_path, (2, 0)),
        (it.has_loose_files, (3, 0)),
        (it.has_mds, (3, 1)),
    )
    for present, rank in state_ranks:
        if present:
            return (*rank, it.key)
    return (4, 0, it.key)


# ── Planning (dry run) ──────────────────────────────────────────────────────
def _conversion_actions(
    it: ManifoldQueueItem, prefix: str, cleanup: CleanupOptions
) -> list[str]:
    """Describe the conversion of an item that has a source to convert from."""
    if it.zip_path:
        actions = [f"{prefix} from zip {it.zip_path.name} to {it.canonical_format}"]
        if cleanup.delete_zip:
            actions.append(f"would delete zip {it.zip_path.name}")
        if cleanup.delete_legacy_dir and it.legacy_dir:
            actions.append(f"would delete legacy folder {it.legacy_dir.name}")
        return actions
    if it.has_mds and it.canonical_format.lower() in ("webdataset", "directory"):
        return [f"{prefix} legacy MDS shards to WebDataset", "would delete legacy MDS folder"]
    if it.has_loose_files:
        actions = [f"{prefix} loose files to {it.canonical_format}"]
        if cleanup.purge_loose:
            actions.append("would delete loose image folders")
        return actions
    if it.legacy_dir:
        actions = [f"{prefix} legacy folder {it.legacy_dir.name} to {it.canonical_format}"]
        if cleanup.delete_legacy_dir:
            actions.append(f"would delete legacy folder {it.legacy_dir.name}")
        return actions
    return ["no source found; would skip"]


def describe_planned_actions(
    it: ManifoldQueueItem, options: QueueOptions | None = None, **kwargs: Any
) -> list[str]:
    """Describe, without touching the filesystem, what a real run would do to ``it``.

    ``options`` or the flat keyword arguments of :func:`run_queue` (push_kaggle,
    delete_zip, delete_legacy_dir, purge_loose, force) select the run flags.
    """
    opts = resolve_queue_options(options, kwargs)
    if it.is_containerized and not opts.force:
        actions = ["already containerized; no conversion"]
    elif it.is_containerized and not (it.zip_path or it.has_loose_files or it.legacy_dir):
        actions = ["force requested but no source remains; existing shards kept"]
    else:
        prefix = "force re-convert" if it.is_containerized else "convert"
        actions = _conversion_actions(it, prefix, opts.cleanup)
    if opts.push_kaggle:
        actions.append("would push a new Kaggle version")
    return actions


# ── Run ─────────────────────────────────────────────────────────────────────
def _print_queue_table(queue: list[ManifoldQueueItem]) -> None:
    print("\n" + _RULE)
    print("LemGendary Dataset Modernization & Publication Queue")
    print(_RULE)
    print(f"{'#':<3} | {'Key':<26} | {'Format':<10} | {'Status':<18} | {'Source / Storage':<25}")
    print("-" * 90)
    for idx, it in enumerate(queue, 1):
        status_str, src_str = queue_status_labels(it)
        print(
            f"{idx:<3} | {it.key:<26} | {it.canonical_format:<10} | {status_str:<18} |"
            f" {src_str:<25}"
        )
    print(_RULE + "\n")


def _print_dry_run(queue: list[ManifoldQueueItem], opts: QueueOptions) -> None:
    for idx, it in enumerate(queue, 1):
        for line in describe_planned_actions(it, opts):
            print(f"[DRY-RUN] #{idx} {it.key}: {line}")
    print("[DRY-RUN] Execution simulation complete. Zero changes written.")


# Outcomes of the conversion step of one queue item.
_STEP_FAILED = "failed"
_STEP_DONE = "done"  # nothing left to do for this item; counts as a success
_STEP_CONTINUE = "continue"


def _conversion_step(it: ManifoldQueueItem, opts: QueueOptions) -> str:
    """Convert ``it`` unless already containerized (or force); return the step outcome."""
    if it.is_containerized and not opts.force:
        size_gb = it.container_bytes / (1024**3)
        print(
            f"[READY] Manifold {it.name} ({it.key}) is already containerized ({size_gb:.1f}GB"
            f" {it.canonical_format}). Shards are verified and ready."
        )
        return _STEP_CONTINUE
    if it.is_containerized:
        print(f"[FORCE] Re-conversion requested for already containerized manifold {it.name}.")
        if not it.zip_path and not it.has_loose_files and not it.legacy_dir:
            print(
                "[NOTICE] No source archive or loose files found (purged"
                " post-modernization). Preserving existing containerized shards."
            )
            return _STEP_DONE
    if not _convert_item(it=it, options=opts.convert):
        print(f"[ERROR] Failed converting {it.key}. Skipping to next manifold.")
        return _STEP_FAILED
    it.is_containerized = True
    return _STEP_CONTINUE


def _process_item(it: ManifoldQueueItem, opts: QueueOptions) -> bool:
    """Convert and optionally publish one queue item; return True on success."""
    step = _conversion_step(it, opts)
    if step != _STEP_CONTINUE:
        return step == _STEP_DONE
    if opts.push_kaggle:
        print(f"[ACTION] Publishing new version of {it.name} to Kaggle...")
        if not push_manifold_to_kaggle(it, no_wait=opts.no_wait):
            print(f"[ERROR] Kaggle publication failed for {it.name}.")
            return False
        print(f"[SUCCESS] {it.name} successfully published to Kaggle.")
    return True


def run_queue(
    items: list[ManifoldQueueItem], options: QueueOptions | None = None, **kwargs: Any
) -> int:
    """Execute modernization queue in prioritized order.

    Configure the run with ``options`` or with the flat keyword arguments
    priority_keys, filter_keys, push_kaggle, shard_size, workers, delete_zip,
    delete_legacy_dir, purge_loose, dry_run, no_wait, mds_shard_size_mb, force.
    """
    opts = resolve_queue_options(options, kwargs)
    # Sort according to priority to establish canonical queue numbering
    queue = sorted(items, key=lambda it: _queue_sort_key(it, opts.priority_keys))
    if opts.filter_keys:
        queue = select_queue_items(queue, opts.filter_keys)
        if not queue:
            print(
                "[ERROR] No manifolds matched the requested dataset(s):"
                f" {', '.join(opts.filter_keys)}. Nothing to do."
            )
            return 1

    _print_queue_table(queue)
    if opts.dry_run:
        _print_dry_run(queue, opts)
        return 0

    success_count = 0
    failure_count = 0
    for idx, it in enumerate(queue, 1):
        print(
            f"\n[{idx}/{len(queue)}] Processing Manifold: {it.name} ({it.key}) [Format:"
            f" {it.canonical_format}]"
        )
        print("-" * 60)
        if _process_item(it, opts):
            success_count += 1
        else:
            failure_count += 1

    print("\n" + _RULE)
    print(f"Modernization Queue Run Complete: {success_count} succeeded, {failure_count} failed.")
    print(_RULE + "\n")
    return 0 if failure_count == 0 else 1
