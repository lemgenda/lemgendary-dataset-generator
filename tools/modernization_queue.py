"""
LemGendary Dataset Modernization & Kaggle Publication Queue Engine.

Coordinates batch modernization of datasets from legacy/raw formats into
their canonical container formats (WebDataset tar shards, MDS, LitData, Parquet)
and automatically publishes new versions to Kaggle with synchronized metadata.

Public facade and CLI entry point (``python -m tools.modernization_queue``).
The implementation lives in the ``tools.modernize`` package:
  - tools.modernize.queue_items    ManifoldQueueItem, load_queue_manifest
  - tools.modernize.loose_convert  loose directory -> WebDataset / MDS
  - tools.modernize.mds_transcode  legacy MDS -> WebDataset
  - tools.modernize.kaggle_push    push_manifold_to_kaggle
  - tools.modernize.queue_runner   run_queue, planning, conversion dispatch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from tools.modernize.kaggle_push import push_manifold_to_kaggle
from tools.modernize.loose_convert import (
    LooseConvertOptions,
    convert_loose_directory_to_mds,
    convert_loose_directory_to_webdataset,
)
from tools.modernize.mds_transcode import (
    _add_tar_members,
    _detect_img_ext,
    _iter_mds_tar_members,
    _mds_sample_to_tar_members,
    transcode_mds_to_webdataset,
)
from tools.modernize.queue_items import (
    ManifoldQueueItem,
    _fast_has_mds,
    _fast_has_tar_shards,
    load_queue_manifest,
    queue_status_labels,
)
from tools.modernize.queue_options import (
    CleanupOptions,
    ConvertOptions,
    QueueArgs,
    QueueOptions,
)
from tools.modernize.queue_runner import (
    _convert_item,
    describe_planned_actions,
    run_queue,
    select_queue_items,
)

__all__ = [
    "CleanupOptions",
    "ConvertOptions",
    "LooseConvertOptions",
    "ManifoldQueueItem",
    "QueueArgs",
    "QueueOptions",
    "_add_tar_members",
    "_convert_item",
    "_detect_img_ext",
    "_fast_has_mds",
    "_fast_has_tar_shards",
    "_iter_mds_tar_members",
    "_mds_sample_to_tar_members",
    "build_queue_args",
    "convert_loose_directory_to_mds",
    "convert_loose_directory_to_webdataset",
    "describe_planned_actions",
    "load_queue_manifest",
    "main",
    "push_manifold_to_kaggle",
    "queue_status_labels",
    "run_queue",
    "select_queue_items",
    "transcode_mds_to_webdataset",
]

_ROOT = (
    Path(__file__).resolve().parent.parent
    if Path(__file__).parent.name == "tools"
    else Path(__file__).resolve().parent
)


def build_queue_args(datasets: list[str] | None = None, **options: Any) -> list[str]:
    """Build the argument list for :func:`main`, shared by the API route and the CLI.

    ``options`` are the fields of :class:`QueueArgs` (workers, mds_shard_mb,
    purge_source, push_kaggle, dry_run, force). ``purge_source`` False keeps
    every source (zip, legacy folder, loose files); True deletes them after a
    successful conversion.
    """
    args = QueueArgs(**options).to_argv()
    keys = [k.strip() for k in (datasets or []) if k and k.strip()]
    if keys:
        args.extend(["--datasets", ",".join(keys)])
    return args


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the modernization queue CLI."""
    parser = argparse.ArgumentParser(
        description="LemGendary Dataset Modernization and Kaggle Publication Queue"
    )
    parser.add_argument(
        "--list", action="store_true", help="Print modernization queue status table and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview plan without executing conversions or uploads",
    )
    parser.add_argument(
        "--priority",
        default="mirnet_exposure,upn_v2",
        help="Comma-separated list of dataset keys to prioritize",
    )
    parser.add_argument(
        "--datasets", default=None, help="Comma-separated list of specific dataset keys to process"
    )
    parser.add_argument(
        "--push-kaggle",
        action="store_true",
        help="Push datasets to Kaggle as new versions after conversion",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for Kaggle server-side extraction tracking",
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Worker threads for WebP transcoding"
    )
    parser.add_argument("--shard-size", type=int, default=5000, help="Samples per WebDataset shard")
    parser.add_argument(
        "--mds-shard-mb", type=int, default=512, help="MDS shard size in MB (default: 512)"
    )
    parser.add_argument(
        "--keep-zip", action="store_true", help="Keep source zip archives (do not delete)"
    )
    parser.add_argument(
        "--keep-legacy-dir",
        action="store_true",
        help="Keep legacy Large directories (do not delete)",
    )
    parser.add_argument(
        "--purge-loose",
        action="store_true",
        help="Delete loose images/targets directories after packaging",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-conversion even if containerized shards already exist",
    )
    return parser


def _split_keys(raw: str | None) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()] if raw else []


def main(argv: list[str] | None = None) -> int:
    """Parse the CLI arguments and list or run the modernization queue."""
    args = _build_parser().parse_args(argv)

    datasets_root = (_ROOT.parent / "LemGendaryDatasets").resolve()
    registry_yaml = _ROOT / "unified_data.yaml"

    if not datasets_root.exists():
        print(f"[ERROR] LemGendaryDatasets directory not found at {datasets_root}")
        return 1

    items = load_queue_manifest(datasets_root, registry_yaml)

    priority_keys = _split_keys(args.priority)
    filter_keys = _split_keys(args.datasets) or None

    if args.list:
        return run_queue(
            items=items,
            priority_keys=priority_keys,
            filter_keys=filter_keys,
            dry_run=True,
        )

    if args.dry_run:
        print("[DRY-RUN] Planning only: nothing will be converted, deleted, written or uploaded.")

    return run_queue(
        items=items,
        priority_keys=priority_keys,
        filter_keys=filter_keys,
        push_kaggle=args.push_kaggle,
        shard_size=args.shard_size,
        workers=args.workers,
        delete_zip=not args.keep_zip,
        delete_legacy_dir=not args.keep_legacy_dir,
        purge_loose=args.purge_loose,
        dry_run=args.dry_run,
        no_wait=args.no_wait,
        mds_shard_size_mb=args.mds_shard_mb,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
