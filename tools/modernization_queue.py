"""
LemGendary Dataset Modernization & Kaggle Publication Queue Engine.

Coordinates batch modernization of datasets from legacy/raw formats into
their canonical container formats (WebDataset tar shards, MDS, LitData, Parquet)
and automatically publishes new versions to Kaggle with synchronized metadata.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
from typing import Any

from PIL import Image
import yaml
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "tools" else Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util

from core.common_sync import (
    get_dataset_version_info,
    push_kaggle_dataset_metadata,
    setup_kaggle_auth,
    track_kaggle_dataset_status,
)

# Load stream_zip_to_container dynamically to bypass tools/__init__.py bulk imports
_stream_tool_path = _ROOT / "tools" / "stream_zip_to_container.py"
_stream_spec = importlib.util.spec_from_file_location("stream_zip_to_container", _stream_tool_path)
if _stream_spec and _stream_spec.loader:
    _stream_mod = importlib.util.module_from_spec(_stream_spec)
    _stream_spec.loader.exec_module(_stream_mod)
    _transcode_to_webp = getattr(_stream_mod, "_transcode_to_webp")
    stream_zip_to_webdataset = getattr(_stream_mod, "stream_zip_to_webdataset")
else:
    raise ImportError("Could not load stream_zip_to_container")

logger = logging.getLogger(__name__)
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class ManifoldQueueItem:
    """Represents a dataset manifold in the modernization queue."""

    key: str
    name: str
    canonical_format: str
    modernized_folder: str
    kaggle_ref: str
    title: str
    subtitle: str
    license_id: str
    keywords: list[str] = field(default_factory=list)
    provenance_sources: list[str] = field(default_factory=list)
    
    # Filesystem state
    target_dir: Path | None = None
    legacy_dir: Path | None = None
    zip_path: Path | None = None
    is_containerized: bool = False
    has_loose_files: bool = False
    container_bytes: int = 0
    file_count: int = 0
    remote_version: int = 0


def _fast_has_tar_shards(shards_dir: Path) -> tuple[bool, int]:
    """Quickly check if shards dir has .tar files and approximate size."""
    if not shards_dir.exists():
        return False, 0
    total_bytes = 0
    found = False
    try:
        # Check root of shards or immediate subdirectories (train, val, test)
        for entry in os.scandir(shards_dir):
            if entry.is_file() and entry.name.endswith(".tar"):
                found = True
                total_bytes += entry.stat().st_size
            elif entry.is_dir():
                for sub_entry in os.scandir(entry.path):
                    if sub_entry.is_file() and sub_entry.name.endswith(".tar"):
                        found = True
                        total_bytes += sub_entry.stat().st_size
    except OSError:
        pass
    return found, total_bytes


def load_queue_manifest(
    datasets_root: Path,
    registry_yaml: Path,
) -> list[ManifoldQueueItem]:
    """Scan unified_data.yaml and LemGendaryDatasets to build the queue state."""
    with open(registry_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    items: list[ManifoldQueueItem] = []
    datasets_dict = cfg.get("datasets", {})

    for key, ds in datasets_dict.items():
        name = ds.get("name", key)
        mod_folder_name = ds.get("modernized_folder", f"LemGendized{name}")
        can_fmt = ds.get("canonical_format", "webdataset")
        kref = ds.get("kaggle_ref", "")
        title = ds.get("title", f"LemGendized {name}")
        subtitle = ds.get("subtitle", "")
        lic = ds.get("license", "CC-BY-NC-4.0")
        kw = ds.get("keywords", [])
        sources = ds.get("provenance_sources", [])

        target_dir = datasets_root / mod_folder_name
        legacy_dir = datasets_root / f"{mod_folder_name}Large"

        zip_candidates = [
            datasets_root / f"{mod_folder_name.lower()}large.zip",
            datasets_root / f"{mod_folder_name.lower()}.zip",
            datasets_root / f"{name.lower()}large.zip",
            datasets_root / f"{name.lower()}.zip",
        ]
        found_zip: Path | None = None
        for z in zip_candidates:
            if z.exists() and z.stat().st_size > 0:
                found_zip = z
                break

        has_shards, shard_bytes = _fast_has_tar_shards(target_dir / "shards") if target_dir.exists() else (False, 0)
        has_parquet = any(target_dir.glob("*.parquet")) if target_dir.exists() else False
        has_mds = (target_dir / "mds").exists() if target_dir.exists() else False
        has_litdata = any(target_dir.glob("chunk-*.bin")) if target_dir.exists() else False

        is_containerized = False
        container_bytes = 0

        if can_fmt == "webdataset" and has_shards:
            is_containerized = True
            container_bytes = shard_bytes
        elif can_fmt == "parquet" and has_parquet:
            is_containerized = True
            container_bytes = sum(f.stat().st_size for f in target_dir.glob("*.parquet"))
        elif can_fmt == "mds" and has_mds:
            is_containerized = True
        elif can_fmt == "litdata" and has_litdata:
            is_containerized = True
        elif can_fmt == "directory" and target_dir.exists():
            is_containerized = True

        has_loose = False
        if target_dir.exists():
            for sub in ("images", "targets", "masks"):
                sub_p = target_dir / sub
                if sub_p.exists():
                    has_loose = True
                    break

        item = ManifoldQueueItem(
            key=key,
            name=name,
            canonical_format=can_fmt,
            modernized_folder=mod_folder_name,
            kaggle_ref=kref,
            title=title,
            subtitle=subtitle,
            license_id=lic,
            keywords=kw,
            provenance_sources=sources,
            target_dir=target_dir,
            legacy_dir=legacy_dir if legacy_dir.exists() else None,
            zip_path=found_zip,
            is_containerized=is_containerized,
            has_loose_files=has_loose,
            container_bytes=container_bytes,
        )
        items.append(item)

    return items


def convert_loose_directory_to_webdataset(
    source_dir: Path,
    target_dir: Path,
    shard_size_samples: int = 5000,
    transcode_webp: bool = True,
    max_workers: int | None = None,
    purge_loose: bool = False,
) -> bool:
    """Convert a directory containing loose images/targets into WebDataset shards."""
    target_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = target_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    workers = max_workers or min(12, os.cpu_count() or 4)

    images_dir = source_dir / "images"
    targets_dir = source_dir / "targets"
    splits = ["train", "val", "test"]

    has_splits = False
    for s in splits:
        if (images_dir / s).exists() or (targets_dir / s).exists():
            has_splits = True
            break

    target_splits = splits if has_splits else ["all"]
    print(f"[CONVERT] Packaging loose directory {source_dir.name} into WebDataset shards ({workers} workers)...")

    for split in target_splits:
        split_img_dir = (images_dir / split) if has_splits else images_dir
        split_tgt_dir = (targets_dir / split) if has_splits else targets_dir
        split_out_dir = shards_dir if split == "all" else shards_dir / split
        split_out_dir.mkdir(parents=True, exist_ok=True)

        img_files: list[Path] = []
        if split_img_dir.exists():
            img_files = [p for p in split_img_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]

        tgt_files: dict[str, Path] = {}
        if split_tgt_dir.exists():
            for p in split_tgt_dir.iterdir():
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                    tgt_files[p.stem] = p

        total_samples = len(img_files) if img_files else len(tgt_files)
        if total_samples == 0:
            continue

        total_shards = (total_samples + shard_size_samples - 1) // shard_size_samples
        print(f"[CONVERT] Split '{split}': {total_samples} samples -> {total_shards} shards.")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            if img_files:
                with tqdm(total=total_samples, desc=f"SHARDS [{split}]", unit="img", dynamic_ncols=True) as pbar:
                    for s_idx in range(total_shards):
                        start_i = s_idx * shard_size_samples
                        end_i = min(start_i + shard_size_samples, total_samples)
                        chunk = img_files[start_i:end_i]
                        shard_path = split_out_dir / f"shard-{s_idx:05d}.tar"

                        def _read_and_transcode(p: Path) -> tuple[str, bytes, str]:
                            raw = p.read_bytes()
                            if transcode_webp:
                                out_b, out_ext = _transcode_to_webp(raw, p.suffix, quality=92)
                                return p.stem, out_b, out_ext
                            return p.stem, raw, p.suffix.lower()

                        encoded_imgs = list(pool.map(_read_and_transcode, chunk))

                        with tarfile.open(shard_path, "w") as tf:
                            for stem, data, ext in encoded_imgs:
                                ti = tarfile.TarInfo(name=f"{stem}{ext}")
                                ti.size = len(data)
                                tf.addfile(ti, io.BytesIO(data))

                                if stem in tgt_files:
                                    tgt_p = tgt_files[stem]
                                    tgt_raw = tgt_p.read_bytes()
                                    if transcode_webp:
                                        t_bytes, t_ext = _transcode_to_webp(tgt_raw, tgt_p.suffix, quality=95)
                                    else:
                                        t_bytes, t_ext = tgt_raw, tgt_p.suffix.lower()
                                    t_ti = tarfile.TarInfo(name=f"{stem}.target{t_ext}")
                                    t_ti.size = len(t_bytes)
                                    tf.addfile(t_ti, io.BytesIO(t_bytes))

                        pbar.update(len(chunk))
            else:
                # Target-only
                tgt_list = list(tgt_files.values())
                with tqdm(total=total_samples, desc=f"TARGET SHARDS [{split}]", unit="img", dynamic_ncols=True) as pbar:
                    for s_idx in range(total_shards):
                        start_i = s_idx * shard_size_samples
                        end_i = min(start_i + shard_size_samples, total_samples)
                        chunk = tgt_list[start_i:end_i]
                        shard_path = split_out_dir / f"shard-{s_idx:05d}.tar"

                        def _read_and_transcode_tgt(p: Path) -> tuple[str, bytes, str]:
                            raw = p.read_bytes()
                            if transcode_webp:
                                out_b, out_ext = _transcode_to_webp(raw, p.suffix, quality=95)
                                return p.stem, out_b, out_ext
                            return p.stem, raw, p.suffix.lower()

                        encoded_tgts = list(pool.map(_read_and_transcode_tgt, chunk))

                        with tarfile.open(shard_path, "w") as tf:
                            for stem, data, ext in encoded_tgts:
                                ti = tarfile.TarInfo(name=f"{stem}{ext}")
                                ti.size = len(data)
                                tf.addfile(ti, io.BytesIO(data))

                        pbar.update(len(chunk))

    # Update dataset_info.yaml
    info_path = target_dir / "dataset_info.yaml"
    if info_path.exists():
        try:
            with open(info_path, "r", encoding="utf-8") as f_in:
                idata = yaml.safe_load(f_in) or {}
            idata["format"] = "webdataset"
            idata["canonical_format"] = "webdataset"
            if transcode_webp:
                idata["image_format"] = "webp"
            with open(info_path, "w", encoding="utf-8") as f_out:
                yaml.safe_dump(idata, f_out, default_flow_style=False, sort_keys=False, allow_unicode=True)
            print(f"[METADATA] Updated {info_path.name} format to webdataset.")
        except Exception as exc:
            print(f"[WARN] Failed updating dataset_info.yaml: {exc}")

    if purge_loose:
        print("[CLEANUP] Purging loose image directories to reclaim storage...")
        for sub in ("images", "targets", "masks"):
            sub_p = source_dir / sub
            if sub_p.exists():
                shutil.rmtree(sub_p, ignore_errors=True)
        print("[SUCCESS] Loose directories purged.")

    return True


def push_manifold_to_kaggle(
    item: ManifoldQueueItem,
    version_notes: str | None = None,
    no_wait: bool = False,
) -> bool:
    """Upload containerized manifold to Kaggle as a new version with synchronized metadata."""
    if not item.kaggle_ref:
        print(f"[SKIP] No kaggle_ref configured for {item.name}.")
        return False

    clean_handle = item.kaggle_ref.replace("kaggle://", "").strip()
    if "/" not in clean_handle:
        clean_handle = f"lemtreursi/{clean_handle}"

    target_dir = item.target_dir
    if not target_dir or not target_dir.exists():
        print(f"[ERROR] Target directory {target_dir} does not exist.")
        return False

    setup_kaggle_auth()
    v_info = get_dataset_version_info(clean_handle)
    current_ver = v_info.get("latest_version") or 0
    next_ver = current_ver + 1
    notes = version_notes or f"v{next_ver}: Modernized {item.canonical_format} format with WebP transcoding"

    print(f"\n[KAGGLE-PUSH] Uploading {item.name} to {clean_handle} (target version: v{next_ver})...")
    print(f"[KAGGLE-PUSH] Payload directory: {target_dir}")

    # Synchronize dataset-metadata.json id field with clean_handle
    meta_path = target_dir / "dataset-metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f_meta:
                m_data = json.load(f_meta)
            m_data["id"] = clean_handle
            m_data["title"] = item.title
            m_data["subtitle"] = item.subtitle
            with open(meta_path, "w", encoding="utf-8") as f_meta:
                json.dump(m_data, f_meta, indent=2)
            print(f"[METADATA] Synchronized {meta_path.name} handle to {clean_handle}")
        except Exception as exc:
            print(f"[WARN] Could not update dataset-metadata.json id: {exc}")

    import kagglehub
    import kagglehub.gcs_upload
    kagglehub.gcs_upload.MAX_FILES_TO_UPLOAD = 5000

    try:
        kagglehub.dataset_upload(clean_handle, str(target_dir), version_notes=notes)
        print(f"[SUCCESS] Upload payload transferred to Kaggle for {clean_handle}.")
    except Exception as exc:
        print(f"[ERROR] Kaggle upload failed for {clean_handle}: {exc}")
        return False

    # Push metadata & column descriptors
    if meta_path.exists():
        try:
            print(f"[METADATA] Pushing dataset settings and licensing to Kaggle API...")
            push_kaggle_dataset_metadata(clean_handle, meta_path)
            print(f"[SUCCESS] Kaggle metadata successfully synchronized for {clean_handle}.")
        except Exception as exc:
            print(f"[WARN] Kaggle metadata push notice: {exc}")

    if not no_wait:
        print(f"[TRACK] Waiting for Kaggle processing completion for v{next_ver}...")
        ok = track_kaggle_dataset_status(clean_handle, target_version=next_ver, timeout=3600)
        return ok

    return True


def run_queue(
    items: list[ManifoldQueueItem],
    priority_keys: list[str] | None = None,
    filter_keys: list[str] | None = None,
    push_kaggle: bool = False,
    shard_size: int = 5000,
    workers: int | None = None,
    delete_zip: bool = True,
    delete_legacy_dir: bool = True,
    purge_loose: bool = False,
    dry_run: bool = False,
    no_wait: bool = False,
) -> int:
    """Execute modernization queue in prioritized order."""
    item_map = {it.key: it for it in items}

    # Filter if specified
    if filter_keys:
        selected_items: list[ManifoldQueueItem] = []
        for k in filter_keys:
            norm_k = k.lower().replace("-", "_")
            matched = [it for it in items if it.key.lower() == norm_k or it.name.lower() == norm_k or norm_k in it.key.lower()]
            selected_items.extend(matched)
        items = list({it.key: it for it in selected_items}.values())

    # Sort according to priority
    def sort_key(it: ManifoldQueueItem) -> tuple[int, int, str]:
        # Rank 0: specified in priority_keys
        if priority_keys:
            for p_idx, pk in enumerate(priority_keys):
                norm_pk = pk.lower().replace("-", "_")
                if it.key.lower() == norm_pk or it.name.lower() == norm_pk or norm_pk in it.key.lower():
                    return (0, p_idx, it.key)
        # Rank 1: already containerized (e.g. ready for Kaggle push)
        if it.is_containerized:
            return (1, 0, it.key)
        # Rank 2: has zip archive ready for streaming
        if it.zip_path:
            return (2, 0, it.key)
        # Rank 3: has loose files
        if it.has_loose_files:
            return (3, 0, it.key)
        # Rank 4: legacy dir only
        return (4, 0, it.key)

    queue = sorted(items, key=sort_key)

    print("\n" + "=" * 90)
    print("LemGendary Dataset Modernization & Publication Queue")
    print("=" * 90)
    header = f"{'#':<3} | {'Key':<26} | {'Format':<10} | {'Status':<18} | {'Source / Storage':<25}"
    print(header)
    print("-" * 90)

    for idx, it in enumerate(queue, 1):
        if it.is_containerized:
            size_gb = it.container_bytes / (1024**3)
            status_str = f"CONTAINERIZED ({size_gb:.1f}GB)"
            src_str = "Shards ready"
        elif it.zip_path:
            zip_gb = it.zip_path.stat().st_size / (1024**3)
            status_str = "PENDING STREAM"
            src_str = f"ZIP ({zip_gb:.1f}GB)"
        elif it.has_loose_files:
            status_str = "PENDING PACK"
            src_str = "Loose WebP/Images"
        elif it.legacy_dir:
            status_str = "PENDING LEGACY"
            src_str = f"{it.legacy_dir.name}/"
        else:
            status_str = "MISSING SOURCE"
            src_str = "None"

        print(f"{idx:<3} | {it.key:<26} | {it.canonical_format:<10} | {status_str:<18} | {src_str:<25}")

    print("=" * 90 + "\n")

    if dry_run:
        print("[DRY-RUN] Execution simulation complete. Zero changes written.")
        return 0

    success_count = 0
    failure_count = 0

    for idx, it in enumerate(queue, 1):
        print(f"\n[{idx}/{len(queue)}] Processing Manifold: {it.name} ({it.key})")
        print("-" * 60)

        # Step 1: Container Conversion (if not already containerized)
        if not it.is_containerized:
            if it.zip_path:
                print(f"[ACTION] Streaming legacy zip {it.zip_path.name} directly into WebDataset shards...")
                ok = stream_zip_to_webdataset(
                    zip_path=it.zip_path,
                    target_dir=it.target_dir,
                    shard_size_samples=shard_size,
                    delete_zip=delete_zip,
                    legacy_dir=it.legacy_dir if delete_legacy_dir else None,
                    transcode_webp=True,
                    max_workers=workers,
                )
                if not ok:
                    print(f"[ERROR] Failed streaming {it.key}. Skipping to next manifold.")
                    failure_count += 1
                    continue
                it.is_containerized = True
            elif it.has_loose_files and it.canonical_format == "webdataset":
                print(f"[ACTION] Packing loose directory into WebDataset shards...")
                ok = convert_loose_directory_to_webdataset(
                    source_dir=it.target_dir,
                    target_dir=it.target_dir,
                    shard_size_samples=shard_size,
                    transcode_webp=True,
                    max_workers=workers,
                    purge_loose=purge_loose,
                )
                if not ok:
                    print(f"[ERROR] Failed packing {it.key}. Skipping to next manifold.")
                    failure_count += 1
                    continue
                it.is_containerized = True
            elif it.legacy_dir and (it.legacy_dir / "targets").exists():
                print(f"[ACTION] Packing legacy directory {it.legacy_dir.name} into WebDataset shards...")
                ok = convert_loose_directory_to_webdataset(
                    source_dir=it.legacy_dir,
                    target_dir=it.target_dir,
                    shard_size_samples=shard_size,
                    transcode_webp=True,
                    max_workers=workers,
                    purge_loose=False,
                )
                if not ok:
                    print(f"[ERROR] Failed packing legacy {it.key}. Skipping to next manifold.")
                    failure_count += 1
                    continue
                if delete_legacy_dir:
                    print(f"[CLEANUP] Purging legacy directory {it.legacy_dir.name}...")
                    shutil.rmtree(it.legacy_dir, ignore_errors=True)
                it.is_containerized = True
            else:
                print(f"[NOTICE] No suitable conversion method found for {it.key}. Skipping.")
                continue

        # Step 2: Push to Kaggle if requested
        if push_kaggle:
            print(f"[ACTION] Publishing new version of {it.name} to Kaggle...")
            push_ok = push_manifold_to_kaggle(it, no_wait=no_wait)
            if not push_ok:
                print(f"[ERROR] Kaggle publication failed for {it.name}.")
                failure_count += 1
                continue
            print(f"[SUCCESS] {it.name} successfully published to Kaggle.")

        success_count += 1

    print("\n" + "=" * 90)
    print(f"Modernization Queue Run Complete: {success_count} succeeded, {failure_count} failed.")
    print("=" * 90 + "\n")

    return 0 if failure_count == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="LemGendary Dataset Modernization and Kaggle Publication Queue")
    parser.add_argument("--list", action="store_true", help="Print modernization queue status table and exit")
    parser.add_argument("--dry-run", action="store_true", help="Preview plan without executing conversions or uploads")
    parser.add_argument("--priority", default="mirnet_exposure,upn_v2", help="Comma-separated list of dataset keys to prioritize")
    parser.add_argument("--datasets", default=None, help="Comma-separated list of specific dataset keys to process")
    parser.add_argument("--push-kaggle", action="store_true", help="Push datasets to Kaggle as new versions after conversion")
    parser.add_argument("--no-wait", action="store_true", help="Do not wait for Kaggle server-side extraction tracking")
    parser.add_argument("--workers", type=int, default=None, help="Worker threads for WebP transcoding")
    parser.add_argument("--shard-size", type=int, default=5000, help="Samples per shard")
    parser.add_argument("--keep-zip", action="store_true", help="Keep source zip archives (do not delete)")
    parser.add_argument("--keep-legacy-dir", action="store_true", help="Keep legacy Large directories (do not delete)")
    parser.add_argument("--purge-loose", action="store_true", help="Delete loose images/targets directories after packaging")
    args = parser.parse_args()

    datasets_root = (_ROOT.parent / "LemGendaryDatasets").resolve()
    registry_yaml = _ROOT / "unified_data.yaml"

    if not datasets_root.exists():
        print(f"[ERROR] LemGendaryDatasets directory not found at {datasets_root}")
        return 1

    items = load_queue_manifest(datasets_root, registry_yaml)

    priority_keys = [k.strip() for k in args.priority.split(",") if k.strip()] if args.priority else []
    filter_keys = [k.strip() for k in args.datasets.split(",") if k.strip()] if args.datasets else None

    if args.list:
        return run_queue(
            items=items,
            priority_keys=priority_keys,
            filter_keys=filter_keys,
            dry_run=True,
        )

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
    )


if __name__ == "__main__":
    sys.exit(main())
