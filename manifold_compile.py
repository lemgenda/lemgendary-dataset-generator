import os
import sys
import argparse
import json
import yaml
import shutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from datetime import datetime
import hashlib
import random
from sklearn.cluster import MiniBatchKMeans
import webdataset as wds
import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm

PHYSICAL_INDEX = set()

from compiler_core import (
    CONFIG,
    DATASETS_META,
    INPUT_ROOT,
    META,
    OUT_PARENT,
    batch_worker,
    clean_slug,
    detect_annotations,
    detect_task,
    download_image,
    generate_colab_training_notebook,
    generate_training_notebook,
    get_device_info,
    init_worker,
    initialize_registry,
    parse_coco,
    parse_matlab,
    parse_parquet,
    process_diffusion,
    process_image,
    process_parquet_shard,
    remove_empty_dirs,
)
from doc_generator import generate_dataset_docs


# 2026: Task tuples passed to batch_worker. Each item is
#     (callable, *heterogeneous_args)
# Annotated as tuple[Any, ...] so downstream consumers accept the mix.
TaskItem = tuple[Any, ...]

# 2026: Type aliases for the three parser return shapes, used at cast sites.
CocoAnnData = tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]
ParquetAnnData = tuple[Path, dict[str, Any], list[str]]
MatlabAnnData = tuple[dict[str, Any], str]


def _fast_scan(path, valid_exts):
    for entry in os.scandir(path):
        if entry.is_dir():
            yield from _fast_scan(entry.path, valid_exts)
        elif entry.is_file():
            ext = entry.name[entry.name.rfind('.'):].lower()
            if ext in valid_exts:
                yield entry.path


def _require_ann_path(ann_path: Path | None) -> Path:
    """Narrow ann_path to non-None.

    detect_annotations()'s contract is that a non-None format is always
    paired with a non-None path. This helper converts that invariant into
    a runtime check at the type-checker boundary, so parse_* callers can
    accept a plain Path without any suppression.
    """
    if ann_path is None:
        raise RuntimeError(
            "Annotation format was detected but no path was returned. "
            "This indicates a bug in converters.detect_annotations()."
        )
    return ann_path

# ─── ARGUMENT PARSER ─────────────────────────────────────────────────────────
# 2026 Phase 1.5: Parser is the SSOT from cli_args.py. This module previously
# declared its own parser that had to stay manually in sync with
# compiler_core.py's. Both now share `build_parser()`.
from cli_args import build_parser

parser = build_parser()
args = parser.parse_args()
# ────────────────────────────────────────────────────────────────────────────

def process_dataset():
    # 2026 Resilience: Force-Kill Handler for Windows (SIGINT v1.1)
    if os.name == 'nt':
        import signal
        def signal_handler(sig, frame):
            print("\n[INTERRUPT] Emergency termination requested. Mission aborted.")
            os._exit(1)
        signal.signal(signal.SIGINT, signal_handler)

    if args.cleanup:
        print("[JANITOR] Cleanup requested. Purging temporary files...")
        # Add cleanup logic here if needed in the future
        print("[JANITOR] Cleanup complete.")
        return

    min_gb = META.get("global_constraints", {}).get("min_size_gb", 0.1)
    max_gb = args.max_gb if args.max_gb is not None else META.get("global_constraints", {}).get("max_size_gb", 50.0)
    prefix_str = META.get("name_prefix", "")
    suffix_str = args.suffix if args.suffix is not None else META.get("name_suffix", "")

    shared_root = INPUT_ROOT
    # Pre-load models globally once to prevent multiprocess race conditions on HF cache
    print("[PRE-FLIGHT] Analyzing task requirements...")
    from models.quality_scorer import QualitySentry
    from models.diffusion import CaptionSentry
    from models.encoder import CLIPManifold
    from models.detection import AutoLabeler

    # Analyze if any target models need AI augmentation
    needs_captioning = False
    needs_styling = False

    for model_key in DATASETS_META:
        if args.model and model_key != args.model: continue
        task = detect_task(model_key)
        if task == "diffusion": needs_captioning = True
        if task == "diffusion" or model_key == "nima_aesthetic": needs_styling = True

    if needs_captioning and not args.no_vetting:
        print("[PRE-FLIGHT] Caching CaptionSentry (BLIP)...")
        tmp = CaptionSentry(device="cpu")
        del tmp

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    if needs_styling and not args.no_vetting:
        if needs_captioning:
            print("[PRE-FLIGHT] Caching CLIPManifold...")
            _ = CLIPManifold(device="cpu")

    print("[PRE-FLIGHT] Pre-flight analysis complete.")

    # 2026 Resilience: Adaptive worker scaling (v5.1)
    final_workers = args.workers if args.workers else CONFIG.get("num_workers", 4)

    if not args.workers and final_workers > 8:
        print(f"[RESILIENCE] Capping auto-detected workers to 8 for stability. Use --workers to override.")
        final_workers = 8

    # 2026 DPED Optimization: Pre-cache canon paths to avoid O(N) exists() calls
    dped_canon_paths = set()
    for model_key, model_config in DATASETS_META.items():
        if args.model and model_key != args.model: continue
        for ref_entry in cast("list[dict[str, Any]]", model_config.get("refs", [])):
            if "dped" in ref_entry["ref"].lower():
                slug = ref_entry["ref"].split("/")[-1].lower()
                canon_roots = [
                    shared_root / slug / "iphone2canon" / "train" / "canon",
                    shared_root / slug / "iphone2canon" / "test" / "canon"
                ]
                for cr in canon_roots:
                    if cr.exists():
                        print(f"[DPED] Caching ground truth manifold for {slug} ({cr.parent.name})...")
                        for r, _, f_list in os.walk(cr):
                            for f in f_list:
                                dped_canon_paths.add(os.path.join(r, f).replace("\\", "/").lower())

    max_workers = int(max(1, final_workers))
    print(f"[PRE-FLIGHT] Python: {sys.executable}")
    print(f"[PRE-FLIGHT] Hardware: {get_device_info()} | Active Workers: {max_workers}", flush=True)

    if max_workers > 4 and args.no_vetting and args.no_labeling:
        print("[I/O-GEAR] WARNING: High worker count detected for I/O-bound task.")
        print("   -> On mechanical HDDs, this will cause SEVERE thrashing (seeking contention).")
        print("   -> If performance is < 10it/s, restart with --workers 2 or 4.")

    ExecutorClass = ThreadPoolExecutor

    for model_key, model_config in DATASETS_META.items():
        if args.model and model_key != args.model: continue
        task = detect_task(model_key)

        pascal_name = model_config.get("name", model_key.replace("_", ""))
        prefix = pascal_name

        output_root = OUT_PARENT / f"{prefix_str}{pascal_name}{suffix_str}"
        output_root_str = str(output_root)

        if not output_root.exists():
            if model_config.get("dataset_type") != "forex" and model_config.get("acquisition_mode") != "mt5_terminal":
                for s in ["train", "val"]: (output_root / "images" / s).mkdir(parents=True, exist_ok=True)
                if task in ["quality", "classification", "detection", "pose", "yolo"]:
                    for s in ["train", "val"]: (output_root / "labels" / s).mkdir(parents=True, exist_ok=True)
                elif task == "segmentation":
                    for s in ["train", "val"]: (output_root / "masks" / s).mkdir(parents=True, exist_ok=True)
                elif task in ["restoration", "super-resolution"]:
                    for s in ["train", "val"]: (output_root / "targets" / s).mkdir(parents=True, exist_ok=True)

        print(f"\n[SOTA v5.0] Commencing compilation for {pascal_name} -> {output_root.name}...")

        # ─── FOREX MANIFOLD ────────────────────────────────────────────────────
        if model_config.get("dataset_type") == "forex" or model_config.get("acquisition_mode") == "mt5_terminal":
            print(f"\n[FOREX MANIFOLD] Compiling Foundation Matrix: {output_root.name}...")
            output_root.mkdir(parents=True, exist_ok=True)

            for empty_dir in [output_root / "images", output_root / "labels", output_root / "masks", output_root / "targets"]:
                if empty_dir.exists():
                    shutil.rmtree(empty_dir)

            pairs_list = model_config.get('pairs', [])
            tfs_list = model_config.get('timeframe_rungs', [1, 5, 15, 60, 240, 1440])
            start_date_str = model_config.get('start_date', '2019-01-01')
            lookback_bars = model_config.get('lookback_bars', 168)

            full_name = prefix_str + pascal_name + suffix_str

            print(f" -> Engaging MT5 Auto-Acquisition Bridge for {len(pairs_list)} symbols across {len(tfs_list)} timeframes...")
            try:
                from mt5_pipeline import run_download_pipeline

                dataset_defs = [{
                    "name": full_name,
                    "pairs": pairs_list,
                    "timeframes": tfs_list,
                    "start_date": start_date_str
                }]

                run_download_pipeline(
                    dataset_defs=dataset_defs,
                    out_dir=str(output_root.parent),
                    login=None, password=None, server=None
                )
            except Exception as e:
                print(f" -> [CRITICAL FAILURE] Temporal compilation dropped: {e}")
                raise e

            from notebook_generator import generate_training_notebook, generate_colab_training_notebook
            target_model = "forex_predictor" if model_key == "forex_universe" else model_key
            generate_training_notebook(pascal_name, target_model, str(output_root / f"{target_model}_training.ipynb"))
            generate_colab_training_notebook(pascal_name, target_model, str(output_root / f"{target_model}_colab_training.ipynb"))

            category_str = model_config.get('category', 'Forex & Financial Time-Series')
            yaml_info = {
                'name': pascal_name,
                'dataset_type': 'forex',
                'category': category_str,
                'pairs': pairs_list,
                'timeframe_rungs': tfs_list,
                'start_date': start_date_str,
                'lookback_bars': lookback_bars,
                'last_processed': datetime.now().isoformat()
            }
            with open(output_root / "dataset_info.yaml", "w", encoding="utf-8") as f:
                yaml.dump(yaml_info, f, default_flow_style=False)

            try:
                generate_dataset_docs(output_root, final_index=None, pascal_name=pascal_name, overrides=yaml_info)
            except Exception as e:
                print(f" -> [WARNING] Documentation engine skipped: {e}")

            print(f"[SUCCESS] Temporal Chunking compilation successfully staged under storage path!\n")
            continue

        # ─── NON‑FOREX MANIFOLDS ──────────────────────────────────────────────
        index = []
        seen_hashes = set()

        # 2026 Phase 1.3: Registry lives inside the manifold folder.
        db_path = output_root / "manifold_registry.db"
        legacy_path = Path(__file__).parent / ".cache" / f"registry_{pascal_name}.db"
        conn = initialize_registry(db_path, migrate_from=legacy_path)

        existing_names = set()
        if db_path.exists():
            print(f"[RESUMPTION] Scanning {pascal_name} registry for existing entries...")
            try:
                rows = conn.execute("SELECT name FROM registry").fetchall()
                existing_names = {r[0] for r in rows}
                if existing_names:
                    print(f"[OK] Found {len(existing_names)} existing samples. Resuming from checkpoint.")
            except Exception as e:
                print(f"[WARNING] Resumption scan failed: {e}")

        # 2026 Resilience: High-Speed Physical Scan (SOTA v6.2)
        existing_on_disk = set()
        img_dir = output_root / "images"
        if img_dir.exists():
            print(f"[RESUMPTION] Surgical scan of {pascal_name} manifold for physical consistency...")
            count = 0
            _buf = []
            for split in ["train", "val"]:
                split_path = img_dir / split
                if not split_path.exists(): continue
                try:
                    with os.scandir(str(split_path)) as it:
                        for entry in it:
                            if entry.is_file():
                                fname = entry.name
                                dot_idx = fname.find('.')
                                _buf.append(fname[:dot_idx].lower() if dot_idx != -1 else fname.lower())
                                count += 1
                                if count % 5000 == 0:
                                    print(f"   -> Indexed {count // 1000}k samples...", flush=True)
                                    existing_on_disk.update(_buf)
                                    _buf = []
                except OSError: pass
            existing_on_disk.update(_buf)
            _buf = None

            global PHYSICAL_INDEX
            PHYSICAL_INDEX = existing_on_disk
            print(f"[OK] Physical discovery complete: {len(existing_on_disk)} samples verified on disk.")

        if args.no_vetting and args.no_labeling:
            init_worker(CONFIG, dped_canon_paths, existing_on_disk)
            executor_ctx = ExecutorClass(max_workers=max_workers)
        else:
            executor_ctx = ExecutorClass(max_workers=max_workers, initializer=init_worker, initargs=(CONFIG, dped_canon_paths, existing_on_disk))

        executor = executor_ctx

        # 2026 Orphan Rescue
        lower_registry = {n.lower() for n in existing_names}
        orphans = [k for k in existing_on_disk if k not in lower_registry]
        lower_registry = None

        if orphans:
            print(f"[REPAIR] Found {len(orphans)} orphans on disk. Commencing batch adoption...")
            CHUNK_SIZE = 100000
            total_adopted = 0
            for i in range(0, len(orphans), CHUNK_SIZE):
                chunk = orphans[i:i + CHUNK_SIZE]
                orphan_entries = []
                for o_key in chunk:
                    parts = o_key.split("_")
                    o_source = "_".join(parts[1:-1]) if len(parts) >= 3 else "unknown"
                    o_split = "train"
                    orphan_entries.append((
                        o_key, o_source, task, o_split, "adopted", 1.0, None, None, None, None
                    ))

                conn.executemany("""
                    INSERT OR IGNORE INTO registry (name, source, task, split, hash, nima_score, caption, style_tag, clip_latent, img_bytes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, orphan_entries)
                conn.commit()
                total_adopted += len(orphan_entries)
                print(f"   -> Adopted {total_adopted // 1000}k / {len(orphans) // 1000}k orphans...", flush=True)

            print(f"[OK] [REPAIR] {total_adopted} orphans successfully merged into registry.")
            existing_names.update(orphans)
            orphans = None

        sfw_tasks: list[TaskItem] = []
        nsfw_tasks: list[TaskItem] = []

        for ref_entry in cast("list[dict[str, Any]]", model_config.get("refs", [])):
            ref = ref_entry["ref"]
            tag = ref_entry.get("tag", "sfw")
            task_tag = None
            m_name = ""
            if ref.startswith("manifold://"):
                m_name = ref.replace("manifold://", "")

                current_suffix = "" if m_name.endswith("MultiTask") else suffix_str
                m_path = OUT_PARENT / f"{prefix_str}{m_name}{current_suffix}"

                if m_path.exists():
                    dataset = m_path / "images"
                    if not dataset.exists(): dataset = m_path / "targets"
                    if not dataset.exists(): dataset = m_path / "masks"

                    slug = f"compiled_{m_name}"
                    mapping = {
                        "NafNetDebluring": "deblur",
                        "NafNetDenoising": "denoise",
                        "MprNetDeraining": "derain",
                        "FfaNetIndoor": "dehaze_indoor",
                        "FfaNetOutdoor": "dehaze_outdoor",
                        "MirNetLowLight": "lowlight",
                        "MirNetExposure": "exposure",
                        "UltraZoom": "superres",
                        "FilmRestorer": "vintage",
                        "CodeFormer": "face_restorer",
                        "ParseNet": "face_parser"
                    }
                    task_tag = mapping.get(m_name)
                    print(f"[RECIRCULATION] Using compiled manifold: {m_name} | Task Tag: {task_tag}")
                else:
                    print(f"[SKIP] Manifold {m_name} not found at {m_path}")
                    continue
            else:
                slug = ref.replace('hf://', '').replace('gh://', '').replace('kaggle://', '').split('/')[-1]
                if ":" in slug:
                    repo_slug = slug.split(":")[0]
                    target_slug = slug.split(":")[-1].replace(".tgz", "").replace(".tar.gz", "").replace(".zip", "")
                    dataset = shared_root / repo_slug / target_slug
                    slug = target_slug
                else:
                    dataset = shared_root / slug

            if task_tag:
                c_slug = f"{task_tag}_compiled_{m_name}"
            else:
                c_slug = clean_slug(slug)
            if not dataset.is_dir():
                dataset = shared_root / slug.lower()
                if not dataset.is_dir():
                    try:
                        matches = [d for d in shared_root.iterdir() if d.is_dir() and slug.lower() in d.name.lower()]
                        if matches:
                            dataset = matches[0]
                            print(f"[DISCOVERY] Mapping {ref} -> {dataset.name}")
                    except Exception:
                        pass

            if not dataset.is_dir():
                print(f"[SKIP] Source {ref} not found in {shared_root}")
                continue

            fmt, ann_path = detect_annotations(dataset)
            ann_data: Any = None
            ann_data_list = []
            if fmt == "coco":
                ann_data = parse_coco(_require_ann_path(ann_path))
            elif fmt == "parquet":
                ann_paths = list(dataset.rglob("*.parquet"))
                ann_data_list = []
                for ap in ann_paths:
                    try:
                        ann_data_list.append(parse_parquet(ap))
                    except Exception as e:
                        print(f"[WARNING] Failed to parse {ap}: {e}")
                ann_data = ann_data_list[0] if ann_data_list else None
            elif fmt == "matlab":
                ann_data = parse_matlab(_require_ann_path(ann_path))
            elif fmt in ["xml", "yolo"]:
                ann_data = _require_ann_path(ann_path)

            valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".safetensors", ".tiff", ".tif", ".bmp", ".npy"}
            images = list(_fast_scan(str(dataset), valid_exts))

            is_virtual = False
            if not images and fmt == "parquet" and ann_data_list:
                for pq_path, _, cols in ann_data_list:
                    if "image" in cols or "pixel_values" in cols:
                        is_virtual = True
                        print(f"[VIRTUAL] {slug} identified as Sharded Parquet dataset ({len(ann_data_list)} shards).")
                        break

            is_lazy = False
            if not images and not is_virtual and fmt == "parquet" and ann_data_list:
                for pq_path, _, cols in ann_data_list:
                    if "url" in cols:
                        is_lazy = True
                        print(f"[LAZY] {slug} identified as URL-based manifest. Commencing background retrieval...")
                        break

            if is_lazy:
                dl_dir = dataset / "downloads"
                dl_dir.mkdir(exist_ok=True)

                to_download = []
                for pq_path, mapping, cols in ann_data_list:
                    url_col = mapping.get("url", "url")
                    key_col = mapping.get("key", "key")
                    if url_col in cols:
                        try:
                            df = pd.read_parquet(pq_path)
                        except Exception as e:
                            print(f"[WARNING] Skipping corrupted lazy parquet shard {pq_path}: {e}")
                            continue
                        for row in df.itertuples():
                            url = getattr(row, url_col)
                            key = str(getattr(row, key_col, hashlib.md5(url.encode()).hexdigest()))
                            ext = ".jpg"
                            dest = dl_dir / f"{key}{ext}"
                            if not dest.exists():
                                to_download.append((url, str(dest)))

                if to_download:
                    print(f"[RETRIEVAL] Downloading {len(to_download)} missing images for {slug}...")
                    with requests.Session() as session:
                        with ThreadPoolExecutor(max_workers=16) as dl_executor:
                            dl_tasks = [dl_executor.submit(download_image, url, dest, session) for url, dest in to_download]
                            for _ in tqdm(as_completed(dl_tasks), total=len(dl_tasks), desc="   -> Downloading", leave=False):
                                pass

                images = list(_fast_scan(str(dl_dir), valid_exts))

            # PRE-COMPUTE ANNOTATION LOOKUPS TO AVOID O(N^2) BOTTLENECKS
            # Explicit types here are essential: without them, the dicts infer
            # as dict[str, Unknown] and downstream .get() lookups lose their
            # key/value types, causing false positives on every consumer.
            coco_file_to_id: dict[str, int] = {}
            parquet_map: dict[str, Any] = {}
            matlab_map: dict[str, list[dict[str, Any]]] = {}

            if fmt == "coco" and ann_data:
                images_meta, anns_meta = cast(CocoAnnData, ann_data)
                for k, v in images_meta.items():
                    coco_file_to_id[v["file_name"]] = k
            elif fmt == "parquet" and ann_data and not is_virtual:
                pq_path, mapping, cols = cast(ParquetAnnData, ann_data)
                try:
                    df = pd.read_parquet(str(pq_path))
                except Exception as e:
                    print(f"[WARNING] Skipping corrupted parquet {pq_path}: {e}")
                    df = pd.DataFrame()
                file_col = mapping.get("file_name", "file_name")
                if file_col in df.columns and len(df) > 0 and (df[file_col].dtype != 'object' or isinstance(df[file_col].iloc[0], str)):
                    for fname, group in df.groupby(file_col):
                        parquet_map[fname] = group
            elif fmt == "matlab" and ann_data:
                data, key = cast(MatlabAnnData, ann_data)
                if key in data:
                    for entry in data[key]:
                        try:
                            fname = entry.get("image_name", entry.get("name"))
                            if fname:
                                if fname not in matlab_map: matlab_map[fname] = []
                                matlab_map[fname].append(entry)
                        except Exception:
                            pass

            if not is_virtual:
                sample_count = len(images)
            else:
                try:
                    sample_count = sum(pd.read_parquet(d[0], columns=[]).shape[0] for d in ann_data_list)
                except Exception:
                    sample_count = 0

            model_val_split = model_config.get("val_split", None)
            if model_val_split is not None:
                train_prob = 1.0 - float(model_val_split)
            elif task == "diffusion" or "image_to_text" in model_key:
                train_prob = 1.0
            else:
                train_prob = CONFIG["train_split"]

            if is_virtual:
                global_idx = 0
                skip_lbl = not model_config.get("labeling", True)

                for pq_path, mapping, cols in ann_data_list:
                    try:
                        import pyarrow.parquet as pq
                        num_rows = pq.read_metadata(str(pq_path)).num_rows
                    except Exception:
                        try:
                            num_rows = pd.read_parquet(pq_path, columns=[]).shape[0]
                        except Exception as e:
                            print(f"[WARNING] Skipping corrupted virtual parquet shard {pq_path}: {e}")
                            continue

                    if num_rows == 0: continue

                    task_item: TaskItem = (process_parquet_shard, pq_path, prefix, c_slug, global_idx, task, fmt, None, output_root_str, skip_lbl, train_prob, existing_names, existing_on_disk, 1.0, num_rows)

                    if tag == "nsfw": nsfw_tasks.append(task_item)
                    else: sfw_tasks.append(task_item)

                    global_idx += num_rows

            else:
                skip_lbl = not model_config.get("labeling", True)

                val_real_count = 0
                val_fake_count = 0

                for i, img_path_str in enumerate(images):
                    name = f"{prefix}_{c_slug}_{i:09d}"

                    if name in existing_names or name.lower() in existing_on_disk:
                        continue

                    img_path = Path(img_path_str)

                    if task == "restoration":
                        p_low = img_path_str.lower()
                        m_low = model_key.lower()
                        if "deraining" not in m_low and "multitask" not in m_low:
                            if any(k in p_low for k in ["rain", "droplet"]): continue
                        if "denoising" in m_low:
                            if any(k in p_low for k in ["blur", "haze", "lowlight", "exposure"]): continue
                        if "debluring" in m_low:
                            if any(k in p_low for k in ["noise", "haze", "lowlight", "exposure"]): continue

                    split = "train" if random.random() < train_prob else "val"

                    if model_key == "codeformer" and "realvsfakefaces" in prefix.lower():
                        if "real" in img_path.parent.name.lower():
                            if val_real_count < 1000:
                                split = "val"
                                val_real_count += 1
                            else:
                                split = "train"
                        elif "fake" in img_path.parent.name.lower():
                            if val_fake_count < 1000:
                                split = "val"
                                val_fake_count += 1
                            else:
                                split = "train"

                    specific_ann_data: Any = None
                    if fmt == "coco" and ann_data:
                        images_meta, anns_meta = cast(CocoAnnData, ann_data)
                        img_id = coco_file_to_id.get(img_path.name)
                        if img_id is not None:
                            specific_ann_data = anns_meta.get(img_id, [])
                    elif fmt == "parquet" and ann_data:
                        pq_path, mapping, cols = cast(ParquetAnnData, ann_data)
                        df_subset = parquet_map.get(img_path.name)
                        if df_subset is not None and not df_subset.empty:
                            specific_ann_data = (df_subset, mapping)
                    elif fmt == "matlab" and ann_data:
                        specific_ann_data = matlab_map.get(img_path.name, [])
                    elif fmt == "safetensors" and ann_data:
                        specific_ann_data = ann_data
                    elif fmt in ["xml", "yolo", "npz"] and ann_data and ann_path:
                        ext = ".xml" if fmt == "xml" else (".txt" if fmt == "yolo" else ".npz")
                        ann_file = ann_path / f"{img_path.stem}{ext}"
                        if ann_file.exists():
                            specific_ann_data = str(ann_file)

                    if task == "diffusion":
                        task_item: TaskItem = (process_diffusion, img_path, prefix, c_slug, i, split, output_root_str)
                    else:
                        task_item: TaskItem = (process_image, img_path, prefix, c_slug, i, task, fmt, specific_ann_data, split, output_root_str, skip_lbl)

                    if tag == "nsfw": nsfw_tasks.append(task_item)
                    else: sfw_tasks.append(task_item)

                print(f"   -> [{slug}] Discovered {sample_count} source tensors.")

        # 2026 Strategy: Dynamic Ratio Balancing (v5.8)
        target_nsfw_ratio = float(model_config.get("nsfw_ratio", 0))

        def _count_one(t: TaskItem) -> int:
            """Return the expected sample count for one task item.

            Parquet shard tasks carry their row count at index 14; every other
            task produces exactly one output.
            """
            if t[0].__name__ == "process_parquet_shard":
                return int(t[14])
            return 1

        def get_count(task_list: list[TaskItem]) -> int:
            return sum(_count_one(t) for t in task_list)

        sfw_count = get_count(sfw_tasks)
        nsfw_count = get_count(nsfw_tasks)

        if target_nsfw_ratio > 0 and nsfw_count > 0:
            max_nsfw = int(sfw_count * target_nsfw_ratio / (1.0 - target_nsfw_ratio))
            if nsfw_count > max_nsfw:
                print(f"[BALANCING] NSFW pool ({nsfw_count}) exceeds {target_nsfw_ratio*100}% cap. Capping at {max_nsfw} samples.")
                nsfw_keep_prob = max_nsfw / nsfw_count

                new_nsfw_tasks: list[TaskItem] = []
                for item in nsfw_tasks:
                    if item[0].__name__ == "process_parquet_shard":
                        new_item = list(item)
                        new_item[13] = nsfw_keep_prob
                        new_item[14] = int(item[14] * nsfw_keep_prob)
                        new_nsfw_tasks.append(tuple(new_item))
                    else:
                        if random.random() <= nsfw_keep_prob:
                            new_nsfw_tasks.append(item)
                nsfw_tasks = new_nsfw_tasks

        all_tasks: list[TaskItem] = sfw_tasks + nsfw_tasks

        if not all_tasks:
            print(f"[NOTICE] No tasks found for {pascal_name}. Manifold is fully processed.")
            continue

        print(f"[MANIFOLD] Found {len(all_tasks)} items needing processing (after disk-skip).")

        compiled_bytes = 0
        processed_count = len(existing_names)
        if not torch.cuda.is_available() and len(all_tasks) > 50000:
            if not args.no_labeling or not args.no_vetting:
                print(f"[CPU-GUARD] Massive dataset ({len(all_tasks)} items) on CPU. Auto-enabling High-Speed Mode.", flush=True)
                args.no_labeling = True
                args.no_vetting = True

        from concurrent.futures import wait, FIRST_COMPLETED
        desc_label = "[PASS 1] Extraction & Vetting" if not args.no_vetting and task in ["quality", "classification"] else "[PASS 1] Extraction & Processing"

        if not args.finalize:
            BATCH_SIZE = 100 if args.no_vetting else 50
            task_batches = [all_tasks[i:i + BATCH_SIZE] for i in range(0, len(all_tasks), BATCH_SIZE)]

            pbar = None
            total_items_to_process = sum(_count_one(t) for t in all_tasks)
            with tqdm(total=total_items_to_process + len(existing_names), initial=len(existing_names), desc=desc_label, smoothing=0.1) as pbar:
                # --- 2026 Resilience: SAFE-START WARMUP (SOTA v6.3) ---
                warmup_limit = min(500, len(all_tasks))
                if warmup_limit > 0:
                    print(f"[SAFE-START] Warming up manifold (Serial Pass: {warmup_limit} samples)...")
                    for i in range(warmup_limit):
                        task_args = all_tasks[i]
                        res = task_args[0](*task_args[1:])
                        pbar.update(1)
                        if res:
                            if not isinstance(res, list):
                                res = [res]
                            batch_entries = []
                            for r in res:
                                if r:
                                    batch_entries.append((
                                        r["name"], r["source"], r["task"], r["split"], r["hash"],
                                        r["nima_score"], r.get("caption"), r.get("style_tag"),
                                        r.get("clip_latent"), r.get("img_bytes")
                                    ))
                            if batch_entries:
                                conn.executemany("""
                                    INSERT OR IGNORE INTO registry (name, source, task, split, hash, nima_score, caption, style_tag, clip_latent, img_bytes)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, batch_entries)
                        if i % 100 == 99:
                            conn.commit()
                    print(f"[SAFE-START] Warmup complete. Engaging Parallel Matrix.")

                remaining_tasks = all_tasks[warmup_limit:]
                task_batches = [remaining_tasks[i:i + BATCH_SIZE] for i in range(0, len(remaining_tasks), BATCH_SIZE)]

                futures = set()
                batch_iter = iter(task_batches)

                num_initial = min(max_workers * 4, len(task_batches))
                for _ in range(num_initial):
                    try:
                        batch = next(batch_iter)
                        futures.add(executor.submit(batch_worker, batch))
                    except StopIteration: break

                try:
                    while futures:
                        done, futures = wait(futures, return_when=FIRST_COMPLETED)
                        batch_entries = []
                        for future in done:
                            try:
                                batch_results = future.result()
                                pbar.update(len(batch_results))
                                for res in batch_results:
                                    if res:
                                        if CONFIG["enable_dedup"] and not args.no_hash and res["hash"] in seen_hashes: continue
                                        if res["hash"]: seen_hashes.add(res["hash"])
                                        compiled_bytes += res.get("size", 0)
                                        batch_entries.append((
                                            res["name"], res["source"], res["task"], res["split"], res["hash"],
                                            res["nima_score"], res.get("caption"), res.get("style_tag"),
                                            res.get("clip_latent"), res.get("img_bytes")
                                        ))
                                        processed_count += 1
                                        if (compiled_bytes / (1024**3)) >= max_gb:
                                            for f in futures: f.cancel()
                                            futures.clear()
                                            break
                            except Exception as e: print(f"[ERROR] Worker Error: {e}")

                        if batch_entries:
                            conn.executemany("""
                                INSERT OR IGNORE INTO registry (name, source, task, split, hash, nima_score, caption, style_tag, clip_latent, img_bytes)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, batch_entries)
                            conn.commit()

                        for _ in range(len(done)):
                            try:
                                batch = next(batch_iter)
                                futures.add(executor.submit(batch_worker, batch))
                            except StopIteration: break
                except KeyboardInterrupt:
                    os._exit(1)

        conn.commit()

        compiled_gb = compiled_bytes / (1024**3)
        if compiled_gb < min_gb:
            print(f"[WARNING] Compiled set size ({compiled_gb:.2f}GB) is below the minimum manifold constraint ({min_gb:.2f}GB).")

        # STEP 2: Style Clustering
        print(f"[STYLING] Commencing Style Clustering on all extracted latents...")
        cursor = conn.execute("SELECT id, clip_latent FROM registry WHERE clip_latent IS NOT NULL")
        ids, latents = [], []
        for row in cursor:
            _lat = np.frombuffer(row[1], dtype=np.float32)
            if len(_lat) > 0:
                ids.append(row[0])
                latents.append(_lat)

        if latents and len(latents) > 0 and len(latents[0]) > 0:
            X = np.stack(latents)
            n_clusters = int(CONFIG.get("n_style_clusters", 16))
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit(X)
            labels = kmeans.labels_
            for i, cid in tqdm(zip(ids, labels), total=len(ids), desc="[STYLING] Updating Clusters"):
                conn.execute("UPDATE registry SET cluster_id = ? WHERE id = ?", (int(cid), i))
            conn.commit()
        else:
            print(f"[STYLING] No valid style latents found. Skipping clustering (Pure Human Mode).")

        # PASS 2: Balanced Interleaving & Sharding per Dataset
        print(f"[SHARD] Commencing PASS 2: Multi-Domain Balanced Sharding...")

        shard_dir = None
        has_diffusion = conn.execute("SELECT 1 FROM registry WHERE task = 'diffusion' LIMIT 1").fetchone() is not None
        if has_diffusion:
            shard_dir = output_root / "shards"
            shard_dir.mkdir(parents=True, exist_ok=True)

        unique_sources = [r[0] for r in conn.execute("SELECT DISTINCT source FROM registry").fetchall()]

        final_index = []
        for source in unique_sources:
            cursor = conn.execute("SELECT * FROM registry WHERE source = ? ORDER BY cluster_id, id", (source,))
            rows = cursor.fetchall()

            if has_diffusion and shard_dir is not None:
                shard_name = f"{prefix_str}{source}{suffix_str}.tar"
                print(f"[SHARD] Writing {shard_name}...")
                sink = wds.TarWriter(str(shard_dir / shard_name))
            else:
                sink = None

            for row in rows:
                res = {"id": row[0], "name": row[1], "source": row[2], "task": row[3], "split": row[4],
                       "hash": row[5], "nima_score": row[6], "caption": row[7], "style_tag": row[8], "cluster_id": row[11]}

                if res["task"] == "diffusion" and row[10] and sink:
                    sink.write({
                        "__key__": res["name"],
                        "jpg": row[10],
                        "txt": res["caption"],
                        "json": json.dumps({"style": res["style_tag"], "cluster": res["cluster_id"], "source": res["source"]})
                    })
                final_index.append(res)

            if sink:
                sink.close()

        random.seed(42)
        random.shuffle(final_index)
        with open(output_root / "index.json", "w", encoding="utf-8") as f:
            json.dump(final_index, f, indent=2)

        remove_empty_dirs(output_root)

        generate_dataset_docs(output_root, final_index, pascal_name)

        try:
            from notebook_generator import generate_training_notebook as gen_nb
            from notebook_generator import generate_colab_training_notebook as gen_colab_nb

            resolved_model = model_key
            if not resolved_model:
                clean_name = pascal_name.replace("LemGendized", "").replace("KaggleReady", "").replace("Large", "").replace("Mini", "")
                import re
                resolved_model = re.sub(r'(?<!^)(?=[A-Z])', '_', clean_name).lower()

            if "naf_net" in resolved_model: resolved_model = resolved_model.replace("naf_net", "nafnet")
            if "upn_v_2" in resolved_model: resolved_model = resolved_model.replace("upn_v_2", "upn_v2")

            gen_nb(pascal_name, resolved_model, output_root / f"{resolved_model}_kaggle_training.ipynb")
            gen_colab_nb(pascal_name, resolved_model, output_root / f"{resolved_model}_colab_training.ipynb")
        except Exception as e:
            print(f"\\n[ERROR] Silent Failure Detected! Could not import notebook_generator: {e}\\n")

        print(f"[SUCCESS] v5.0 Ascension Complete: {len(final_index)} samples compiled for {pascal_name}.")
        executor.shutdown(wait=True)

# ---------------- GENERATORS ----------------

if __name__ == '__main__':
    process_dataset()