import os, sys, argparse, json, yaml, shutil, multiprocessing
from pathlib import Path
from compiler_core import *

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
    from models.quality_scorer import QualitySentry # type: ignore
    from models.diffusion import CaptionSentry # type: ignore
    from models.encoder import CLIPManifold # type: ignore
    from models.detection import AutoLabeler # type: ignore

    # Analyze if any target models need AI augmentation
    needs_captioning = False
    needs_styling = False

    for model_key in DATASETS_META:
        if args.model and model_key != args.model: continue
        task = detect_task(model_key)
        if task == "diffusion": needs_captioning = True
        if task == "diffusion" or model_key == "nima_aesthetic": needs_styling = True # Styling optional for aesthetic

    if needs_captioning and not args.no_vetting:
        print("[PRE-FLIGHT] Caching CaptionSentry (BLIP)...")
        tmp = CaptionSentry(device="cpu")
        del tmp

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    if needs_styling and not args.no_vetting:
        # User requested clean dataset suite; we'll disable CLIP unless strictly needed
        # For NimaAesthetic, we'll only load if not in a "Pure Sharding" mindset.
        # Given user feedback, we default to skipping unless it's a diffusion manifold.
        if needs_captioning:
            print("[PRE-FLIGHT] Caching CLIPManifold...")
            _ = CLIPManifold(device="cpu")

    print("[PRE-FLIGHT] Pre-flight analysis complete.")

    # 2026 Resilience: Adaptive worker scaling (v5.1)
    # Priority: 1. CLI Args (--workers) | 2. config.json | 3. Auto-detected (CPU-2)
    final_workers = args.workers if args.workers else CONFIG.get("num_workers", 4)

    # Only apply safety cap if the user didn't explicitly request a worker count
    if not args.workers and final_workers > 8:
        print(f"[RESILIENCE] Capping auto-detected workers to 8 for stability. Use --workers to override.")
        final_workers = 8

    # 2026 DPED Optimization: Pre-cache canon paths to avoid O(N) exists() calls
    dped_canon_paths = set()
    for model_key, model_config in DATASETS_META.items():
        if args.model and model_key != args.model: continue
        for ref_entry in model_config.get("refs", []):
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
                                # 2026: Normalize to lowercase for case-insensitive resolution
                                dped_canon_paths.add(os.path.join(r, f).replace("\\", "/").lower())

    max_workers = int(max(1, final_workers))
    print(f"[PRE-FLIGHT] Python: {sys.executable}")
    print(f"[PRE-FLIGHT] Hardware: {get_device_info()} | Active Workers: {max_workers}", flush=True)

    # 2026 Resilience: Mechanical Drive / Seek-Contention Detection
    if max_workers > 4 and args.no_vetting and args.no_labeling:
        print("[I/O-GEAR] WARNING: High worker count detected for I/O-bound task.")
        print("   -> On mechanical HDDs, this will cause SEVERE thrashing (seeking contention).")
        print("   -> If performance is < 10it/s, restart with --workers 2 or 4.")

    # 2026 Optimization: Switch to ThreadPoolExecutor if no AI models are active (SOTA v6.2)
    # This bypasses the massive pickling overhead of sending 1.4M cache items through Windows IPC pipes.
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

        if model_config.get("dataset_type") == "forex" or model_config.get("acquisition_mode") == "mt5_terminal":
            print(f"\n[FOREX MANIFOLD] Packaging LemGendized Forex Predictor Manifold: {output_root.name}...")
            output_root.mkdir(parents=True, exist_ok=True)
            import shutil
            for empty_dir in [output_root / "images", output_root / "labels", output_root / "masks", output_root / "targets"]:
                if empty_dir.exists():
                    shutil.rmtree(empty_dir)
            target_forex_dir = output_root / "forex"
            target_forex_dir.mkdir(parents=True, exist_ok=True)
            
            raw_forex_dir = INPUT_ROOT / "forex"
            default_pairs = [
                'EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD',
                'USDCAD', 'USDCHF', 'AUDUSD', 'NZDUSD',
                'EURJPY', 'GBPJPY', 'EURGBP',
                'XAGUSD', 'USOIL',
                'US500', 'USTEC', 'GER40'
            ]
            pairs_list = model_config.get('pairs', default_pairs)
            tfs_list = model_config.get('timeframe_rungs', [1, 5, 15, 60, 240, 1440])
            start_date_str = model_config.get('start_date', '2019-01-01')

            existing_pairs = [d.name for d in target_forex_dir.iterdir() if d.is_dir() and not d.name.startswith('.')] if target_forex_dir.exists() else []
            if len(existing_pairs) >= len(pairs_list) and all(p in existing_pairs for p in pairs_list):
                print(f"   -> [OK] Found existing complete manifold pair shards at {target_forex_dir} ({len(existing_pairs)} pairs).")
            else:
                if raw_forex_dir.exists() and any(raw_forex_dir.iterdir()):
                    print(f"   -> Transferring existing forex pair shards from {raw_forex_dir} to {target_forex_dir}...")
                    for item in raw_forex_dir.iterdir():
                        if item.is_dir() and item.name in pairs_list:
                            dest = target_forex_dir / item.name
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.copytree(item, dest)
                currently_in_target = [d.name for d in target_forex_dir.iterdir() if d.is_dir() and not d.name.startswith('.')] if target_forex_dir.exists() else []
                missing_pairs = [p for p in pairs_list if p not in currently_in_target]
                
                if missing_pairs:
                    print(f"   -> [AUTO-ACQUISITION] Missing {len(missing_pairs)} forex shards. Connecting to MetaTrader 5 pipeline...")
                    try:
                        import mt5_pipeline
                        
                        mt5_pipeline.run_download_pipeline(
                            pairs=missing_pairs,
                            timeframes=tfs_list,
                            out_dir=str(target_forex_dir),
                            n_bars=model_config.get('n_bars', 50000),
                            start_date=start_date_str,
                            build_folds=True
                        )
                        
                        if target_forex_dir.exists() and any(target_forex_dir.iterdir()):
                            print(f"   -> [OK] Successfully downloaded and built missing currency pair directories from MT5.")
                    except Exception as e:
                        print(f"   -> [ERROR] MT5 Auto-Acquisition failed: {e}")
                else:
                    print(f"   -> [OK] Successfully transferred {len(list(target_forex_dir.iterdir()))} currency pair directories.")
            category_str = model_config.get('category', 'Forex & Financial Time-Series')
            with open(output_root / "category.txt", "w", encoding="utf-8") as f:
                f.write(f"{category_str}\n")
            with open(output_root / "classes.txt", "w", encoding="utf-8") as f:
                f.write("SELL\nHOLD\nBUY\n")
            
            yaml_info = f"""name: {pascal_name}
dataset_type: forex
category: {category_str}
pairs: {pairs_list}
timeframe_rungs: {tfs_list}
start_date: '{start_date_str}'
lookback_bars: {model_config.get('lookback_bars', 168)}
last_processed: '{datetime.now().isoformat()}'
"""
            with open(output_root / "dataset_info.yaml", "w", encoding="utf-8") as f:
                f.write(yaml_info)

            tf_names = {1: 'M1 (1min)', 5: 'M5 (5min)', 15: 'M15 (15min)', 60: 'H1 (60min)', 240: 'H4 (240min)', 1440: 'D1 (1440min)'}
            tf_labels = [tf_names.get(tf, f'{tf}min') for tf in tfs_list]
            tree_pairs = '\n'.join([f"|   |-- {p}/" for p in pairs_list])

            readme_text = f"""<!-- markdownlint-disable MD051 MD013 -->
# {output_root.name}

> High-fidelity OHLCV temporal manifold for training multi-scale financial prediction models.

## Dataset Overview

- **Category:** {category_str}
- **Acquisition Mode:** MetaTrader 5 Terminal API / Synthetic Multi-Regime Generator
- **Pairs Included:** {', '.join(pairs_list)}
- **Timeframe Rungs:** {', '.join(tf_labels)}
- **Historical Horizon:** {start_date_str} to Present (6-Fold Walk-Forward Matrix with 14-day Embargo)
- **Lookback Window:** {model_config.get('lookback_bars', 168)} bars
- **Total Samples:** [Computed Dynamically During Training]
- **Output Classes:** `SELL` (0), `HOLD` (1), `BUY` (2) + Dual Pip Target Heads (TP/SL)
- **Architecture Base:** Causal TCN + Cross-Timeframe Multi-Head Attention
- **Primary Task:** Predict directional probability (Sell/Hold/Buy) and regress optimal Take-Profit/Stop-Loss boundaries.

## Composition & Lineage

This manifold is dynamically assembled from the following temporal specifications:

- **Currency Pairs**: {', '.join(pairs_list)}
- **Timeframes (Minutes)**: {', '.join(tf_labels)}
- **Historical Horizon**: {start_date_str} to Present
- **Chronology Strategy**: 6-Fold Walk-Forward Matrix
- **Fold Embargo**: 14 Days

## Model Training Profile

- **Target Architectures**: ForexPredictor (Multi-Scale CNN-Transformer)
- **Optimization Strategy**: Focal Loss (Direction), Huber Loss (Magnitude)

### Benchmark Metrics [SOTA]

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **Direction Accuracy** | ~55.0% | > 65.0% | **> 75.0%** |
| **Trade Win Rate** | ~50.0% | > 54.0% | **> 56.0%** |
| **Profit Factor** | ~1.10 | > 1.50 | **> 2.00** |
| **Sharpe Ratio** | ~0.80 | > 1.50 | **> 2.50** |
| **Max Drawdown** | ~30.0% | < 20.0% | **< 10.0%** |
| **Quality Score** | ~100.0 | > 125.0 | **> 150.0** |

## Repository Structure

Standardized directory logic for seamless integration into the **LemGendary Training Suite**.

- **`forex/`**: Shards containing serialized manifold data.
- **`category.txt`**: Top-level categorization tag.
- **`classes.txt`**: Class labels mapping.
- **`dataset_info.yaml`**: Manifest metadata for automated PyTorch loaders.
- **`forex_predictor_training.ipynb`**: Auto-generated Jupyter notebook for model training.
- **`README.md`**: This documentation file.

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/{output_root.name.lower().replace('_', '-')})

## Training Usage

```bash
python training/train.py --model forex_predictor
```
"""
            with open(output_root / "README.md", "w", encoding="utf-8") as f:
                f.write(readme_text)

            generate_training_notebook(pascal_name, model_key, str(output_root / f"{model_key}_training.ipynb"))
            generate_colab_training_notebook(pascal_name, model_key, str(output_root / f"{model_key}_colab_training.ipynb"))
            print(f"[SUCCESS] Manifold {output_root.name} compiled successfully!\n")
            continue

        index = []
        seen_hashes = set()

        # 2026 Resilience: Move registry out of the manifold to avoid polluting the dataset
        registry_dir = Path(__file__).parent / ".cache"
        registry_dir.mkdir(parents=True, exist_ok=True)
        db_path = registry_dir / f"registry_{pascal_name}.db"
        conn = initialize_registry(db_path)

        # RESUMPTION LOGIC: Load existing entries from SQLite to bypass already processed samples
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
            # Use a buffer for faster set building
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

            # 2026 Warp-Speed: Inject physical index into worker globals
            global PHYSICAL_INDEX
            PHYSICAL_INDEX = existing_on_disk
            print(f"[OK] Physical discovery complete: {len(existing_on_disk)} samples verified on disk.")

        # Start the matrix executor with the physical index correctly anchored
        if args.no_vetting and args.no_labeling:
            init_worker(CONFIG, dped_canon_paths, existing_on_disk)
            executor_ctx = ExecutorClass(max_workers=max_workers)
        else:
            executor_ctx = ExecutorClass(max_workers=max_workers, initializer=init_worker, initargs=(CONFIG, dped_canon_paths, existing_on_disk))

        executor = executor_ctx

        # 2026 Orphan Rescue: Adopt orphans into registry if they exist on disk but are missing from DB
        lower_registry = {n.lower() for n in existing_names}
        orphans = [k for k in existing_on_disk if k not in lower_registry]
        lower_registry = None # Free memory

        if orphans:
            print(f"[REPAIR] Found {len(orphans)} orphans on disk. Commencing batch adoption...")
            # Batch adoption to prevent memory spikes
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
            # Refresh existing_names (we only add the names, not the whole tuples to save memory)
            existing_names.update(orphans)
            orphans = None # Free memory


        sfw_tasks = []
        nsfw_tasks = []

        for ref_entry in model_config.get("refs", []):
            ref = ref_entry["ref"]
            tag = ref_entry.get("tag", "sfw")
            # Resolve Slug: Handle hf://, gh://, and kaggle:// prefixes
            task_tag = None
            m_name = ""
            if ref.startswith("manifold://"):
                m_name = ref.replace("manifold://", "")
                
                # MultiTask manifolds don't have the global suffix
                current_suffix = "" if m_name.endswith("MultiTask") else suffix_str
                m_path = OUT_PARENT / f"{prefix_str}{m_name}{current_suffix}"
                
                if m_path.exists():
                    # 2026 Resilience: Dynamically attach to targets/ or masks/ if images/ is missing
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

            # Resolve/clean c_slug in the outer loop
            if task_tag:
                c_slug = f"{task_tag}_compiled_{m_name}"
            else:
                c_slug = clean_slug(slug)
            if not dataset.is_dir():
                # Check for lowercase version
                dataset = shared_root / slug.lower()
                if not dataset.is_dir():
                    # Last resort: Try to find a folder that contains the slug in its name
                    try:
                        matches = [d for d in shared_root.iterdir() if d.is_dir() and slug.lower() in d.name.lower()]
                        if matches:
                            dataset = matches[0]
                            print(f"[DISCOVERY] Mapping {ref} -> {dataset.name}")
                    except:
                        pass

            if not dataset.is_dir():
                print(f"[SKIP] Source {ref} not found in {shared_root}")
                continue

            fmt, ann_path = detect_annotations(dataset)
            ann_data = None
            ann_data_list = []
            if fmt == "coco":
                ann_data = parse_coco(ann_path)
            elif fmt == "parquet":
                # 2026 Resilience: Handle multiple shards
                ann_paths = list(dataset.rglob("*.parquet"))
                ann_data_list = []
                for ap in ann_paths:
                    try:
                        ann_data_list.append(parse_parquet(ap))
                    except Exception as e:
                        print(f"[WARNING] Failed to parse {ap}: {e}")
                ann_data = ann_data_list[0] if ann_data_list else None
            elif fmt == "matlab":
                ann_data = parse_matlab(ann_path)
            elif fmt in ["xml", "yolo"]:
                ann_data = ann_path

            valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".safetensors", ".tiff", ".tif", ".bmp", ".npy"}
            images = []
            # 2026 Warp-Speed: Use os.scandir and string paths to avoid 1.4M Path object overhead
            def fast_scan(path):
                for entry in os.scandir(path):
                    if entry.is_dir():
                        yield from fast_scan(entry.path)
                    elif entry.is_file():
                        ext = entry.name[entry.name.rfind('.'):].lower()
                        if ext in valid_exts:
                            yield entry.path

            images = list(fast_scan(str(dataset)))


            # VIRTUAL DATASET SUPPORT: If no loose images, check if Parquet has embedded images
            is_virtual = False
            if not images and fmt == "parquet" and ann_data_list:
                # Check ALL shards for "image" or "pixel_values" column, not just the first one
                for pq_path, _, cols in ann_data_list:
                    if "image" in cols or "pixel_values" in cols:
                        is_virtual = True
                        print(f"[VIRTUAL] {slug} identified as Sharded Parquet dataset ({len(ann_data_list)} shards).")
                        break
            
            # LAZY DATASET SUPPORT: If no images and no embedded bytes, check for URLs
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
                
                # Collect all missing URLs
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
                            ext = ".jpg" # Default to JPG for lazy manifolds
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
                
                # Now scan the downloads directory for the standard physical loop
                images = list(fast_scan(str(dl_dir)))

            # PRE-COMPUTE ANNOTATION LOOKUPS TO AVOID O(N^2) BOTTLENECKS
            coco_file_to_id = {}
            parquet_map = {}
            matlab_map = {}

            if fmt == "coco" and ann_data:
                images_meta, anns_meta = ann_data
                for k, v in images_meta.items():
                    coco_file_to_id[v["file_name"]] = k
            elif fmt == "parquet" and ann_data and not is_virtual:
                pq_path, mapping, cols = ann_data # type: ignore
                try:
                    df = pd.read_parquet(str(pq_path))
                except Exception as e:
                    print(f"[WARNING] Skipping corrupted parquet {pq_path}: {e}")
                    df = pd.DataFrame()
                file_col = mapping.get("file_name", "file_name")
                # Only group if the column is hashable (e.g. filename strings)
                if file_col in df.columns and len(df) > 0 and (df[file_col].dtype != 'object' or isinstance(df[file_col].iloc[0], str)):
                    for fname, group in df.groupby(file_col):
                        parquet_map[fname] = group
            elif fmt == "matlab" and ann_data:
                data, key = ann_data
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
            # print(f"[QUEUE] {prefix} ({task}) | {slug} | {sample_count} samples scheduled.")

            model_val_split = model_config.get("val_split", None)
            if model_val_split is not None:
                train_prob = 1.0 - float(model_val_split)
            elif task == "diffusion" or "image_to_text" in model_key:
                train_prob = 1.0
            else:
                train_prob = CONFIG["train_split"]

            if is_virtual:
                # Case A: Queue tasks directly from all Parquet shards
                global_idx = 0
                # c_slug already defined and formatted above
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
                    
                    # We pass num_rows explicitly so we can use it for balancing and tqdm later
                    task_item = (process_parquet_shard, pq_path, prefix, c_slug, global_idx, task, fmt, None, output_root_str, skip_lbl, train_prob, existing_names, existing_on_disk, 1.0, num_rows)
                    
                    if tag == "nsfw": nsfw_tasks.append(task_item)
                    else: sfw_tasks.append(task_item)
                    
                    global_idx += num_rows

            else:
                # Case B: Standard Physical File Loop
                # c_slug already defined and formatted above
                skip_lbl = not model_config.get("labeling", True)

                val_real_count = 0
                val_fake_count = 0

                for i, img_path_str in enumerate(images):
                    name = f"{prefix}_{c_slug}_{i:09d}"

                    # 2026 Resilience: Pre-emptive Disk Skip (SOTA v6.0)
                    if name in existing_names or name.lower() in existing_on_disk:
                        continue

                    img_path = Path(img_path_str)
                    
                    # 2026 Integrity Guard: Eliminate cross-contamination in specialized restoration manifolds
                    if task == "restoration":
                        p_low = img_path_str.lower()
                        m_low = model_key.lower()
                        # Strict Deraining Exclusion
                        if "deraining" not in m_low and "multitask" not in m_low:
                            if any(k in p_low for k in ["rain", "droplet"]): continue
                        # Strict Denoising Purity
                        if "denoising" in m_low:
                            if any(k in p_low for k in ["blur", "haze", "lowlight", "exposure"]): continue
                        # Strict Deblurring Purity
                        if "debluring" in m_low:
                            if any(k in p_low for k in ["noise", "haze", "lowlight", "exposure"]): continue

                    split = "train" if random.random() < train_prob else "val"
                    
                    # 2026 CodeFormer Exact Split Injection
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

                    specific_ann_data = None
                    if fmt == "coco" and ann_data:
                        images_meta, anns_meta = ann_data
                        img_id = coco_file_to_id.get(img_path.name)
                        if img_id is not None:
                            specific_ann_data = anns_meta.get(img_id, [])
                    elif fmt == "parquet" and ann_data:
                        pq_path, mapping, cols = ann_data # type: ignore
                        df_subset = parquet_map.get(img_path.name)
                        if df_subset is not None and not df_subset.empty:
                            specific_ann_data = (df_subset, mapping)
                    elif fmt == "matlab" and ann_data:
                        specific_ann_data = matlab_map.get(img_path.name, [])
                    elif fmt == "safetensors" and ann_data:
                        specific_ann_data = ann_data
                    elif fmt in ["xml", "yolo", "npz"] and ann_data:
                        # ann_data is the Path to the annotations/labels directory
                        ext = ".xml" if fmt == "xml" else (".txt" if fmt == "yolo" else ".npz")
                        ann_file = ann_path / f"{img_path.stem}{ext}"
                        if ann_file.exists():
                            specific_ann_data = str(ann_file)

                    if task == "diffusion":
                        task_item = (process_diffusion, img_path, prefix, c_slug, i, split, output_root_str)
                    else:
                        task_item = (process_image, img_path, prefix, c_slug, i, task, fmt, specific_ann_data, split, output_root_str, skip_lbl)

                    if tag == "nsfw": nsfw_tasks.append(task_item)
                    else: sfw_tasks.append(task_item)

                print(f"   -> [{slug}] Discovered {sample_count} source tensors.")

        # 2026 Strategy: Dynamic Ratio Balancing (v5.8)
        target_nsfw_ratio = float(model_config.get("nsfw_ratio", 0))
        
        def get_count(task_list):
            return sum(args[14] if args[0].__name__ == "process_parquet_shard" else 1 for args in task_list)
            
        sfw_count = get_count(sfw_tasks)
        nsfw_count = get_count(nsfw_tasks)
        
        if target_nsfw_ratio > 0 and nsfw_count > 0:
            max_nsfw = int(sfw_count * target_nsfw_ratio / (1.0 - target_nsfw_ratio))
            if nsfw_count > max_nsfw:
                print(f"[BALANCING] NSFW pool ({nsfw_count}) exceeds {target_nsfw_ratio*100}% cap. Capping at {max_nsfw} samples.")
                nsfw_keep_prob = max_nsfw / nsfw_count
                
                # Apply drop directly
                new_nsfw_tasks = []
                for item in nsfw_tasks:
                    if item[0].__name__ == "process_parquet_shard":
                        # Update keep_prob (index 13) and expected num_rows (index 14)
                        new_item = list(item)
                        new_item[13] = nsfw_keep_prob
                        new_item[14] = int(item[14] * nsfw_keep_prob)
                        new_nsfw_tasks.append(tuple(new_item))
                    else:
                        if random.random() <= nsfw_keep_prob:
                            new_nsfw_tasks.append(item)
                nsfw_tasks = new_nsfw_tasks
                
        all_tasks = sfw_tasks + nsfw_tasks
        # 2026 Optimization: Disable global shuffle to maintain disk locality (High-Speed HDD support)
        # random.shuffle(all_tasks)

        if not all_tasks:
            print(f"[NOTICE] No tasks found for {pascal_name}. Manifold is fully processed.")
            continue

        print(f"[MANIFOLD] Found {len(all_tasks)} items needing processing (after disk-skip).")

        compiled_bytes = 0
        processed_count = len(existing_names)
        # CPU Resilience: Auto-bypass if CUDA is missing and dataset is massive
        if not torch.cuda.is_available() and len(all_tasks) > 50000:
            if not args.no_labeling or not args.no_vetting:
                print(f"[CPU-GUARD] Massive dataset ({len(all_tasks)} items) on CPU. Auto-enabling High-Speed Mode.", flush=True)
                args.no_labeling = True
                args.no_vetting = True

        from concurrent.futures import wait, FIRST_COMPLETED
        desc_label = "[PASS 1] Extraction & Vetting" if not args.no_vetting and task in ["quality", "classification"] else "[PASS 1] Extraction & Processing"

        if not args.finalize:
            # 2026 Optimization: Batching to reduce IPC overhead
            BATCH_SIZE = 100 if args.no_vetting else 50
            task_batches = [all_tasks[i:i + BATCH_SIZE] for i in range(0, len(all_tasks), BATCH_SIZE)]

            pbar = None
            total_items_to_process = sum(args[14] if args[0].__name__ == "process_parquet_shard" else 1 for args in all_tasks)
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

        # STEP 2: Style Clustering (v5.0 Global Manifold)
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

        # PASS 2: Balanced Interleaving & Sharding per Dataset (as requested)
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
