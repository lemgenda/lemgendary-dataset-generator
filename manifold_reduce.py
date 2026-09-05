import os, sys, argparse, json, yaml, shutil, multiprocessing
from pathlib import Path
from compiler_core import *

def reduce_dataset():
    print("\n[SCANNING] Locating existing manifolds in LemGendaryDatasets...")
    manifolds = [d for d in OUT_PARENT.iterdir() if d.is_dir() and ((d / "images").exists() or (d / "targets").exists()) and d.name.endswith("Large")]
    if not manifolds:
        print("[ERROR] No valid Large datasets found to reduce.")
        return

    for i, m in enumerate(manifolds):
        base_name = m.name[:-5] # Strip 'Large'
        kaggle_ready_path = m.parent / f"{base_name}KaggleReady"
        if kaggle_ready_path.exists():
            print(f"\033[92m{i+1}. {m.name} (KaggleReady exists)\033[0m")
        else:
            print(f"\033[93m{i+1}. {m.name}\033[0m")

    try:
        sel = input("\nSelect manifold to sample (number, comma-separated numbers, or 'a' for all): ").strip()
        if not sel: return
        if sel.lower() == 'a':
            target_indices = list(range(len(manifolds)))
        else:
            target_indices = []
            for part in sel.split(','):
                idx = int(part.strip()) - 1
                if idx < 0 or idx >= len(manifolds): raise ValueError
                target_indices.append(idx)
    except (ValueError, IndexError):
        print("[ERROR] Invalid selection.")
        return
    except KeyboardInterrupt:
        print("\n[ABORTED] Operation cancelled by user.")
        return

    try:
        max_gb = float(input("Target max size in GB (e.g. 15.0): "))
        suffix = input("New suffix (e.g. Mini): ").strip()
    except ValueError:
        print("[ERROR] Invalid input.")
        return
    except KeyboardInterrupt:
        print("\n[ABORTED] Operation cancelled by user.")
        return

    for idx in target_indices:
        source_root = manifolds[idx]

        old_suffix = str(CONFIG.get("name_suffix", "Large"))
        if source_root.name.endswith(old_suffix):
            base_name = source_root.name[:-len(old_suffix)]
        else:
            base_name = source_root.name

        target_name = f"{base_name}{suffix}"
        target_root = OUT_PARENT / target_name

        # Detect dataset type from dataset_info.yaml
        dataset_type = "quality"
        info_path = source_root / "dataset_info.yaml"
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = yaml.safe_load(f)
                dataset_type = info.get("dataset_type", "quality")

        print(f"\n[REDUCING] {source_root.name} -> {target_name} ({max_gb} GB)...")

        if dataset_type == "forex":
            _reduce_forex_dataset(source_root, target_root, info)
            continue

        for d in ["images", "labels", "targets", "masks"]:
            for s in ["train", "val"]: (target_root / d / s).mkdir(parents=True, exist_ok=True)


        new_index = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        # Calculate dynamic physical train_prob to enforce disjoint subsets perfectly
        primary_dir_name = "images" if (source_root / "images").exists() else "targets"
        train_dir = source_root / primary_dir_name / "train"
        val_dir = source_root / primary_dir_name / "val"
        train_count = sum(1 for _ in train_dir.iterdir() if _.is_file()) if train_dir.exists() else 0
        val_count = sum(1 for _ in val_dir.iterdir() if _.is_file()) if val_dir.exists() else 0
        total_count = train_count + val_count
        train_prob = train_count / total_count if total_count > 0 else 1.0

        for split in ["train", "val"]:
            img_dir = source_root / "images" / split
            lbl_dir = source_root / "labels" / split
            tgt_dir = source_root / "targets" / split
            mask_dir = source_root / "masks" / split
            primary_split_dir = source_root / primary_dir_name / split

            if not primary_split_dir.exists(): continue

            # Single-pass iteration for extreme speed (165k+ files)
            all_imgs = [p for p in primary_split_dir.iterdir() if p.suffix.lower() in valid_exts]

            if not all_imgs: continue

            # 2026 Resilience: Group by source slug to ensure balanced representation from all sources
            from collections import defaultdict
            images_by_slug = defaultdict(list)
            for img in all_imgs:
                try:
                    # The filename format is prefix_slug_idx.ext (e.g. data_koniq_000000000.jpg)
                    slug = img.name.split('_')[1]
                except IndexError:
                    slug = "unknown"
                images_by_slug[slug].append(img)

            for slug in images_by_slug:
                random.shuffle(images_by_slug[slug])

            # Round-robin interleaved sampling: pulls 1 image from every dataset sequentially
            sampled_imgs = []
            lists = list(images_by_slug.values())
            while lists:
                lists = [lst for lst in lists if lst]
                if not lists: break
                for lst in lists:
                    if lst:
                        sampled_imgs.append(lst.pop())

            split_limit_bytes = max_gb * (1024**3) * (train_prob if split == "train" else (1 - train_prob))
            current_bytes = 0

            try:
                with tqdm(total=split_limit_bytes, desc=f"Copying {split}", unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                    for img_path in sampled_imgs:
                        if current_bytes >= split_limit_bytes: break

                        # Copy Primary Data (images or targets)
                        dest_primary = target_root / primary_dir_name / split / img_path.name
                        shutil.copy2(img_path, dest_primary)
                        file_size = dest_primary.stat().st_size

                        # Copy Label
                        lbl_path = lbl_dir / (img_path.stem + ".txt")
                        if lbl_path.exists():
                            dest_lbl = target_root / "labels" / split / lbl_path.name
                            shutil.copy2(lbl_path, dest_lbl)
                            file_size += dest_lbl.stat().st_size

                        if primary_dir_name == "images":
                            # Copy Target
                            tgt_path = tgt_dir / img_path.name
                            if tgt_path.exists():
                                dest_tgt = target_root / "targets" / split / tgt_path.name
                                shutil.copy2(tgt_path, dest_tgt)
                                file_size += dest_tgt.stat().st_size
                        else:
                            # Copy Image (if primary was targets)
                            img_path_alt = img_dir / img_path.name
                            if img_path_alt.exists():
                                dest_img = target_root / "images" / split / img_path_alt.name
                                shutil.copy2(img_path_alt, dest_img)
                                file_size += dest_img.stat().st_size

                        # Copy Mask
                        mask_path = mask_dir / img_path.name
                        if not mask_path.exists():
                            mask_path = mask_dir / (img_path.stem + ".png")
                        if mask_path.exists():
                            dest_mask = target_root / "masks" / split / mask_path.name
                            shutil.copy2(mask_path, dest_mask)
                            file_size += dest_mask.stat().st_size

                        current_bytes += file_size
                        pbar.update(file_size)
                        try:
                            # Extract slug from filename (prefix_slug_idx.ext)
                            slug = img_path.name.split('_')[1]
                        except (IndexError, AttributeError):
                            slug = "unknown"

                        # Resolve task type from parent meta
                        task_type = "quality"
                        for k, v in DATASETS_META.items():
                            if v["name"] in source_root.name:
                                task_type = v.get("task", "quality")
                                break

                        new_index.append({
                            "name": img_path.stem,
                            "split": split,
                            "source": slug,
                            "task": task_type
                        })
            except KeyboardInterrupt:
                print(f"\n[ABORTED] Reduction cancelled by user.")
                return

        with open(target_root / "index.json", "w", encoding="utf-8") as f:
            json.dump(new_index, f, indent=2)

        remove_empty_dirs(target_root)
        generate_dataset_docs(target_root, new_index, target_name)
        print("[INFO] Run `python notebook_generator.py --all` to regenerate notebooks for reduced datasets.")
        print(f"\n[SUCCESS] Reduced manifold created at {target_root.name}")

def _reduce_forex_dataset(source_root, target_root, info):
    import numpy as np
    print("\n--- Forex Reduction Configuration ---")
    
    # 1. Timeframes
    tf_input = input("Select timeframes to keep (comma-separated, e.g., 1,5,15,60,240,1440 or 'a' for all) [Default: a]: ").strip().lower()
    avail_tfs = info.get("timeframe_rungs", [1, 5, 15, 60, 240, 1440])
    if tf_input == 'a' or not tf_input:
        kept_tfs = avail_tfs
    else:
        kept_tfs = [int(x.strip()) for x in tf_input.split(',')]
        
    # 2. Phases
    print("\nPhases: 1 (Titan 4), 2 (G7 Majors), 3 (High-Beta), 4 (Full Universe)")
    phase_input = input("Select phases to keep (comma-separated, e.g., 1,2 or 'a' for all) [Default: a]: ").strip().lower()
    
    phase_map = {
        1: ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
        2: ["USDCAD", "USDCHF", "AUDUSD", "NZDUSD"],
        3: ["EURGBP", "EURJPY", "GBPJPY", "USOIL"],
        4: ["US500", "USTEC", "GER40", "XAGUSD"]
    }
    
    if phase_input == 'a' or not phase_input:
        kept_phases = [1, 2, 3, 4]
    else:
        kept_phases = [int(x.strip()) for x in phase_input.split(',')]
        
    kept_pairs = set()
    for p in kept_phases:
        kept_pairs.update(phase_map.get(p, []))
        
    # 3. Folds
    fold_input = input("\nEnter target number of folds (1 to 6) [Default: 6]: ").strip()
    try:
        target_folds = int(fold_input) if fold_input else 6
    except ValueError:
        target_folds = 6
    target_folds = max(1, min(6, target_folds))
    
    print(f"\n[FOREX REDUCING] Copying to {target_root.name}...")
    
    # Determine the sliding window of original folds
    # If N=3, we take most recent 4 original folds: [3, 4, 5, 6]
    # The first two (3 and 4) are merged into new Fold 1.
    orig_start = 7 - target_folds - 1
    orig_start = max(1, orig_start)
    window = list(range(orig_start, 7)) # e.g., for N=3 -> [3, 4, 5, 6]
    
    # Map original to new folds
    # window[0] and window[1] -> new fold 1
    # window[2] -> new fold 2, etc.
    fold_mapping = {}
    if len(window) > 1:
        fold_mapping[window[0]] = 1
        fold_mapping[window[1]] = 1
        for i in range(2, len(window)):
            fold_mapping[window[i]] = i
    else:
        fold_mapping[window[0]] = 1
        
    forex_src = source_root / "forex"
    forex_dst = target_root / "forex"
    
    actual_copied_pairs = set()
    
    if forex_src.exists():
        for pair_dir in forex_src.iterdir():
            if not pair_dir.is_dir() or pair_dir.name not in kept_pairs:
                continue
                
            actual_copied_pairs.add(pair_dir.name)
            for tf_dir in pair_dir.iterdir():
                if not tf_dir.is_dir(): continue
                try:
                    if int(tf_dir.name) not in kept_tfs: continue
                except ValueError:
                    continue
                    
                folds_src = tf_dir / "folds"
                folds_dst = forex_dst / pair_dir.name / tf_dir.name / "folds"
                folds_dst.mkdir(parents=True, exist_ok=True)
                
                # Copy val fold
                val_src = folds_src / "val"
                if val_src.exists():
                    shutil.copytree(val_src, folds_dst / "val", dirs_exist_ok=True)
                    
                # Process the numbered folds
                merged_fold_1_srcs = []
                for orig_f, new_f in fold_mapping.items():
                    f_src = folds_src / f"fold_{orig_f}"
                    if not f_src.exists(): continue
                    
                    f_dst = folds_dst / f"fold_{new_f}"
                    if new_f == 1 and target_folds < 6:
                        merged_fold_1_srcs.append(f_src)
                    else:
                        shutil.copytree(f_src, f_dst, dirs_exist_ok=True)
                        
                # Merge into fold_1 if needed
                if merged_fold_1_srcs:
                    f1_dst = folds_dst / "fold_1"
                    f1_dst.mkdir(parents=True, exist_ok=True)
                    
                    files_to_merge = ["X.npy", "y_dir.npy", "y_mag.npy", "timestamps.npy"]
                    for fname in files_to_merge:
                        arrs = []
                        for m_src in merged_fold_1_srcs:
                            fpath = m_src / fname
                            if fpath.exists():
                                arrs.append(np.load(fpath))
                        if arrs:
                            merged_arr = np.concatenate(arrs, axis=0)
                            np.save(f1_dst / fname, merged_arr)

    # Update dataset_info.yaml
    if info_path := source_root / "dataset_info.yaml":
        new_info = info.copy()
        new_info["pairs"] = list(actual_copied_pairs)
        new_info["timeframe_rungs"] = kept_tfs
        with open(target_root / "dataset_info.yaml", "w") as f:
            yaml.dump(new_info, f, sort_keys=False)
            
    print(f"\n[SUCCESS] Reduced Forex manifold created at {target_root.name}")

def purge_ghost_manifolds():
    """
    2026 Ghost Manifold Audit (v1.0).
    Identifies and removes folders containing only notebooks with no manifold data.
    """
    print("\n[GHOST-AUDIT] Scanning for empty manifold folders in LemGendaryDatasets...")
    ghosts = []
    if not OUT_PARENT.exists(): return
    for item in OUT_PARENT.iterdir():
        if not item.is_dir() or item.name == ".git": continue
        # Marker Check: A real manifold must have data or an index
        has_data = any((item / d).exists() for d in ["images", "shards", "index.json", "dataset_info.yaml"])
        if not has_data:
            ghosts.append(item)
    
    if ghosts:
        print(f"  [FOUND] {len(ghosts)} ghost folders identified.")
        print(f"  [ACTION] Ghost purge disabled. Keeping folders intact.")
        # for g in ghosts:
        #     try:
        #         shutil.rmtree(g)
        #         print(f"  [PURGED] {g.name}")
        #     except Exception as e:
        #         print(f"  [ERROR] Failed to purge {g.name}: {e}")
    else:
        print("  [OK] No ghost manifolds detected.")


if __name__ == '__main__':
    reduce_dataset()
