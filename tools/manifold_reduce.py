import json
import logging
import math
import random
import shutil
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "tools" else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.compiler_core import (
    CONFIG,
    DATASETS_META,
    OUT_PARENT,
    remove_empty_dirs,
)
from core.doc_generator import generate_dataset_docs


# ─── Phase 0: Data-presence helper ──────────────────────────────────────────
def _has_manifold_data(path: Path) -> bool:
    """Return True if the manifold has actual data (works for both legacy
    `*Large` names and modern suffix-free names)."""
    # Forex: any *.parquet directly in the folder
    try:
        if any(path.glob("*.parquet")):
            return True
    except OSError as exc:
        logger.debug("Error globbing parquet files in %s: %s", path, exc)

    # Image / target / mask manifolds: at least one file in a split folder
    for split in ("train", "val", "test"):
        for sub in ("images", "targets", "masks"):
            d = path / sub / split
            if not d.exists():
                continue
            try:
                if any(f.is_file() for f in d.iterdir()):
                    return True
            except OSError as exc:
                logger.debug("Error reading directory %s: %s", d, exc)
    return False


def _prompt_multiselect(label, options, default_all=True):
    """
    Present a numbered list of options and return the user-selected subset.
    Accepts comma-separated numbers or 'a' for all.
    Returns a list of selected option values.
    """
    print(f"\n{label}")
    for i, opt in enumerate(options):
        print(f"  {i + 1}. {opt}")
    prompt = "Select (comma-separated numbers, or 'a' for all)"
    if default_all:
        prompt += " [Default: all]: "
    else:
        prompt += ": "

    raw = input(prompt).strip().lower()
    if not raw or raw == 'a':
        return list(options)

    selected = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part) - 1
            if idx < 0 or idx >= len(options):
                raise ValueError
            selected.append(options[idx])
        except ValueError:
            print(f"[WARNING] Ignoring invalid selection: '{part}'")
    return selected if selected else list(options)


def _select_manifolds(manifolds):
    try:
        sel = input("\nSelect manifold to reduce (number, comma-separated, or 'a' for all): ").strip()
        if not sel:
            return None
        if sel.lower() == 'a':
            return list(range(len(manifolds)))
        target_indices = []
        for part in sel.split(','):
            idx = int(part.strip()) - 1
            if idx < 0 or idx >= len(manifolds):
                raise ValueError
            target_indices.append(idx)
        return target_indices
    except (ValueError, IndexError):
        print("[ERROR] Invalid selection.")
        return None
    except KeyboardInterrupt:
        print("\n[ABORTED] Operation cancelled by user.")
        return None


DEFAULT_MAX_GB = 190.0
DEFAULT_SUFFIX = "Reduced"


def _prompt_reduction_params(max_gb_override=None, suffix_override=None, interactive=True):
    if not interactive:
        max_gb = float(max_gb_override) if max_gb_override is not None else DEFAULT_MAX_GB
        suffix = str(suffix_override) if suffix_override is not None else DEFAULT_SUFFIX
        return max_gb, suffix
    try:
        if max_gb_override is not None:
            max_gb = float(max_gb_override)
        else:
            raw_gb = input("Target max size in GB [Default: 190.0]: ").strip()
            max_gb = float(raw_gb) if raw_gb else 190.0
        if suffix_override is not None:
            suffix = str(suffix_override)
        else:
            raw_suffix = input("New suffix [Default: Reduced]: ").strip()
            suffix = raw_suffix if raw_suffix else "Reduced"
        return max_gb, suffix
    except ValueError:
        print("[ERROR] Invalid input.")
        return None, None
    except KeyboardInterrupt:
        print("\n[ABORTED] Operation cancelled by user.")
        return None, None


def _find_manifold(manifolds, dataset):
    """Return the manifold folder named ``dataset`` (exact, case-insensitive; legacy suffix tolerated)."""
    wanted = dataset.strip().lower()
    old_suffix = str(CONFIG.get("name_suffix", "Large")).lower()
    for m in manifolds:
        name = m.name.lower()
        if name == wanted or (old_suffix and name == f"{wanted}{old_suffix}"):
            return m
    return None


def _sample_count(total, fraction):
    """Number of samples kept from ``total`` for a sampling ``fraction`` (None keeps all)."""
    if fraction is None or total == 0:
        return total
    return max(1, math.ceil(total * fraction))


def reduce_dataset(
    max_gb_override=None,
    suffix_override=None,
    dataset=None,
    fraction=None,
    assume_yes=False,
    dry_run=False,
):
    """Create reduced manifold variant(s).

    Interactive by default. ``assume_yes`` runs without any prompt (API/jobs) and
    then requires ``dataset``. ``fraction`` in (0, 1] keeps that share of samples
    (vision) or of the most recent years (forex). ``dry_run`` writes nothing.
    Returns a process exit code (0 on success).
    """
    if fraction is not None and not 0 < fraction <= 1:
        print(f"[ERROR] --fraction must be in (0, 1], got {fraction}.")
        return 1
    interactive = not assume_yes
    if not interactive and not dataset:
        print("[ERROR] Non-interactive reduction requires --dataset.")
        return 1

    print("\n[SCANNING] Locating existing manifolds in LemGendaryDatasets...")
    # ─── Phase 0: eligibility filter relaxed to support both legacy and modern names ───
    manifolds = [
        d for d in sorted(OUT_PARENT.iterdir())
        if d.is_dir()
        and d.name.startswith("LemGendized")
        and _has_manifold_data(d)
    ]
    if not manifolds:
        print("[ERROR] No valid manifolds found to reduce.")
        return 1

    if dataset:
        match = _find_manifold(manifolds, dataset)
        if match is None:
            print(f"[ERROR] Manifold '{dataset}' not found or has no loose data to reduce.")
            return 1
        targets = [match]
    else:
        targets = _choose_manifolds_interactively(manifolds)
        if not targets:
            return 1

    for source_root in targets:
        code = _reduce_one(source_root, max_gb_override, suffix_override, fraction, interactive, dry_run)
        if code != 0:
            return code
    return 0


def _choose_manifolds_interactively(manifolds):
    for i, m in enumerate(manifolds):
        # Status display: mark if a KaggleReady / Reduced sibling already exists
        has_reduced = any(
            (m.parent / f"{m.name.rsplit('Large', 1)[0]}{s}").exists()
            for s in ("KaggleReady", "Reduced")
        )
        if has_reduced:
            print(f"\033[92m{i + 1}. {m.name} (reduced variant exists)\033[0m")
        else:
            print(f"\033[93m{i + 1}. {m.name}\033[0m")

    target_indices = _select_manifolds(manifolds)
    if not target_indices:
        return []
    return [manifolds[idx] for idx in target_indices]


def _reduce_one(source_root, max_gb_override, suffix_override, fraction, interactive, dry_run):
    """Reduce a single manifold. Returns an exit code."""
    old_suffix = str(CONFIG.get("name_suffix", "Large"))
    # ─── Phase 0: guard against empty suffix (post-modernization state) ───
    if old_suffix and source_root.name.endswith(old_suffix):
        base_name = source_root.name[:-len(old_suffix)]
    else:
        base_name = source_root.name

    dataset_type = "quality"
    info = {}
    info_path = source_root / "dataset_info.yaml"
    if info_path.exists():
        with open(info_path, 'r', encoding="utf-8") as f:
            info = yaml.safe_load(f) or {}
            dataset_type = info.get("dataset_type", "quality")

    if dataset_type == "forex":
        return _reduce_forex_dataset(
            source_root, base_name, info,
            interactive=interactive, fraction=fraction,
            suffix_override=suffix_override, dry_run=dry_run,
        )

    # --- Vision / quality manifold flow ---
    max_gb, suffix = _prompt_reduction_params(
        max_gb_override=max_gb_override, suffix_override=suffix_override, interactive=interactive
    )
    if max_gb is None:
        return 1

    target_name = f"{base_name}{suffix}"
    target_root = OUT_PARENT / target_name
    share = f", {fraction:.0%} sampling" if fraction is not None else ""

    if dry_run:
        print(f"\n[DRY-RUN] Would reduce {source_root.name} -> {target_name} ({max_gb} GB cap{share}). Nothing written.")
        return 0

    print(f"\n[REDUCING] {source_root.name} -> {target_name} ({max_gb} GB cap{share})...")

    for d in ["images", "labels", "targets", "masks"]:
        for s in ["train", "val"]:
            (target_root / d / s).mkdir(parents=True, exist_ok=True)

    new_index = []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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

        if not primary_split_dir.exists():
            continue

        all_imgs = [p for p in primary_split_dir.iterdir() if p.suffix.lower() in valid_exts]

        if not all_imgs:
            continue

        from collections import defaultdict
        images_by_slug = defaultdict(list)
        for img in all_imgs:
            try:
                slug = img.name.split('_')[1]
            except IndexError:
                slug = "unknown"
            images_by_slug[slug].append(img)

        for slug in images_by_slug:
            random.shuffle(images_by_slug[slug])

        sampled_imgs = []
        lists = list(images_by_slug.values())
        while lists:
            lists = [lst for lst in lists if lst]
            if not lists:
                break
            for lst in lists:
                if lst:
                    sampled_imgs.append(lst.pop())
        sampled_imgs = sampled_imgs[:_sample_count(len(sampled_imgs), fraction)]

        split_limit_bytes = max_gb * (1024 ** 3) * (train_prob if split == "train" else (1 - train_prob))
        current_bytes = 0

        try:
            with tqdm(total=split_limit_bytes, desc=f"Copying {split}", unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for img_path in sampled_imgs:
                    if current_bytes >= split_limit_bytes:
                        break

                    dest_primary = target_root / primary_dir_name / split / img_path.name
                    shutil.copy2(img_path, dest_primary)
                    file_size = dest_primary.stat().st_size

                    lbl_path = lbl_dir / (img_path.stem + ".txt")
                    if lbl_path.exists():
                        dest_lbl = target_root / "labels" / split / lbl_path.name
                        shutil.copy2(lbl_path, dest_lbl)
                        file_size += dest_lbl.stat().st_size

                    if primary_dir_name == "images":
                        tgt_path = tgt_dir / img_path.name
                        if tgt_path.exists():
                            dest_tgt = target_root / "targets" / split / tgt_path.name
                            shutil.copy2(tgt_path, dest_tgt)
                            file_size += dest_tgt.stat().st_size
                    else:
                        img_path_alt = img_dir / img_path.name
                        if img_path_alt.exists():
                            dest_img = target_root / "images" / split / img_path_alt.name
                            shutil.copy2(img_path_alt, dest_img)
                            file_size += dest_img.stat().st_size

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
                        slug = img_path.name.split('_')[1]
                    except (IndexError, AttributeError):
                        slug = "unknown"

                    task_type = "quality"
                    for v in DATASETS_META.values():
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
            print("\n[ABORTED] Reduction cancelled by user.")
            return 1

    with open(target_root / "index.json", "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2)

    remove_empty_dirs(target_root)
    generate_dataset_docs(target_root, new_index, target_name)
    print("[INFO] Run `lemtrain notebooks generate --all` to regenerate notebooks for reduced datasets.")
    print(f"\n[SUCCESS] Reduced manifold created at {target_root.name}")
    return 0


def _reduce_forex_dataset(source_root, base_name, info, interactive=True, fraction=None, suffix_override=None, dry_run=False):

    print(f"\n--- Forex Reduction: {source_root.name} ---")

    # Discover available years from the forex directory structure.
    forex_src = source_root / "forex"
    avail_pairs = sorted(
        d.name for d in forex_src.iterdir() if d.is_dir()
    ) if forex_src.exists() else []

    avail_tfs_raw = set()
    avail_years = set()
    if forex_src.exists():
        for pair_dir in forex_src.iterdir():
            if not pair_dir.is_dir():
                continue
            for tf_dir in pair_dir.iterdir():
                if not tf_dir.is_dir():
                    continue
                try:
                    avail_tfs_raw.add(int(tf_dir.name))
                except ValueError:
                    logger.debug("Skipping non-numeric timeframe directory %s", tf_dir)
                    continue
                # Now look for year subdirectories under the timeframe
                for year_dir in tf_dir.iterdir():
                    if year_dir.is_dir() and year_dir.name.isdigit():
                        avail_years.add(int(year_dir.name))
    avail_tfs = sorted(avail_tfs_raw)
    avail_years = sorted(avail_years)

    if not avail_pairs or not avail_tfs or not avail_years:
        print(f"[ERROR] No compiled forex data found under {forex_src} (missing pairs, timeframes, or year folders).")
        return 1

    # Non-interactive default keeps every year (or the most recent share of them).
    default_years = avail_years[-_sample_count(len(avail_years), fraction):]
    default_start = default_years[0]
    default_end = default_years[-1]
    if interactive:
        # Ask user for year range to keep.
        print(f"\nAvailable years: {avail_years}")
        try:
            start_raw = input(f"Enter start year [Default: {default_start}]: ").strip()
            start_year = int(start_raw) if start_raw else default_start
            end_raw = input(f"Enter end year [Default: {default_end}]: ").strip()
            end_year = int(end_raw) if end_raw else default_end
        except ValueError:
            start_year, end_year = default_start, default_end
        except KeyboardInterrupt:
            print("\n[ABORTED] Operation cancelled by user.")
            return 1
    else:
        start_year, end_year = default_start, default_end

    if start_year > end_year:
        print("[ERROR] Start year must be <= end year.")
        return 1

    # Filter years that actually exist within the range.
    kept_years = [y for y in avail_years if start_year <= y <= end_year]
    if not kept_years:
        print("[ERROR] No years in the specified range exist in the source.")
        return 1

    tf_labels = [str(tf) for tf in avail_tfs]
    if interactive:
        try:
            kept_tf_strs = _prompt_multiselect("Select timeframes to include:", tf_labels)
        except KeyboardInterrupt:
            print("\n[ABORTED] Operation cancelled by user.")
            return 1
    else:
        kept_tf_strs = tf_labels
    kept_tfs = set(int(x) for x in kept_tf_strs)

    # All pairs are kept (unchanged)
    kept_pairs = set(avail_pairs)

    # Output suffix
    if suffix_override is not None:
        suffix = str(suffix_override)
    elif interactive:
        try:
            raw_suffix = input("\nNew suffix [Default: Reduced]: ").strip()
            suffix = raw_suffix if raw_suffix else DEFAULT_SUFFIX
        except KeyboardInterrupt:
            print("\n[ABORTED] Operation cancelled by user.")
            return 1
    else:
        suffix = DEFAULT_SUFFIX

    target_name = f"{base_name}{suffix}"
    target_root = OUT_PARENT / target_name

    plan = f"  Years: {kept_years[0]} – {kept_years[-1]}  |  Timeframes: {sorted(kept_tfs)}  |  Pairs: all ({len(kept_pairs)})"
    if dry_run:
        print(f"\n[DRY-RUN] Would reduce forex {source_root.name} -> {target_name}. Nothing written.")
        print(plan)
        return 0

    print(f"\n[FOREX REDUCING] {source_root.name} -> {target_name}")
    print(plan)

    forex_dst = target_root / "forex"
    actual_copied_pairs = set()

    if forex_src.exists():
        for pair_dir in sorted(forex_src.iterdir()):
            if not pair_dir.is_dir() or pair_dir.name not in kept_pairs:
                continue

            actual_copied_pairs.add(pair_dir.name)

            for tf_dir in sorted(pair_dir.iterdir()):
                if not tf_dir.is_dir():
                    continue
                try:
                    if int(tf_dir.name) not in kept_tfs:
                        continue
                except ValueError:
                    logger.debug("Skipping non-numeric timeframe directory %s", tf_dir)
                    continue

                # Copy only the years we want.
                for year_dir in sorted(tf_dir.iterdir()):
                    if not year_dir.is_dir() or not year_dir.name.isdigit():
                        continue
                    year_int = int(year_dir.name)
                    if year_int not in kept_years:
                        continue

                    dst_year = forex_dst / pair_dir.name / tf_dir.name / year_dir.name
                    dst_year.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(year_dir, dst_year, dirs_exist_ok=True)

    # Write updated dataset_info.yaml with the new year range.
    new_info = info.copy()
    new_info["pairs"] = sorted(actual_copied_pairs)
    new_info["timeframe_rungs"] = sorted(kept_tfs)
    # Adjust start_date to the first kept year (January 1st)
    new_info["start_date"] = f"{kept_years[0]}-01-01"
    # Optionally add an end_date or years field (optional)
    # new_info["years"] = kept_years

    with open(target_root / "dataset_info.yaml", "w", encoding="utf-8") as f:
        yaml.dump(new_info, f, sort_keys=False)

    # Copy companion flat files from source.
    companion_globs = ["*.txt", "*.md", "*.ipynb"]
    for pattern in companion_globs:
        for src_file in source_root.glob(pattern):
            dst_file = target_root / src_file.name
            shutil.copy2(src_file, dst_file)

    print(f"\n[SUCCESS] Reduced Forex manifold created at {target_root.name}")
    return 0


def purge_ghost_manifolds():
    """
    Ghost Manifold Audit: identifies folders with no manifold data (no images, shards,
    index, or dataset_info). Currently reports only; purge is disabled to prevent
    accidental data loss.
    """
    print("\n[GHOST-AUDIT] Scanning for empty manifold folders in LemGendaryDatasets...")
    ghosts = []
    if not OUT_PARENT.exists():
        return
    for item in OUT_PARENT.iterdir():
        if not item.is_dir() or item.name == ".git":
            continue
        has_data = any((item / d).exists() for d in ["images", "shards", "index.json", "dataset_info.yaml"])
        if not has_data:
            ghosts.append(item)

    if ghosts:
        print(f"  [FOUND] {len(ghosts)} ghost folders identified.")
        print("  [ACTION] Ghost purge disabled. Keeping folders intact.")
    else:
        print("  [OK] No ghost manifolds detected.")


def main(argv=None):
    """Command-line entry point; returns an exit code."""
    import argparse

    parser = argparse.ArgumentParser(description="Manifold downsampling and scale reduction")
    parser.add_argument("--reduce", action="store_true", help="Run dataset reduction flow (default action)")
    parser.add_argument("--dataset", default=None, help="Manifold folder name to reduce (required with --yes)")
    parser.add_argument("--fraction", type=float, default=None, help="Share of samples to keep, in (0, 1]")
    parser.add_argument("--max_gb", "--max-gb", type=float, default=None, help="Target max size in GB")
    parser.add_argument("--suffix", default=None, help="Suffix for reduced manifold")
    parser.add_argument("--yes", "-y", action="store_true", help="Non-interactive: never prompt, use defaults")
    parser.add_argument("--dry-run", action="store_true", help="Report the plan only; write nothing")
    args = parser.parse_args(argv)
    return reduce_dataset(
        max_gb_override=args.max_gb,
        suffix_override=args.suffix,
        dataset=args.dataset,
        fraction=args.fraction,
        assume_yes=args.yes,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    sys.exit(main())
