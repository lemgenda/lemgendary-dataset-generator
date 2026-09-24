"""
LemGendary Dataset Compiler — Modernize Manifold

Creates modernized manifold variants from legacy sets:
    LemGendizedNimaAestheticLarge -> LemGendizedNimaAesthetic (new folder)

Safety & Architecture:
    - The legacy manifold folder is PRESERVED untouched (never renamed).
    - A new modernized directory is created for the target manifold.
    - Directory layout, labels, annotations, notebooks, and metadata are replicated.
    - Images are transcoded in parallel to modern WebP (or configured format).
    - Modern dataset metadata (dataset_info.yaml, README.md, index.json) is regenerated.
    - Pluggable container formats (WebDataset / MDS / LitData / Parquet) are generated.
    - Remote Kaggle repositories are synchronized if not skipped.
    - name_suffix in unified_data.yaml is updated only when all eligible sets are modernized.

Usage:
    python tools/modernize_manifold.py                                   # interactive selection
    python tools/modernize_manifold.py --dry-run                         # preview plan
    python tools/modernize_manifold.py --all --yes                       # batch creation
    python tools/modernize_manifold.py --datasets yolov8n --also-format webdataset
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

from PIL import Image
import yaml

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "tools" else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config_schema import ImageFormatPolicy
from formats.transcode import ImageTranscoder

REGISTRY_YAML = ROOT / "unified_data.yaml"
MANIFOLD_SYNC = ROOT / "tools" / "manifold_sync.py" if (ROOT / "tools" / "manifold_sync.py").exists() else ROOT / "manifold_sync.py"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

LEGACY_SUFFIX = "Large"
BRAND_PREFIX = "LemGendized"
_IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _out_parent() -> Path:
    try:
        with open(REGISTRY_YAML, encoding="utf-8") as f:
            reg = yaml.safe_load(f) or {}
        folder = reg.get("_registry_metadata", {}).get(
            "output_folder_name", "../LemGendaryDatasets"
        )
        return (ROOT / folder).resolve()
    except Exception as exc:
        logger.debug("Failed reading registry for output_folder_name, using fallback: %s", exc)
        return (ROOT.parent / "LemGendaryDatasets").resolve()


# ─── Data-Presence Check ────────────────────────────────────────────────────
def _has_manifold_data(path: Path) -> bool:
    """Return True if the manifold folder has actual data (not just empty dirs)."""
    # Forex: any *.parquet directly in the folder
    try:
        if any(path.glob("*.parquet")):
            return True
    except OSError as exc:
        logger.debug("Error checking parquet files in %s: %s", path, exc)

    # Image / target / mask manifolds: check top-of-split for at least one file
    for split in ("train", "val", "test"):
        for sub in ("images", "targets", "masks"):
            d = path / sub / split
            if not d.exists():
                continue
            try:
                if any(f.is_file() for f in d.iterdir()):
                    return True
            except OSError as exc:
                logger.debug("Error checking directory %s: %s", d, exc)
    return False


def _is_manifold_fully_modernized(path: Path) -> bool:
    """Return True only if the target modern manifold is fully generated with docs/index."""
    if not path.exists() or not _has_manifold_data(path):
        return False
    return (path / "dataset_info.yaml").exists() and (path / "index.json").exists()


# ─── Registry I/O ───────────────────────────────────────────────────────────
def _load_registry() -> dict:
    with open(REGISTRY_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_registry(data: dict) -> None:
    with open(REGISTRY_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data, f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def _find_kaggle_key(registry: dict, base_name: str) -> str | None:
    """Find the dataset key whose `name` matches the base_name (no prefix/suffix)."""
    for key, entry in registry.get("datasets", {}).items():
        if entry.get("name", "") == base_name:
            return key
    return None


# ─── Eligibility Scan ───────────────────────────────────────────────────────
def _enumerate_eligible(registry: dict) -> list[dict]:
    out = _out_parent()
    if not out.exists():
        return []

    eligible: list[dict] = []
    for entry in sorted(out.iterdir()):
        if not entry.is_dir():
            continue
        if not entry.name.endswith(LEGACY_SUFFIX):
            continue
        if not entry.name.startswith(BRAND_PREFIX):
            continue
        if not _has_manifold_data(entry):
            continue

        new_name = entry.name[: -len(LEGACY_SUFFIX)]
        target_path = out / new_name

        # If target already exists and is fully modernized, skip it
        if _is_manifold_fully_modernized(target_path):
            continue

        is_resume = target_path.exists() and _has_manifold_data(target_path)

        base = entry.name[len(BRAND_PREFIX): -len(LEGACY_SUFFIX)]
        kaggle_key = _find_kaggle_key(registry, base)
        old_ref = ""
        if kaggle_key:
            old_ref = registry["datasets"][kaggle_key].get("kaggle_ref", "")

        eligible.append({
            "folder_name": entry.name,
            "current_path": entry,
            "target_name": new_name,
            "target_path": target_path,
            "kaggle_key": kaggle_key,
            "old_kaggle_ref": old_ref,
            "base_name": base,
            "is_resume": is_resume,
        })
    return eligible


# ─── Display ────────────────────────────────────────────────────────────────
def _print_eligible(eligible: list[dict]) -> None:
    print()
    print("── Eligible Manifolds for Modernization ────────────────────────────")
    print(f"{'#':<4} {'Source Manifold (Legacy)':<48} {'Target Modern Manifold':<48}")
    print(f"{'─'*4} {'─'*48} {'─'*48}")
    for i, item in enumerate(eligible, 1):
        status = " [RESUMABLE]" if item.get("is_resume") else ""
        print(f"{i:<4} {item['folder_name']:<48} {item['target_name'] + status:<48}")
    print()


def _parse_selection(raw: str, eligible: list[dict]) -> list[dict]:
    raw = raw.strip().lower()
    if raw in ("a", "all"):
        return list(eligible)
    picked: list[dict] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part) - 1
            if 0 <= idx < len(eligible):
                picked.append(eligible[idx])
        except ValueError:
            print(f"[WARN] Ignoring invalid selection: '{part}'")
    return picked


def _match_query(item: dict, query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    return q in {
        item["folder_name"].lower(),
        item["target_name"].lower(),
        item["base_name"].lower(),
        (item["kaggle_key"] or "").lower(),
    }


def _select_from_datasets_arg(
    raw: str, eligible: list[dict]
) -> list[dict]:
    queries = [q.strip() for q in raw.split(",") if q.strip()]
    picked: list[dict] = []
    matched_queries: set[str] = set()
    for item in eligible:
        for q in queries:
            if _match_query(item, q):
                picked.append(item)
                matched_queries.add(q.lower())
                break
    unmatched = [q for q in queries if q.lower() not in matched_queries]
    for q in unmatched:
        print(f"[WARN] No eligible manifold matched: '{q}'")
    return picked


# ─── Confirmation ───────────────────────────────────────────────────────────
def _confirm(selected: list[dict]) -> bool:
    print()
    print("── Modernization Plan ───────────────────────────────────────────────")
    for item in selected:
        status = " (RESUME partial)" if item.get("is_resume") else " (new folder)"
        print(f"  {item['folder_name']}  ->  {item['target_name']}{status}")
    print()
    print(f"  Total: {len(selected)} modernized manifold(s) will be processed.")
    print("  Legacy manifold folder(s) will be preserved untouched.")
    print()
    try:
        ans = input("Type YES to confirm: ").strip()
    except KeyboardInterrupt:
        print("\n[ABORTED] Cancelled by user.")
        return False
    return ans == "YES"


# ─── Modernized Creation ────────────────────────────────────────────────────
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

    for item in selected:
        src_dir: Path = item["current_path"]
        dst_dir: Path = item["target_path"]

        if dst_dir.exists() and _is_manifold_fully_modernized(dst_dir):
            print(f"[SKIP] Target manifold already fully modernized: {dst_dir.name}")
            created.append(item)
            continue

        is_resume = dst_dir.exists() and _has_manifold_data(dst_dir)
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            if is_resume:
                print(f"[RESUME] Target manifold exists with partial data: {dst_dir.name}. Checking for missing files...")
            else:
                print(f"[CREATE] Created modern target directory: {dst_dir.name}")
        except OSError as e:
            print(f"[HALT] Failed creating directory {dst_dir.name}: {e}")
            return False, created

        transcode_tasks: list[tuple[Path, Path, Literal["image", "target", "mask"]]] = []
        copy_tasks: list[tuple[Path, Path]] = []
        skipped_copies = 0
        skipped_images = 0

        for root, _dirs, files in os.walk(src_dir):
            rel_dir = Path(root).relative_to(src_dir)
            target_sub = dst_dir / rel_dir
            target_sub.mkdir(parents=True, exist_ok=True)

            for fname in files:
                src_file = Path(root) / fname
                suffix = src_file.suffix.lower()
                rel_file = rel_dir / fname

                is_image = suffix in _IMAGE_EXTS and any(
                    part in rel_file.parts for part in ("images", "targets", "masks")
                )

                if is_image and not skip_transcode and policy.format != "keep":
                    kind: Literal["image", "target", "mask"] = "image"
                    if "masks" in rel_file.parts:
                        kind = "mask"
                    elif "targets" in rel_file.parts:
                        kind = "target"

                    out_ext = ".webp" if policy.format == "webp" else transcoder.extension_for(policy.format)
                    if kind == "mask" and policy.mask_format == "webp-lossless":
                        out_ext = ".webp"

                    target_file = target_sub / (src_file.stem + out_ext)
                    if target_file.exists() and target_file.stat().st_size > 0:
                        skipped_images += 1
                    else:
                        transcode_tasks.append((src_file, target_file, kind))
                else:
                    target_file = target_sub / fname
                    if target_file.exists() and target_file.stat().st_size == src_file.stat().st_size:
                        skipped_copies += 1
                    else:
                        copy_tasks.append((src_file, target_file))

        if skipped_copies > 0 or skipped_images > 0:
            print(f"[RESUME] Found existing files: {skipped_images} image(s), {skipped_copies} metadata file(s) already complete.")

        if copy_tasks:
            print(f"[COPY] Copying {len(copy_tasks)} structure, label, and metadata file(s)...")
            if tqdm is not None and len(copy_tasks) > 50:
                copy_iter = tqdm(
                    copy_tasks,
                    desc=f"  -> Copying metadata [{dst_dir.name}]",
                    unit="file",
                    ncols=88,
                    mininterval=0.5,
                )
            else:
                copy_iter = copy_tasks

            for s_f, d_f in copy_iter:
                try:
                    shutil.copy2(s_f, d_f)
                except Exception as exc:
                    logger.debug("Copy failed for %s -> %s: %s", s_f, d_f, exc)
        elif skipped_copies > 0:
            print(f"[COPY] All {skipped_copies} structure, label, and metadata file(s) already in place.")

        if transcode_tasks:
            print(f"[TRANSCODE] Modernizing {len(transcode_tasks)} image(s) to {policy.format.upper()} in {dst_dir.name}...")

            def _worker(task: tuple[Path, Path, Literal["image", "target", "mask"]]) -> bool:
                s_p, d_p, k = task
                try:
                    with Image.open(s_p) as img:
                        data, _ = transcoder.encode(img, kind=k)
                    with open(d_p, "wb") as f_out:
                        f_out.write(data)
                    return True
                except Exception as exc:
                    logger.debug("Transcode failure for %s: %s", s_p, exc)
                    try:
                        shutil.copy2(s_p, d_p)
                    except Exception:
                        pass
                    return False

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                if tqdm is not None and len(transcode_tasks) > 20:
                    name_disp = dst_dir.name if len(dst_dir.name) <= 24 else f"{dst_dir.name[:21]}..."
                    results = list(tqdm(
                        executor.map(_worker, transcode_tasks),
                        total=len(transcode_tasks),
                        desc=f"  -> Transcode [{name_disp}]",
                        unit="img",
                        dynamic_ncols=True,
                        bar_format="{l_bar}{bar:15}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                        mininterval=0.5,
                    ))
                else:
                    results = list(executor.map(_worker, transcode_tasks))
            successes = sum(1 for r in results if r)
            print(f"[TRANSCODE] Completed: {successes}/{len(transcode_tasks)} images converted to {policy.format.upper()}.")
        elif skipped_images > 0:
            print(f"[TRANSCODE] All {skipped_images} image(s) already converted to {policy.format.upper()} in {dst_dir.name}.")

        created.append(item)
    return True, created


# ─── Docs Regeneration ──────────────────────────────────────────────────────
def _regen_docs(item: dict) -> bool:
    """Regenerate dataset_info.yaml, README.md, category.txt, classes.txt, and index.json."""
    try:
        from core.doc_generator import generate_dataset_docs
        generate_dataset_docs(
            item["target_path"],
            final_index=None,
            pascal_name=item["base_name"],
        )
        print(f"[OK] Docs regenerated: {item['target_name']}")
        return True
    except Exception as e:
        print(f"[WARN] Docs regeneration failed for {item['target_name']}: {e}")
        return False


# ─── Kaggle Re-upload ───────────────────────────────────────────────────────
def _compute_new_kaggle_ref(item: dict) -> str:
    owner = "lemtreursi"
    old_ref = item.get("old_kaggle_ref") or ""
    if old_ref.startswith("kaggle://"):
        parts = old_ref[len("kaggle://"):].split("/")
        if parts and parts[0]:
            owner = parts[0]
    new_slug = item["target_name"].lower()
    return f"kaggle://{owner}/{new_slug}"


def _kaggle_reupload(item: dict, dry_run: bool = False) -> tuple[bool, str]:
    new_ref = _compute_new_kaggle_ref(item)
    item["new_kaggle_ref"] = new_ref
    clean_ref = new_ref.replace("kaggle://", "")

    if dry_run:
        return True, f"[DRY-RUN] Would upload {item['target_name']} -> {new_ref}"

    python_exe = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    cmd = [
        python_exe, str(MANIFOLD_SYNC),
        "--action", "sync",
        "--model", item["target_name"],
        "--url", clean_ref,
    ]
    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=False, text=True, check=False,
        )
        if result.returncode == 0:
            return True, f"Uploaded to {new_ref}"
        return False, f"manifold_sync exited {result.returncode}"
    except Exception as e:
        return False, f"subprocess error: {e}"


# ─── Registry Update ────────────────────────────────────────────────────────
def _update_kaggle_ref(registry: dict, item: dict) -> bool:
    key = item.get("kaggle_key")
    if not key:
        return False
    new_ref = item.get("new_kaggle_ref", "")
    if not new_ref:
        return False
    registry["datasets"][key]["kaggle_ref"] = new_ref
    return True


def _write_name_suffix(registry: dict, value: str) -> None:
    registry.setdefault("_registry_metadata", {})["name_suffix"] = value
    _save_registry(registry)
    print(f"[OK] unified_data.yaml name_suffix set to: '{value}'")


# ─── Reporting ──────────────────────────────────────────────────────────────
def _report(
    modernized: list[dict],
    kaggle_ok: list[dict],
    kaggle_fail: list[tuple[dict, str]],
    suffix_written: bool,
) -> None:
    print()
    print("══════════════════════════════════════════════════════════════════════")
    print(" MODERNIZATION SUMMARY")
    print("══════════════════════════════════════════════════════════════════════")
    print(f" Modernized manifolds created: {len(modernized)}")
    for item in modernized:
        print(f"   - {item['folder_name']} (source) -> {item['target_name']} (new)")
    if kaggle_ok:
        print(f"\n Kaggle re-uploads succeeded: {len(kaggle_ok)}")
        for item in kaggle_ok:
            print(f"   - {item.get('new_kaggle_ref', '?')}")
    if kaggle_fail:
        print(f"\n Kaggle re-uploads FAILED: {len(kaggle_fail)}")
        for item, err in kaggle_fail:
            print(f"   - {item['target_name']}: {err}")
    print(f"\n name_suffix updated to \"\": {'YES' if suffix_written else 'NO'}")
    if not suffix_written:
        print("   (name_suffix write skipped or preserved for remaining legacy sets)")
    print("══════════════════════════════════════════════════════════════════════")


# ─── Main ───────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LemGendary Manifold Modernization (create modern manifold variants, WebP transcoding, container formats)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show plan, do not modify anything")
    parser.add_argument("--all", action="store_true", help="Select all eligible manifolds")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation gate")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated list of folder names, base names, or dataset keys")
    parser.add_argument("--skip-kaggle", action="store_true",
                        help="Skip Kaggle re-upload (metadata still written)")
    parser.add_argument("--image-format", type=str, default="webp",
                        choices=["webp", "jpeg", "png", "keep"],
                        help="Transcode images during modernization (default: webp)")
    parser.add_argument("--image-quality", type=int, default=92,
                        help="Quality for image transcoding (1-100)")
    parser.add_argument("--target-quality", type=int, default=95,
                        help="Quality for target image transcoding (1-100)")
    parser.add_argument("--mask-format", type=str, default="webp-lossless",
                        choices=["webp-lossless", "png"],
                        help="Format for mask transcoding")
    parser.add_argument("--also-format", type=str, default=None,
                        help="Optional container format to convert dataset into (e.g., webdataset, mds, litdata, parquet)")
    parser.add_argument("--skip-transcode", action="store_true",
                        help="Skip WebP image transcoding")
    parser.add_argument("--skip-container", action="store_true",
                        help="Skip modern container format writing")
    return parser


def run(args: argparse.Namespace) -> int:
    if not REGISTRY_YAML.exists():
        print(f"[ERROR] Registry not found: {REGISTRY_YAML}")
        return 1

    registry = _load_registry()
    eligible = _enumerate_eligible(registry)

    if not eligible:
        print("[INFO] No eligible manifolds found (no unmodernized folders with actual data).")
        return 0

    # ── Selection ──────────────────────────────────────────────────────────
    if args.datasets:
        selected = _select_from_datasets_arg(args.datasets, eligible)
        if not selected:
            print("[ERROR] No manifolds selected.")
            return 1
    elif args.all:
        selected = list(eligible)
    else:
        _print_eligible(eligible)
        try:
            raw = input("Select (comma-separated numbers, or 'a' for all): ").strip()
        except KeyboardInterrupt:
            print("\n[ABORTED] Cancelled by user.")
            return 0
        selected = _parse_selection(raw, eligible)
        if not selected:
            print("[ERROR] No manifolds selected.")
            return 1

    policy = ImageFormatPolicy(
        format=args.image_format,
        quality=args.image_quality,
        target_quality=args.target_quality,
        mask_format=args.mask_format,
    )

    # ── Confirmation ───────────────────────────────────────────────────────
    if not args.yes:
        if not _confirm(selected):
            print("[ABORTED] User declined. No changes made.")
            return 0

    # ── Dry-run short-circuit ──────────────────────────────────────────────
    if args.dry_run:
        print()
        print("[DRY-RUN] Would perform the following:")
        for item in selected:
            new_ref = _compute_new_kaggle_ref(item)
            print(f"  {item['folder_name']}  ->  {item['target_name']} (new folder)  (kaggle: {new_ref})")
            print(f"    + create new modernized manifold directory: {item['target_name']}")
            print("    + copy non-image structures, labels, and metadata files")
            if not args.skip_transcode and policy.format != "keep":
                print(f"    + transcode images -> {args.image_format} (q={args.image_quality})")
            if not args.skip_container and args.also_format:
                print(f"    + convert to modern container format -> {args.also_format}")
            print("    + regenerate modern dataset documentation and index manifests")
            if not args.skip_kaggle:
                print(f"    + upload modernized manifold to kaggle: {new_ref}")
        print()
        print("[DRY-RUN] unified_data.yaml would be updated with new kaggle_ref values.")
        print("[DRY-RUN] name_suffix would be set to '' after all succeed.")
        return 0

    # ── Phase A: Create modernized manifolds ───────────────────────────────
    print()
    print("── Phase A: Create Modernized Manifold Folders ──────────────────────")
    create_ok, modernized = _create_modernized_batch(
        selected,
        policy=policy,
        skip_transcode=args.skip_transcode,
    )
    if not create_ok:
        print()
        print("[HALT] Modernization batch aborted. name_suffix NOT written.")
        _report(modernized, [], [], suffix_written=False)
        return 1

    # ── Phase B: Docs regeneration ─────────────────────────────────────────
    print()
    print("── Phase B: Metadata regeneration ───────────────────────────────────")
    for item in modernized:
        _regen_docs(item)

    # ── Phase C: Modern Container Conversion ───────────────────────────────
    if not args.skip_container and args.also_format:
        print()
        print(f"── Phase C: Modern Container Conversion ({args.also_format}) ──────────")
        try:
            from formats.base import parse_also_format
            from tools import migrate_manifold_format
            container_formats = parse_also_format(args.also_format)
            for item in modernized:
                target_path = item["target_path"]
                print(f"[CONTAINER] Converting {target_path.name} to {', '.join(container_formats)}...")
                migrate_manifold_format.migrate_manifold(
                    root=target_path,
                    formats=container_formats,
                    force_duplicate=True,
                    accept_space_loss=True,
                )
        except Exception as exc:
            print(f"[WARN] Container conversion encountered an issue: {exc}")

    # ── Phase D: Kaggle re-upload ──────────────────────────────────────────
    kaggle_ok: list[dict] = []
    kaggle_fail: list[tuple[dict, str]] = []
    if args.skip_kaggle:
        print()
        print("── Phase D: Kaggle re-upload SKIPPED (--skip-kaggle) ────────────────")
    else:
        print()
        print("── Phase D: Kaggle re-upload ────────────────────────────────────────")
        for item in modernized:
            ok, msg = _kaggle_reupload(item, dry_run=False)
            if ok:
                print(f"[OK] {item['target_name']}: {msg}")
                kaggle_ok.append(item)
                _update_kaggle_ref(registry, item)
            else:
                print(f"[FAIL] {item['target_name']}: {msg}")
                kaggle_fail.append((item, msg))

    # ── Phase E: Persist registry updates ──────────────────────────────────
    if kaggle_ok or args.skip_kaggle:
        _save_registry(registry)
        print(f"[OK] unified_data.yaml updated ({len(kaggle_ok)} kaggle_ref entries).")

    # ── Phase F: name_suffix write (only if all eligible succeeded) ────────
    all_success = (not kaggle_fail) and len(modernized) == len(selected)
    suffix_written = False
    if all_success and len(modernized) == len(eligible):
        _write_name_suffix(registry, "")
        suffix_written = True

    _report(modernized, kaggle_ok, kaggle_fail, suffix_written=suffix_written)
    return 0 if all_success else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())