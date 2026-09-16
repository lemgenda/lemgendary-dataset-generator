"""
LemGendary Dataset Compiler — Modernize Manifold

Retires the legacy 'Large' suffix from LemGendary manifold folder names.
    LemGendizedNimaAestheticLarge -> LemGendizedNimaAesthetic

This is a folder rename + metadata regeneration. It does NOT recompile any
images. Fast operation.

Usage:
    python modernize_manifold.py                       # interactive
    python modernize_manifold.py --dry-run             # show plan only
    python modernize_manifold.py --all --yes           # batch, no prompts
    python modernize_manifold.py --datasets nima_technical,nima_aesthetic

Safety:
    - Only datasets with actual data are eligible.
    - Confirmation gate (Type YES) before any rename.
    - Any single rename failure halts the entire batch.
    - name_suffix in unified_data.yaml is only written after ALL renames AND
      Kaggle re-uploads succeed.
    - Renamed Kaggle slugs are NEW (Kaggle has no in-place rename); old
      Kaggle datasets are preserved and must be manually deleted if desired.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
REGISTRY_YAML = ROOT / "unified_data.yaml"
MANIFOLD_SYNC = ROOT / "manifold_sync.py"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

LEGACY_SUFFIX = "Large"
BRAND_PREFIX = "LemGendized"


def _out_parent() -> Path:
    try:
        with open(REGISTRY_YAML, encoding="utf-8") as f:
            reg = yaml.safe_load(f) or {}
        folder = reg.get("_registry_metadata", {}).get(
            "output_folder_name", "../LemGendaryDatasets"
        )
        return (ROOT / folder).resolve()
    except Exception:
        return (ROOT.parent / "LemGendaryDatasets").resolve()


# ─── Data-Presence Check ────────────────────────────────────────────────────
def _has_manifold_data(path: Path) -> bool:
    """Return True if the manifold folder has actual data (not just empty dirs)."""
    # Forex: any *.parquet directly in the folder
    try:
        if any(path.glob("*.parquet")):
            return True
    except OSError:
        pass

    # Image / target / mask manifolds: check top-of-split for at least one file
    for split in ("train", "val", "test"):
        for sub in ("images", "targets", "masks"):
            d = path / sub / split
            if not d.exists():
                continue
            try:
                if any(f.is_file() for f in d.iterdir()):
                    return True
            except OSError:
                pass
    return False


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
        })
    return eligible


# ─── Display ────────────────────────────────────────────────────────────────
def _print_eligible(eligible: list[dict]) -> None:
    print()
    print("── Eligible Manifolds for Modernization ────────────────────────────")
    print(f"{'#':<4} {'Current Folder Name':<48} {'Target Folder Name':<48}")
    print(f"{'─'*4} {'─'*48} {'─'*48}")
    for i, item in enumerate(eligible, 1):
        print(f"{i:<4} {item['folder_name']:<48} {item['target_name']:<48}")
    print()


def _parse_selection(raw: str, eligible: list[dict]) -> list[dict]:
    raw = raw.strip().lower()
    if raw == "a" or raw == "all":
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
        print(f"  {item['folder_name']}  ->  {item['target_name']}")
    print()
    print(f"  Total: {len(selected)} manifold(s) will be renamed.")
    print()
    try:
        ans = input("Type YES to confirm: ").strip()
    except KeyboardInterrupt:
        print("\n[ABORTED] Cancelled by user.")
        return False
    return ans == "YES"


# ─── Rename (all-or-nothing) ────────────────────────────────────────────────
def _rename_batch(selected: list[dict]) -> tuple[bool, list[dict]]:
    """Rename all selected folders. Returns (success, renamed_list).

    Halt-on-any-failure policy: if a rename fails mid-batch, we still report
    which renames succeeded so the user knows the partial state. The caller
    must NOT write name_suffix.
    """
    renamed: list[dict] = []
    for item in selected:
        if item["target_path"].exists():
            print(f"[HALT] Target already exists: {item['target_path']}")
            return False, renamed
        try:
            os.rename(item["current_path"], item["target_path"])
            print(f"[OK] Renamed: {item['folder_name']} -> {item['target_name']}")
            renamed.append(item)
        except OSError as e:
            print(f"[HALT] Rename failed for {item['folder_name']}: {e}")
            return False, renamed
    return True, renamed


# ─── Docs Regeneration ──────────────────────────────────────────────────────
def _regen_docs(item: dict) -> bool:
    """Regenerate dataset_info.yaml, README.md, category.txt, classes.txt."""
    try:
        sys.path.insert(0, str(ROOT))
        from doc_generator import generate_dataset_docs
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
    renamed: list[dict],
    kaggle_ok: list[dict],
    kaggle_fail: list[tuple[dict, str]],
    suffix_written: bool,
) -> None:
    print()
    print("══════════════════════════════════════════════════════════════════════")
    print(" MODERNIZATION SUMMARY")
    print("══════════════════════════════════════════════════════════════════════")
    print(f" Renamed folders: {len(renamed)}")
    for item in renamed:
        print(f"   - {item['folder_name']}  ->  {item['target_name']}")
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
        print("   (name_suffix write blocked — retry modernize after fixing failures)")
    print("══════════════════════════════════════════════════════════════════════")


# ─── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="LemGendary Manifold Modernization (suffix removal)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show plan, do not modify anything")
    parser.add_argument("--all", action="store_true", help="Select all eligible manifolds")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation gate")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated list of folder names, base names, or dataset keys")
    parser.add_argument("--skip-kaggle", action="store_true",
                        help="Skip Kaggle re-upload (metadata still written)")
    args = parser.parse_args()

    if not REGISTRY_YAML.exists():
        print(f"[ERROR] Registry not found: {REGISTRY_YAML}")
        return 1

    registry = _load_registry()
    eligible = _enumerate_eligible(registry)

    if not eligible:
        print("[INFO] No eligible manifolds found (no folders with actual data).")
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
            print(f"  {item['folder_name']}  ->  {item['target_name']}  (kaggle: {new_ref})")
        print()
        print("[DRY-RUN] unified_data.yaml would be updated with new kaggle_ref values.")
        print("[DRY-RUN] name_suffix would be set to '' after all succeed.")
        return 0

    # ── Phase A: Rename batch (all-or-nothing) ─────────────────────────────
    print()
    print("── Phase A: Rename ──────────────────────────────────────────────────")
    rename_ok, renamed = _rename_batch(selected)
    if not rename_ok:
        print()
        print("[HALT] Rename batch aborted. name_suffix NOT written.")
        _report(renamed, [], [], suffix_written=False)
        return 1

    # ── Phase B: Docs regeneration ─────────────────────────────────────────
    print()
    print("── Phase B: Metadata regeneration ───────────────────────────────────")
    for item in renamed:
        _regen_docs(item)

    # ── Phase C: Kaggle re-upload ──────────────────────────────────────────
    kaggle_ok: list[dict] = []
    kaggle_fail: list[tuple[dict, str]] = []
    if args.skip_kaggle:
        print()
        print("── Phase C: Kaggle re-upload SKIPPED (--skip-kaggle) ────────────────")
    else:
        print()
        print("── Phase C: Kaggle re-upload ────────────────────────────────────────")
        for item in renamed:
            ok, msg = _kaggle_reupload(item, dry_run=False)
            if ok:
                print(f"[OK] {item['target_name']}: {msg}")
                kaggle_ok.append(item)
                _update_kaggle_ref(registry, item)
            else:
                print(f"[FAIL] {item['target_name']}: {msg}")
                kaggle_fail.append((item, msg))

    # ── Phase D: Persist registry updates ──────────────────────────────────
    if kaggle_ok or args.skip_kaggle:
        _save_registry(registry)
        print(f"[OK] unified_data.yaml updated ({len(kaggle_ok)} kaggle_ref entries).")

    # ── Phase E: name_suffix write (only if everything succeeded) ──────────
    all_success = (not kaggle_fail) and len(renamed) == len(selected)
    suffix_written = False
    if all_success:
        _write_name_suffix(registry, "")
        suffix_written = True
    else:
        print()
        print("[HALT] name_suffix NOT written — Kaggle failures or incomplete batch.")

    _report(renamed, kaggle_ok, kaggle_fail, suffix_written=suffix_written)
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())