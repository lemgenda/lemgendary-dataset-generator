"""Upload manifolds to Kaggle and download them back; run as ``python -m tools.manifold_sync``."""

import sys
import shutil
import argparse
from pathlib import Path

import yaml

from core.common_sync import (
    DownloadRequest,
    cleanup_temp_archives,
    perform_dataset_download,
    perform_dataset_upload,
    setup_kaggle_auth,
)

ROOT = (
    Path(__file__).resolve().parent.parent
    if Path(__file__).parent.name == "tools"
    else Path(__file__).resolve().parent
)
# Load dataset registry directly from YAML
OUT_PARENT = (ROOT.parent / "LemGendaryDatasets").resolve()
DATASETS_META = {}
YAML_DATA = {}
META = {}
yaml_file = ROOT / "unified_data.yaml"
if yaml_file.exists():
    with open(yaml_file, "r", encoding="utf-8") as f:
        YAML_DATA = yaml.safe_load(f) or {}
    META = YAML_DATA.get("_registry_metadata", {})
    DATASETS_META = YAML_DATA.get("datasets", {})
    if META.get("output_folder_name"):
        OUT_PARENT = Path(META["output_folder_name"]).resolve()

def setup_auth(
    custom_user: str | None = None,
    custom_token: str | None = None,
    default_user: str | None = None,
) -> bool:
    """Setup Kaggle authentication delegating to core.common_sync SSOT."""
    return setup_kaggle_auth(
        default_user=default_user or "lemtreursi",
        custom_user=custom_user,
        custom_token=custom_token,
    )


def resolve_manifold_folder(manifold_id: str) -> str:
    """Resolve a dataset registry key, slug, or folder name to the canonical folder name."""
    if not manifold_id:
        return ""
    target = manifold_id.strip()
    if (OUT_PARENT / target).exists():
        return target
    prefix = META.get("name_prefix", "LemGendized")
    suffix = META.get("name_suffix", "")
    for k, v in DATASETS_META.items():
        slug = v.get("name", "")
        mod_folder = v.get("modernized_folder") or f"{prefix}{slug}{suffix}"
        compiled = f"{prefix}{slug}{suffix}"
        candidates = {k.lower(), slug.lower(), mod_folder.lower(), compiled.lower()}
        if target.lower() in candidates:
            if (OUT_PARENT / mod_folder).exists():
                return mod_folder
            if (OUT_PARENT / compiled).exists():
                return compiled
            return mod_folder
    return target

def resolve_kaggle_ref(manifold_id: str) -> str | None:
    """Public alias of get_kaggle_ref used by metadata sync callers."""
    return get_kaggle_ref(manifold_id)


def _normalized_id(value: str) -> str:
    """Lower-case ID with separators and the LemGendized / large affixes removed."""
    return (
        value.lower()
        .replace("_", "")
        .replace("-", "")
        .replace("lemgendized", "")
        .replace("large", "")
    )


def get_kaggle_ref(manifold_id: str | None) -> str | None:
    """Retrieve kaggle ref from yaml with multi-variant matching."""
    if not manifold_id:
        return None
    target_clean = manifold_id.strip()
    norm_target = _normalized_id(target_clean)
    prefix = META.get("name_prefix", "LemGendized")
    suffix = META.get("name_suffix", "")

    for k, v in DATASETS_META.items():
        ref = v.get("kaggle_ref", "")
        clean_ref = ref[9:] if ref.startswith("kaggle://") else ref

        # 1. Direct key match (e.g., forex_titan_core)
        if k == target_clean:
            return clean_ref

        # 2. Slug name match (e.g., ForexTitanCore)
        slug_name = v.get("name", "")
        if slug_name and slug_name == target_clean:
            return clean_ref

        # 3. Compiled directory name match (e.g., LemGendizedForexTitanCoreLarge)
        compiled_name = f"{prefix}{slug_name}{suffix}"
        if compiled_name == target_clean:
            return clean_ref

        # 4. Normalized fuzzy match
        if norm_target in [_normalized_id(k), _normalized_id(slug_name)]:
            return clean_ref

    return None

def action_sync(manifold_name, repo_id=None, no_wait=False):
    """Zip and Upload a manifold to Kaggle with byte-metered progress and status monitoring."""
    resolved_folder = resolve_manifold_folder(manifold_name)
    src_dir = OUT_PARENT / resolved_folder
    if not src_dir.exists():
        print(
            f"[ERROR] Local manifold '{manifold_name}' (resolved: '{resolved_folder}') not found at"
            f" {src_dir}"
        )
        return

    resolved_repo = repo_id or get_kaggle_ref(manifold_name) or get_kaggle_ref(resolved_folder)
    if not resolved_repo:
        resolved_repo = f"lemtreursi/{resolved_folder.lower().replace('_', '')}"

    clean_repo_id = resolved_repo.replace("kaggle://", "")
    owner = clean_repo_id.split("/")[0] if "/" in clean_repo_id else "lemtreursi"
    setup_auth(default_user=owner)

    # Pre-cleanup any dangling staging directories (preserves recent archives for resumption)
    cleanup_temp_archives(manifold_name=resolved_folder, base_dir=OUT_PARENT, force=False)

    print(f"\n[SYNC] Preparing to upload '{resolved_folder}' to Kaggle ({clean_repo_id})")
    success = perform_dataset_upload(
        src_path=src_dir,
        clean_repo_id=clean_repo_id,
        no_wait=no_wait
    )
    if not success:
        sys.exit(1)


def action_get(repo_id, output_name=None):
    """Download and extract a manifold from Kaggle with byte-level progress and full resumption."""
    clean_repo_id = repo_id.replace("kaggle://", "")
    slug = clean_repo_id.split("/")[-1]
    if not output_name:
        output_name = slug
        for _, v in DATASETS_META.items():
            ref = v.get("kaggle_ref", "")
            if clean_repo_id.lower() in ref.lower():
                output_name = v.get("modernized_folder") or _compiled_folder_name(
                    v.get("name", "")
                )
                break

    dest_dir = OUT_PARENT / output_name

    print(f"\n[GET] Requesting LemGendized Manifold: {clean_repo_id}")

    # Collision check if directory already exists
    if dest_dir.exists():
        print(f"[WARNING] Local manifold '{output_name}' already exists at {dest_dir}!")
        ans = input("Do you want to RESUME extraction or overwrite? (R/o/n): ").strip().lower()
        if ans == 'o':
            confirm = input(
                f"[CRITICAL] Are you ABSOLUTELY sure? Type 'YES' to delete {dest_dir}: "
            ).strip()
            if confirm == 'YES':
                print("[GET] Purging existing manifold...")
                shutil.rmtree(dest_dir, ignore_errors=True)
            else:
                print("[GET] Override cancelled. Aborting.")
                return
        elif ans == 'n':
            print("[GET] Aborting.")
            return

    # Disk space check (Require at least 20GB free as a baseline safeguard)
    _, _, free = shutil.disk_usage(OUT_PARENT)
    free_gb = free / (1024**3)
    if free_gb < 20.0:
        print(f"[WARNING] Low disk space! Only {free_gb:.2f} GB free on {OUT_PARENT.drive}.")
        ans = input("Continue anyway? (y/N): ").strip().lower()
        if ans != 'y':
            print("[GET] Aborting.")
            return
    else:
        print(f"[GET] Disk check passed ({free_gb:.2f} GB free).")

    owner = clean_repo_id.split("/")[0] if "/" in clean_repo_id else "lemtreursi"
    setup_auth(default_user=owner)

    success = perform_dataset_download(
        DownloadRequest(
            clean_repo_id=clean_repo_id,
            target_dir=dest_dir,
            root_datasets_dir=OUT_PARENT,
            manifold_name=output_name,
            slug=slug,
            is_competition=False,
        )
    )
    if not success:
        sys.exit(1)



def _compiled_folder_name(slug: str) -> str:
    """Compiled manifold folder name for a registry slug (registry prefix + slug + suffix)."""
    return f"{META.get('name_prefix', 'LemGendized')}{slug}{META.get('name_suffix', '')}"


def _build_parser() -> argparse.ArgumentParser:
    """CLI definition for the sync manager."""
    parser = argparse.ArgumentParser(description="LemGendary Manifold Sync Manager")
    parser.add_argument("--action", choices=["sync", "get"], help="Action to perform")
    parser.add_argument(
        "--model", "--manifold", dest="model", type=str, help="Local manifold name or YAML key"
    )
    parser.add_argument("--url", type=str, help="Kaggle Dataset URL or ID (e.g., user/dataset)")
    parser.add_argument("--user", type=str, help="Custom Kaggle Username")
    parser.add_argument("--token", type=str, help="Custom Kaggle API Token")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Skip monitoring Kaggle server-side extraction status",
    )
    parser.add_argument("--push", action="store_true", help="Alias for --action sync")
    parser.add_argument("--pull", action="store_true", help="Alias for --action get")
    parser.add_argument("--force", action="store_true", help="Force sync operation")
    return parser


def _resolve_action(args: argparse.Namespace) -> str:
    """Action from --action / --push / --pull, else an interactive menu."""
    if args.action:
        return args.action
    if args.push:
        return "sync"
    if args.pull:
        return "get"
    print("\n=== LemGendary Sync Manager ===")
    print("1. [SYNC] Zip & Upload local manifold to Kaggle")
    print("2. [GET]  Download & Extract manifold from Kaggle")
    choice = input("Select operation (1/2): ").strip()
    if choice == '1':
        return "sync"
    if choice == '2':
        return "get"
    print("Invalid choice.")
    sys.exit(1)


def _selected_key(sel: str, dataset_keys: list[str]) -> str | None:
    """Registry key for a 1-based menu number, or None when ``sel`` is not one."""
    if sel.isdigit() and 1 <= int(sel) <= len(dataset_keys):
        return dataset_keys[int(sel) - 1]
    return None


def _target_for_key(key: str) -> tuple[str, str | None]:
    """(compiled folder name, Kaggle ref) for a registry key."""
    return _compiled_folder_name(DATASETS_META[key].get("name", "")), get_kaggle_ref(key)


def _pick_sync_target(repo_id: str | None) -> tuple[str, str | None]:
    """Interactive choice of a local manifold to upload."""
    dataset_keys = list(DATASETS_META.keys())
    print("\n--- AVAILABLE MANIFOLDS TO SYNC ---")
    for i, k in enumerate(dataset_keys, 1):
        manifold_folder = DATASETS_META[k].get("modernized_folder") or _compiled_folder_name(
            DATASETS_META[k].get("name", "")
        )
        status_tag = "[COMPILED]" if (OUT_PARENT / manifold_folder).exists() else "[NOT COMPILED]"
        print(f" {i:2d}. {k:<35} ({manifold_folder}) {status_tag}")

    sel = input("\nEnter selection (number, dataset key, or folder name): ").strip()
    chosen_key = _selected_key(sel, dataset_keys)
    if chosen_key:
        return _target_for_key(chosen_key)
    return sel, repo_id


def _pick_get_target() -> tuple[str | None, str | None]:
    """Interactive choice of a Kaggle manifold to download."""
    dataset_keys = list(DATASETS_META.keys())
    print("\n--- AVAILABLE MANIFOLDS ON KAGGLE ---")
    for i, k in enumerate(dataset_keys, 1):
        k_ref = get_kaggle_ref(k) or "N/A"
        print(f" {i:2d}. {k:<35} -> {k_ref}")

    sel = input("\nEnter selection (number, dataset key, or Kaggle ID): ").strip()
    chosen_key = _selected_key(sel, dataset_keys)
    if chosen_key:
        return _target_for_key(chosen_key)
    if "/" in sel:
        return None, sel
    return sel, get_kaggle_ref(sel)


def _match_compiled_folder(model: str) -> str:
    """Map a registry key or slug (any case) to its compiled folder name."""
    for k, v in DATASETS_META.items():
        slug = v.get("name", "")
        compiled = _compiled_folder_name(slug)
        if model in [k, slug, compiled] or model.lower() in [
            k.lower(),
            slug.lower(),
            compiled.lower(),
        ]:
            return compiled
    return model


def _model_from_repo(repo_id: str) -> str | None:
    """Compiled folder name of the registry entry whose Kaggle ref matches ``repo_id``."""
    for k, v in DATASETS_META.items():
        k_ref = get_kaggle_ref(k)
        if k_ref and k_ref.lower() in repo_id.lower():
            return _compiled_folder_name(v.get("name", ""))
    return None


def _resolve_targets(
    action: str, model: str | None, repo_id: str | None
) -> tuple[str | None, str | None]:
    """Fill in the manifold folder and Kaggle ID from the registry or interactive prompts."""
    if action == "sync" and not model:
        model, repo_id = _pick_sync_target(repo_id)
    if action == "get" and not repo_id and not model:
        model, repo_id = _pick_get_target()

    if not repo_id:
        if model:
            repo_id = get_kaggle_ref(model)
        if not repo_id:
            repo_id = input("Enter Kaggle ID (username/slug): ").strip()

    if repo_id and repo_id.startswith("http"):
        # Quick parse of URL
        parts = repo_id.rstrip("/").split("/")
        repo_id = f"{parts[-2]}/{parts[-1]}"

    # Auto-resolve model directory name if a key, slug, or repo_id was given
    if model and not (OUT_PARENT / model).exists():
        model = _match_compiled_folder(model)
    if not model and repo_id:
        model = _model_from_repo(repo_id) or model
    return model, repo_id


def main():
    """Parse the CLI arguments and run the sync or get action."""
    args = _build_parser().parse_args()
    action = _resolve_action(args)
    model, repo_id = _resolve_targets(action, args.model, args.url)

    # Setup Auth
    default_user = repo_id.split("/")[0] if repo_id and "/" in repo_id else None
    if not setup_auth(args.user, args.token, default_user):
        sys.exit(1)

    if action == "sync":
        action_sync(model, repo_id, no_wait=args.no_wait)
    elif action == "get":
        action_get(repo_id, model)

if __name__ == "__main__":
    main()
