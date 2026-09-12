import os
import sys
import shutil
import argparse
import subprocess
import json
from pathlib import Path

# Load dataset registry directly from YAML
OUT_PARENT = (Path(__file__).parent.parent / "LemGendaryDatasets").resolve()
DATASETS_META = {}
YAML_DATA = {}
META = {}
yaml_file = Path(__file__).parent / "unified_data.yaml"
if yaml_file.exists():
    import yaml
    with open(yaml_file, "r", encoding="utf-8") as f:
        YAML_DATA = yaml.safe_load(f) or {}
    META = YAML_DATA.get("_registry_metadata", {})
    DATASETS_META = YAML_DATA.get("datasets", {})
    if META.get("output_folder_name"):
        OUT_PARENT = Path(META["output_folder_name"]).resolve()

def setup_auth(custom_user=None, custom_token=None, default_user=None):
    """Setup Kaggle authentication using environment variables."""
    # 1. Custom provided
    if custom_user and custom_token:
        os.environ["KAGGLE_USERNAME"] = custom_user
        os.environ["KAGGLE_API_TOKEN"] = custom_token
        os.environ["KAGGLE_KEY"] = custom_token
        return True
    
    # 2. Check existing env
    if "KAGGLE_API_TOKEN" in os.environ:
        if default_user and "KAGGLE_USERNAME" not in os.environ:
            os.environ["KAGGLE_USERNAME"] = default_user
        return True
    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        return True
        
    # 3. Fallback to .kaggle_token and default user
    token_file = Path(".kaggle_token")
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token
            if default_user and "KAGGLE_USERNAME" not in os.environ:
                os.environ["KAGGLE_USERNAME"] = default_user
            print(f"[AUTH] Using credentials from .kaggle_token (user: '{os.environ.get('KAGGLE_USERNAME', default_user)}')")
            return True
            
    print("[ERROR] Kaggle authentication missing. Provide via arguments or ensure .kaggle_token exists.")
    return False

def get_kaggle_ref(manifold_id):
    """Retrieve kaggle ref from yaml with multi-variant matching."""
    if not manifold_id:
        return None
    target_clean = str(manifold_id).strip()
    norm_target = target_clean.lower().replace("_", "").replace("-", "").replace("lemgendized", "").replace("large", "")
    prefix = META.get("name_prefix", "LemGendized")
    suffix = META.get("name_suffix", "Large")

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
        norm_k = k.lower().replace("_", "").replace("-", "").replace("lemgendized", "").replace("large", "")
        norm_slug = slug_name.lower().replace("_", "").replace("-", "").replace("lemgendized", "").replace("large", "")
        if norm_target in [norm_k, norm_slug]:
            return clean_ref

    return None

def action_sync(manifold_name, repo_id, no_wait=False):
    """Zip and Upload a manifold to Kaggle with byte-metered progress and status monitoring."""
    src_dir = OUT_PARENT / manifold_name
    if not src_dir.exists():
        print(f"[ERROR] Local manifold '{manifold_name}' not found at {src_dir}")
        return

    clean_repo_id = repo_id.replace("kaggle://", "")
    owner = clean_repo_id.split("/")[0] if "/" in clean_repo_id else "lemtreursi"
    setup_auth(default_user=owner)

    from common_sync import cleanup_temp_archives, perform_dataset_upload

    # Pre-cleanup any dangling staging directories (preserves recent archives for resumption)
    cleanup_temp_archives(manifold_name=manifold_name, base_dir=OUT_PARENT, force=False)

    print(f"\n[SYNC] Preparing to upload '{manifold_name}' to Kaggle ({clean_repo_id})")
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
        prefix = META.get("name_prefix", "LemGendized")
        suffix = META.get("name_suffix", "Large")
        for _, v in DATASETS_META.items():
            ref = v.get("kaggle_ref", "")
            if clean_repo_id.lower() in ref.lower():
                output_name = f"{prefix}{v.get('name', '')}{suffix}"
                break

    dest_dir = OUT_PARENT / output_name

    print(f"\n[GET] Requesting LemGendized Manifold: {clean_repo_id}")

    # Collision check if directory already exists
    if dest_dir.exists():
        print(f"[WARNING] Local manifold '{output_name}' already exists at {dest_dir}!")
        ans = input("Do you want to RESUME extraction or overwrite? (R/o/n): ").strip().lower()
        if ans == 'o':
            confirm = input(f"[CRITICAL] Are you ABSOLUTELY sure? Type 'YES' to delete {dest_dir}: ").strip()
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

    from common_sync import perform_dataset_download
    success = perform_dataset_download(
        clean_repo_id=clean_repo_id,
        target_dir=dest_dir,
        root_datasets_dir=OUT_PARENT,
        manifold_name=output_name,
        slug=slug,
        is_competition=False
    )
    if not success:
        sys.exit(1)



def main():
    parser = argparse.ArgumentParser(description="LemGendary Manifold Sync Manager")
    parser.add_argument("--action", choices=["sync", "get"], help="Action to perform")
    parser.add_argument("--model", type=str, help="Local manifold name or YAML key")
    parser.add_argument("--url", type=str, help="Kaggle Dataset URL or ID (e.g., user/dataset)")
    parser.add_argument("--user", type=str, help="Custom Kaggle Username")
    parser.add_argument("--token", type=str, help="Custom Kaggle API Token")
    parser.add_argument("--no-wait", action="store_true", help="Skip monitoring Kaggle server-side extraction status")
    
    args = parser.parse_args()
    
    # Interactive menu if no action provided
    action = args.action
    if not action:
        print("\n=== LemGendary Sync Manager ===")
        print("1. [SYNC] Zip & Upload local manifold to Kaggle")
        print("2. [GET]  Download & Extract manifold from Kaggle")
        choice = input("Select operation (1/2): ").strip()
        if choice == '1':
            action = "sync"
        elif choice == '2':
            action = "get"
        else:
            print("Invalid choice.")
            sys.exit(1)
            
    prefix = META.get("name_prefix", "LemGendized")
    suffix = META.get("name_suffix", "Large")
    dataset_keys = list(DATASETS_META.keys())

    # Resolve Model/URL
    model = args.model
    repo_id = args.url
    
    if action == "sync" and not model:
        print("\n--- AVAILABLE MANIFOLDS TO SYNC ---")
        for i, k in enumerate(dataset_keys, 1):
            slug = DATASETS_META[k].get("name", "")
            manifold_folder = f"{prefix}{slug}{suffix}"
            folder_path = OUT_PARENT / manifold_folder
            status_tag = "[COMPILED]" if folder_path.exists() else "[NOT COMPILED]"
            print(f" {i:2d}. {k:<35} ({manifold_folder}) {status_tag}")
        
        sel = input("\nEnter selection (number, dataset key, or folder name): ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(dataset_keys):
            chosen_key = dataset_keys[int(sel) - 1]
            chosen_slug = DATASETS_META[chosen_key].get("name", "")
            model = f"{prefix}{chosen_slug}{suffix}"
            repo_id = get_kaggle_ref(chosen_key)
        else:
            model = sel

    if action == "get" and not repo_id and not model:
        print("\n--- AVAILABLE MANIFOLDS ON KAGGLE ---")
        for i, k in enumerate(dataset_keys, 1):
            k_ref = get_kaggle_ref(k) or "N/A"
            print(f" {i:2d}. {k:<35} -> {k_ref}")
        
        sel = input("\nEnter selection (number, dataset key, or Kaggle ID): ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(dataset_keys):
            chosen_key = dataset_keys[int(sel) - 1]
            chosen_slug = DATASETS_META[chosen_key].get("name", "")
            model = f"{prefix}{chosen_slug}{suffix}"
            repo_id = get_kaggle_ref(chosen_key)
        elif "/" in sel:
            repo_id = sel
        else:
            model = sel
            repo_id = get_kaggle_ref(model)
        
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
        for k, v in DATASETS_META.items():
            slug = v.get("name", "")
            compiled = f"{prefix}{slug}{suffix}"
            if model in [k, slug, compiled] or model.lower() in [k.lower(), slug.lower(), compiled.lower()]:
                model = compiled
                break

    if not model and repo_id:
        for k, v in DATASETS_META.items():
            k_ref = get_kaggle_ref(k)
            if k_ref and k_ref.lower() in repo_id.lower():
                slug = v.get("name", "")
                model = f"{prefix}{slug}{suffix}"
                break

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
