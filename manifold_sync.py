import os
import sys
import shutil
import argparse
import subprocess
import json
from pathlib import Path

# Load dataset registry directly from YAML
OUT_PARENT = Path("../LemGendaryDatasets").resolve()
DATASETS_META = {}
YAML_DATA = {}
META = {}
yaml_file = Path("unified_data.yaml")
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

    from kaggle_manager import get_dataset_version_info, track_kaggle_dataset_status, cleanup_temp_archives

    # Pre-cleanup any dangling staging directories
    cleanup_temp_archives(manifold_name=manifold_name, base_dir=OUT_PARENT)

    # Detect version before upload to identify target version
    version_info = get_dataset_version_info(clean_repo_id)
    target_version = (version_info.get("latest_version") or 0) + 1

    print(f"\n[SYNC] Preparing to upload '{manifold_name}' to Kaggle ({clean_repo_id})")
    
    file_count = 0
    for _, _, files in os.walk(src_dir):
        file_count += len(files)

    staging_dir = None
    if file_count > 50:
        print(f"[SYNC] Detected {file_count} files (>50 threshold). Archiving manifold with progress tracking...")
        staging_dir = OUT_PARENT / f".staging_{manifold_name}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

        target_zip = staging_dir / f"{manifold_name}.zip"
        try:
            from archive_manager import create_archive
            success = create_archive(src_dir, target_zip, format="zip")
            if not success or not target_zip.exists():
                raise RuntimeError(f"Archive creation failed for {src_dir}")

            zip_size_gb = target_zip.stat().st_size / (1024**3)
            print(f"[SYNC] Archive created: {target_zip.name} ({zip_size_gb:.2f} GB)")
            print(f"[SYNC] Uploading archive to Kaggle via KaggleHub API...")
            import kagglehub
            kagglehub.dataset_upload(clean_repo_id, str(staging_dir))
        except Exception as e:
            print(f"[ERROR] Upload failed: {e}")
            return
        finally:
            if target_zip.exists():
                try:
                    target_zip.unlink()
                except OSError:
                    pass
            if staging_dir and staging_dir.exists():
                print(f"[SYNC] Cleaning up local staging archive...")
                try:
                    shutil.rmtree(staging_dir, ignore_errors=True)
                except OSError as ex:
                    print(f"[WARN] Could not remove staging directory: {ex}")
            cleanup_temp_archives(manifold_name=manifold_name, base_dir=OUT_PARENT)
    else:
        print(f"[SYNC] Uploading {file_count} files directly to Kaggle via KaggleHub API...")
        import kagglehub
        try:
            kagglehub.dataset_upload(clean_repo_id, str(src_dir))
        except Exception as e:
            print(f"[ERROR] Upload failed: {e}")
            return

    print(f"[SUCCESS] Manifold uploaded successfully! Kaggle will process it server-side.")

    if not no_wait:
        success = track_kaggle_dataset_status(
            clean_repo_id,
            target_version=target_version,
            expected_files=file_count
        )
        if not success:
            sys.exit(1)

def action_get(repo_id, output_name=None):
    """Download and extract a manifold from Kaggle with byte-level progress."""
    clean_repo_id = repo_id.replace("kaggle://", "")
    if not output_name:
        output_name = clean_repo_id.split("/")[-1]
        
    dest_dir = OUT_PARENT / output_name
    
    print(f"\n[GET] Requesting LemGendized Manifold: {clean_repo_id}")
    
    # Collision check
    if dest_dir.exists():
        print(f"[WARNING] Local manifold '{output_name}' already exists at {dest_dir}!")
        ans = input("Do you want to OVERRIDE and delete the existing local manifold? (y/N): ").strip().lower()
        if ans == 'y':
            confirm = input(f"[CRITICAL] Are you ABSOLUTELY sure? Type 'YES' to delete {dest_dir}: ").strip()
            if confirm == 'YES':
                print(f"[GET] Purging existing manifold...")
                shutil.rmtree(dest_dir, ignore_errors=True)
            else:
                print("[GET] Override cancelled. Aborting.")
                return
        else:
            print("[GET] Aborting.")
            return

    # Disk space check (Require at least 20GB free as a baseline safeguard)
    total, used, free = shutil.disk_usage(OUT_PARENT)
    free_gb = free / (1024**3)
    if free_gb < 20.0:
        print(f"[WARNING] Low disk space! Only {free_gb:.2f} GB free on {OUT_PARENT.drive}.")
        ans = input("Continue anyway? (y/N): ").strip().lower()
        if ans != 'y':
            print("[GET] Aborting.")
            return
    else:
        print(f"[GET] Disk check passed ({free_gb:.2f} GB free).")

    print(f"[GET] Downloading from Kaggle via KaggleHub API...")
    import kagglehub
    from kaggle_manager import copy_with_progress, copy_tree_with_progress
    from archive_manager import smart_extract

    owner = clean_repo_id.split("/")[0] if "/" in clean_repo_id else "lemtreursi"
    setup_auth(default_user=owner)

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = kagglehub.dataset_download(clean_repo_id)
        print("Downloaded to cache. Finalizing manifold...")
        if os.path.isfile(path):
            target = dest_dir / os.path.basename(path)
            copy_with_progress(path, target)
            try:
                os.remove(path)
            except OSError:
                pass
        else:
            for item in os.listdir(path):
                s = os.path.join(path, item)
                d = dest_dir / item
                if os.path.isdir(s):
                    copy_tree_with_progress(s, d)
                else:
                    copy_with_progress(s, d)
            try:
                shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass

        # Clean up kagglehub download cache to reclaim disk space
        hub_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / clean_repo_id.replace("/", os.sep)
        if hub_cache.exists():
            shutil.rmtree(hub_cache, ignore_errors=True)

        zip_files = list(dest_dir.glob("*.zip"))
        for zf in zip_files:
            print(f"[EXTRACT] Unpacking manifold archive: {zf.name}")
            smart_extract(zf, dest_dir, delete_after=True)

        print(f"[SUCCESS] Manifold extracted to {dest_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to download manifold: {e}")
        if not any(dest_dir.iterdir()):
            try:
                dest_dir.rmdir()
            except OSError:
                pass

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
