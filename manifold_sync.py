import os
import sys
import shutil
import argparse
import subprocess
import json
from pathlib import Path

# Try to import from core, fallback if missing
try:
    from compiler_core import META, YAML_DATA, OUT_PARENT, DATASETS_META
except ImportError:
    OUT_PARENT = Path("../LemGendaryDatasets").resolve()
    DATASETS_META = {}
    import yaml
    if Path("unified_data.yaml").exists():
        YAML_DATA = yaml.safe_load(open("unified_data.yaml"))
        DATASETS_META = YAML_DATA.get("datasets", {})

def setup_auth(custom_user=None, custom_token=None, default_user=None):
    """Setup Kaggle authentication using environment variables."""
    # 1. Custom provided
    if custom_user and custom_token:
        os.environ["KAGGLE_USERNAME"] = custom_user
        os.environ["KAGGLE_KEY"] = custom_token
        return True
    
    # 2. Check existing env
    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        return True
        
    # 3. Fallback to .kaggle_token and default user
    token_file = Path(".kaggle_token")
    if token_file.exists() and default_user:
        with open(token_file, "r") as f:
            token = f.read().strip()
        if token:
            os.environ["KAGGLE_USERNAME"] = default_user
            os.environ["KAGGLE_KEY"] = token
            print(f"[AUTH] Using credentials for user '{default_user}' from .kaggle_token")
            return True
            
    print("[ERROR] Kaggle authentication missing. Provide via arguments or ensure .kaggle_token exists.")
    return False

def get_kaggle_ref(manifold_id):
    """Retrieve kaggle ref from yaml."""
    for k, v in DATASETS_META.items():
        if k == manifold_id or v.get("name") == manifold_id:
            ref = v.get("kaggle_ref", "")
            if ref.startswith("kaggle://"):
                return ref[9:] # user/slug
    return None

def action_sync(manifold_name, repo_id):
    """Zip and Upload a manifold to Kaggle."""
    src_dir = OUT_PARENT / manifold_name
    if not src_dir.exists():
        print(f"[ERROR] Local manifold '{manifold_name}' not found at {src_dir}")
        return
        
    print(f"\n[SYNC] Preparing to upload '{manifold_name}' to Kaggle ({repo_id})")
    
    # Staging area
    staging_dir = OUT_PARENT / f".staging_{manifold_name}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[SYNC] Zipping manifold (This may take a while)...")
    # We want the zip to contain the folder 'manifold_name' itself.
    try:
        shutil.make_archive(str(staging_dir / manifold_name), 'zip', root_dir=str(OUT_PARENT), base_dir=manifold_name)
    except Exception as e:
        print(f"[ERROR] Failed to zip manifold: {e}")
        shutil.rmtree(staging_dir)
        return
        
    print(f"[SYNC] Generating metadata stub...")
    user, slug = repo_id.split("/")
    meta_path = staging_dir / "dataset-metadata.json"
    
    # Try to copy existing metadata if it exists, otherwise generate stub
    existing_meta = src_dir / "dataset-metadata.json"
    if existing_meta.exists():
        shutil.copy2(existing_meta, meta_path)
    else:
        with open(meta_path, "w") as f:
            json.dump({
                "id": repo_id,
                "title": manifold_name,
                "isPrivate": True,
                "licenses": [{"name": "other"}]
            }, f, indent=2)
            
    print(f"[SYNC] Uploading to Kaggle via API...")
    try:
        # Check if dataset exists to determine create vs version
        check_cmd = ["kaggle", "datasets", "status", repo_id]
        res = subprocess.run(check_cmd, capture_output=True, text=True)
        
        if "404" in res.stdout or "403" in res.stdout or res.returncode != 0:
            print("[SYNC] Dataset does not exist on Kaggle. Creating new dataset...")
            push_cmd = ["kaggle", "datasets", "create", "-p", str(staging_dir)]
        else:
            print("[SYNC] Dataset exists. Pushing new version...")
            push_cmd = ["kaggle", "datasets", "version", "-p", str(staging_dir), "-m", "Manifold Sync Update"]
            
        subprocess.run(push_cmd, check=True)
        print(f"[SUCCESS] Manifold uploaded successfully! Kaggle will unzip it server-side.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Kaggle API upload failed: {e}")
    finally:
        print(f"[SYNC] Cleaning up local staging archives...")
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

def action_get(repo_id, output_name=None):
    """Download and extract a manifold from Kaggle."""
    if not output_name:
        output_name = repo_id.split("/")[-1]
        
    dest_dir = OUT_PARENT / output_name
    
    print(f"\n[GET] Requesting LemGendized Manifold: {repo_id}")
    
    # Collision check
    if dest_dir.exists():
        print(f"[WARNING] Local manifold '{output_name}' already exists at {dest_dir}!")
        ans = input("Do you want to OVERRIDE and delete the existing local manifold? (y/N): ").strip().lower()
        if ans == 'y':
            confirm = input(f"[CRITICAL] Are you ABSOLUTELY sure? Type 'YES' to delete {dest_dir}: ").strip()
            if confirm == 'YES':
                print(f"[GET] Purging existing manifold...")
                shutil.rmtree(dest_dir)
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

    print(f"[GET] Downloading from Kaggle...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Use Kaggle CLI to download and unzip
    cmd = [
        "kaggle", "datasets", "download", 
        "-d", repo_id, 
        "-p", str(dest_dir),
        "--unzip"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] Downloaded and extracted manifold to {dest_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download manifold: {e}")
        if not any(dest_dir.iterdir()):
            dest_dir.rmdir()

def main():
    parser = argparse.ArgumentParser(description="LemGendary Manifold Sync Manager")
    parser.add_argument("--action", choices=["sync", "get"], help="Action to perform")
    parser.add_argument("--model", type=str, help="Local manifold name or YAML key")
    parser.add_argument("--url", type=str, help="Kaggle Dataset URL or ID (e.g., user/dataset)")
    parser.add_argument("--user", type=str, help="Custom Kaggle Username")
    parser.add_argument("--token", type=str, help="Custom Kaggle API Token")
    
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
            
    # Resolve Model/URL
    model = args.model
    repo_id = args.url
    
    if action == "sync" and not model:
        model = input("Enter local manifold name to Sync: ").strip()
        
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
        action_sync(model, repo_id)
    elif action == "get":
        action_get(repo_id, model)

if __name__ == "__main__":
    main()
