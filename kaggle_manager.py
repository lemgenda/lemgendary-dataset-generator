# LemGendary Kaggle Manager (Last Verified: 2026-05-08)
import argparse
import subprocess
import os
from pathlib import Path

def upload_dataset(local_dir, handle, version_notes):
    import kagglehub
    print(f"🚀 [KAGGLER] Synchronizing dataset to Kaggle: {handle}...")
    try:
        kagglehub.dataset_upload(
            handle=handle,
            local_dataset_dir=local_dir,
            version_notes=version_notes
        )
        print("✅ [KAGGLER] Dataset version published successfully.")
    except Exception as e:
        print(f"❌ [KAGGLER] Upload failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="LemGendary Kaggle Manager (Hybrid Sync v1.0)")
    parser.add_argument("--action", type=str, default="download", choices=["download", "upload"], help="Action to perform")
    parser.add_argument("--repo_id", type=str, required=True, help="Kaggle Dataset/Competition ID or Handle")
    parser.add_argument("--output_dir", type=str, required=True, help="Local directory for download/upload")
    parser.add_argument("--is_competition", action="store_true", help="Flag if source is a competition (download only)")
    parser.add_argument("--notes", type=str, default="SOTA Dataset Update", help="Version notes for upload")
    args = parser.parse_args()

    dest = Path(args.output_dir)
    
    if args.action == "upload":
        upload_dataset(str(dest), args.repo_id, args.notes)
        return

    # Download Logic
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    import time
    max_retries = 3
    retry_delay = 5 # seconds

    # 2026: SOTA Strict Venv Enforcement - use sys.executable to ensure we use the local venv CLI
    import sys
    cmd = [sys.executable, "-m", "kaggle"]
    if args.is_competition:
        cmd += ["competitions", "download", "-c", args.repo_id, "-p", str(dest.parent)]
    else:
        cmd += ["datasets", "download", "-d", args.repo_id, "-p", str(dest.parent)]

    for attempt in range(max_retries):
        try:
            print(f"[KAG] Attempt {attempt + 1}/{max_retries}: {args.repo_id}...")
            subprocess.run(cmd, check=True)
            
            # Verify the zip was downloaded
            slug = args.repo_id.split("/")[-1]
            zip_path = dest.parent / f"{slug}.zip"
            
            if zip_path.exists():
                print(f"[SUCCESS] {args.repo_id} zip acquired.")
                return
            else:
                potential_zips = list(dest.parent.glob(f"{slug}*"))
                if potential_zips:
                    print(f"[SUCCESS] {args.repo_id} acquired as {potential_zips[0].name}")
                    return
                else:
                    raise FileNotFoundError(f"Zip not found for {slug}")

        except Exception as e:
            print(f"[WARNING] Attempt {attempt + 1} failed for {args.repo_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt)) # Exponential backoff
            else:
                print(f"[ERROR] Final attempt failed for {args.repo_id}.")
                exit(1)

if __name__ == "__main__":
    main()
