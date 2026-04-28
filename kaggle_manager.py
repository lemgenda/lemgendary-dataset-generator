import os
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="LemGendary Kaggle Downloader")
    parser.add_argument("--repo_id", type=str, required=True, help="Kaggle Dataset/Competition ID")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--is_competition", action="store_true", help="Flag if source is a competition")
    args = parser.parse_args()

    dest = Path(args.output_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # We use the kaggle CLI internally
    print(f"[KAG] Downloading {args.repo_id} to {args.output_dir}...")
    try:
        # 2026: SOTA Strict Venv Enforcement - use sys.executable to ensure we use the local venv CLI
        import sys
        cmd = [sys.executable, "-m", "kaggle"]
        if args.is_competition:
            cmd += ["competitions", "download", "-c", args.repo_id, "-p", str(dest.parent)]
        else:
            cmd += ["datasets", "download", "-d", args.repo_id, "-p", str(dest.parent)]
        
        subprocess.run(cmd, check=True)
        
        # Verify the zip was downloaded
        slug = args.repo_id.split("/")[-1]
        zip_path = dest.parent / f"{slug}.zip"
        
        if zip_path.exists():
            print(f"[SUCCESS] {args.repo_id} zip acquired: {zip_path}")
        else:
            # Fallback check: sometimes Kaggle doesn't add the .zip extension in the filename
            potential_zips = list(dest.parent.glob(f"{slug}*"))
            if potential_zips:
                print(f"[SUCCESS] {args.repo_id} acquired as {potential_zips[0].name}")
            else:
                print(f"[ERROR] Could not find downloaded file for {args.repo_id}")
                exit(1)
                
    except Exception as e:
        print(f"[ERROR] Kaggle download failed for {args.repo_id}: {e}")
        exit(1)

if __name__ == "__main__":
    main()
