import argparse
import os
import shutil

try:
    import gdown
except ImportError:
    gdown = None

# 2026 Resilience: Force ASCII progress bars globally to prevent Unicode Mojibake in PowerShell
# Removed TQDM_ASCII override to allow Unicode block characters

def main():
    parser = argparse.ArgumentParser(description="LemGendary Google Drive Downloader")
    parser.add_argument("--repo_id", type=str, required=True, help="Google Drive File or Folder ID")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory path (will be converted to .zip in parent dir)")
    parser.add_argument("--action", type=str, default="download")
    args = parser.parse_args()

    if gdown is None:
        raise RuntimeError("gdown package is required for Google Drive operations. Please install via pip: pip install gdown")

    if args.action == "download":
        print(f"STATUS:GD-PULLING {args.repo_id}")
        # The output_dir passed from hub is a folder like raw-sets/slug
        parent_dir = os.path.dirname(args.output_dir)
        slug = os.path.basename(args.output_dir)
        zip_path = os.path.join(parent_dir, f"{slug}.zip")
        
        try:
            print(f"Attempting to download {args.repo_id} as a file archive to {zip_path}...")
            # fuzzy=True helps if the URL is provided instead of an ID
            res = gdown.download(id=args.repo_id, output=zip_path, quiet=False)
            if res is None:
                raise Exception("Failed to download as a file.")
            print("STATUS:DOWNLOADED")
        except Exception as e:
            print(f"WARNING: File download failed ({e}). Attempting folder download...")
            if os.path.exists(zip_path):
                try: os.remove(zip_path)
                except: pass
                
            try:
                res = gdown.download_folder(id=args.repo_id, output=args.output_dir, quiet=False)
                if res is None:
                    raise Exception("Failed to download as a folder.")
                print("STATUS:COMPLETED")
            except Exception as e2:
                print(f"ERROR: Could not download Google Drive source {args.repo_id}: {e2}")
                print("STATUS:FAILED")

if __name__ == "__main__":
    main()
