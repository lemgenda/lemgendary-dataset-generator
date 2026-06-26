import argparse
import kagglehub
import os
import shutil
from tqdm import tqdm
# 2026 Resilience: Force ASCII progress bars globally to prevent Unicode Mojibake in PowerShell
import functools

def copy_with_progress(src, dst):
    """SOTA Buffered Copy with tqdm progress tracking."""
    file_size = os.path.getsize(src)
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        with tqdm(total=file_size, unit="B", unit_scale=True, desc=f"FINALIZING: {os.path.basename(src)}") as pbar:
            while True:
                buf = fsrc.read(64*1024) # 64KB chunks
                if not buf:
                    break
                fdst.write(buf)
                pbar.update(len(buf))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--is_competition", action="store_true")
    parser.add_argument("--action", default="download") 
    args = parser.parse_args()

    if args.action == "download":
        print(f"PULLING: {args.repo_id}")
        if args.is_competition:
            import subprocess
            subprocess.run(["kaggle", "competitions", "download", "-c", args.repo_id, "-p", args.output_dir])
        else:
            path = kagglehub.dataset_download(args.repo_id)
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)
            
            # 2026 Resilience: Kagglehub downloads to a cache. 
            # We use a buffered copy with progress for massive datasets.
            print(f"Downloaded to cache. Finalizing manifold...")
            
            if os.path.isfile(path):
                target = os.path.join(args.output_dir, os.path.basename(path))
                copy_with_progress(path, target)
                # Cleanup cache after successful copy
                try: os.remove(path)
                except: pass
            else:
                # Directory copy
                items = os.listdir(path)
                for item in items:
                    s = os.path.join(path, item)
                    d = os.path.join(args.output_dir, item)
                    if os.path.isdir(s):
                        # Recursive directory copy is harder with progress, 
                        # so we use standard copytree but print status
                        print(f"Moving folder: {item}...")
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        copy_with_progress(s, d)
                
                # Cleanup cache folder
                try: shutil.rmtree(path)
                except: pass
                
    elif args.action == "upload":
        print(f"PUSHING: {args.repo_id}")
        kagglehub.model_upload(args.repo_id, args.output_dir)

if __name__ == "__main__":
    main()
