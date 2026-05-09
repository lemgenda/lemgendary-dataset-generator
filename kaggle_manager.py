import argparse
import kagglehub
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--is_competition", action="store_true")
    parser.add_argument("--action", default="download") # For sync/upload if needed
    args = parser.parse_args()

    if args.action == "download":
        print(f"PULLING: {args.repo_id}")
        if args.is_competition:
            # Note: kagglehub handles competitions differently or we use kaggle api
            import subprocess
            subprocess.run(["kaggle", "competitions", "download", "-c", args.repo_id, "-p", args.output_dir])
        else:
            path = kagglehub.dataset_download(args.repo_id)
            # The .ps1 expects the zip or files in output_dir
            import shutil
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)
            
            # kagglehub downloads to a cache, we might need to copy it
            print(f"Downloaded to {path}")
            # If path is a file (zip), move it
            if os.path.isfile(path):
                shutil.copy2(path, os.path.join(args.output_dir, os.path.basename(path)))
            else:
                # If path is a directory, copy contents
                for item in os.listdir(path):
                    s = os.path.join(path, item)
                    d = os.path.join(args.output_dir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
    elif args.action == "upload":
        print(f"PUSHING: {args.repo_id}")
        kagglehub.model_upload(args.repo_id, args.output_dir)

if __name__ == "__main__":
    main()
