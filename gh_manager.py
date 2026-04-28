import os
import argparse
import subprocess
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="LemGendary GitHub Downloader")
    parser.add_argument("--repo_url", type=str, required=True, help="GitHub Repo URL (user/repo)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--branch", type=str, default=None, help="Branch to clone")
    args = parser.parse_args()

    repo_url = f"https://github.com/{args.repo_url}.git"
    dest = Path(args.output_dir)
    
    if dest.exists():
        print(f"[GH] Destination {dest} already exists. Attempting pull...")
        try:
            subprocess.run(["git", "-C", str(dest), "pull"], check=True)
            print(f"[SUCCESS] {args.repo_url} updated.")
            return
        except:
            print(f"[WARN] Pull failed. Removing and re-cloning...")
            shutil.rmtree(dest)

    print(f"[GH] Cloning {args.repo_url} (depth 1)...")
    try:
        clone_cmd = ["git", "clone", "--depth", "1"]
        if args.branch:
            clone_cmd.extend(["--branch", args.branch])
        clone_cmd.extend([repo_url, str(dest)])
        
        subprocess.run(clone_cmd, check=True)
        
        # Cleanup .git to save space in the manifold root
        git_dir = dest / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir)
            print(f"[CLEANUP] Removed .git metadata for {args.repo_url}")
            
        print(f"[SUCCESS] {args.repo_url} cloned to {args.output_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to clone {args.repo_url}: {e}")
        exit(1)

if __name__ == "__main__":
    main()
