import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_url", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    print(f"CLONING: {args.repo_url}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Simple git clone
    subprocess.run(["git", "clone", args.repo_url, args.output_dir])

if __name__ == "__main__":
    main()
