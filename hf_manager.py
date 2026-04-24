import os
import argparse
from huggingface_hub import snapshot_download, login
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="LemGendary HF Downloader")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Repo ID")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--repo_type", type=str, default="model", help="Repository type: model, dataset, space")
    parser.add_argument("--token_file", type=str, default=".huggingface_token", help="Path to token file")
    args = parser.parse_args()

    # Auth
    token = None
    if os.path.exists(args.token_file):
        with open(args.token_file, "r") as f:
            token = f.read().strip()
            if token:
                login(token=token)
                print(f"[AUTH] Logged into Hugging Face via {args.token_file}")

    # Download
    print(f"[HF] Starting {args.repo_type} download for {args.repo_id}...")
    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            local_dir=args.output_dir,
            token=token
        )
        print(f"[SUCCESS] {args.repo_id} downloaded to {args.output_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to download {args.repo_id}: {e}")
        exit(1)

if __name__ == "__main__":
    main()
