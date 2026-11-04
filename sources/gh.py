"""GitHub source: clone a repository into the dataset input folder.

Run from the project root as ``python -m sources.gh``.
"""
import argparse
import os
import subprocess
import sys


def main():
    """Clone a GitHub repository into the output directory (CLI entry point)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_url", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    print(f"CLONING: {args.repo_url}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Simple git clone
    cflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if sys.platform == "win32" else 0
    result = subprocess.run(
        ["git", "clone", args.repo_url, args.output_dir], creationflags=cflags, check=False
    )
    if result.returncode != 0:
        print(f"ERROR: git clone failed with exit code {result.returncode}", file=sys.stderr)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
