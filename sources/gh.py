"""GitHub source: clone a repository into the dataset input folder.

Run from the project root as ``python -m sources.gh``.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile

from core.secrets import get_suite_pat

logger = logging.getLogger(__name__)

_ASKPASS_SCRIPT = """#!/bin/sh
case "$1" in
    *Username*) exec printf '%s\\n' "x-access-token" ;;
    *Password*) exec printf '%s\\n' "${LEMGENDARY_GIT_TOKEN}" ;;
    *)          exec printf '\\n' ;;
esac
"""


def clone_repository(repo_url: str, output_dir: str) -> int:
    """Clone a GitHub repository into output_dir using SUITE_PAT if available."""
    print(f"CLONING: {repo_url}")
    os.makedirs(output_dir, exist_ok=True)

    token = get_suite_pat()
    git_env = os.environ.copy()
    git_env["GIT_TERMINAL_PROMPT"] = "0"

    askpass_dir = None
    if token:
        askpass_dir = tempfile.mkdtemp(prefix="lemgendary-gh-askpass-")
        askpass = os.path.join(askpass_dir, "askpass.sh")
        with open(askpass, "w", encoding="utf-8") as fh:
            fh.write(_ASKPASS_SCRIPT)
        try:
            os.chmod(askpass, stat.S_IRWXU)
        except OSError as err:
            logger.debug("Failed to set askpass executable permissions: %s", err)
        git_env["GIT_ASKPASS"] = askpass
        git_env["LEMGENDARY_GIT_TOKEN"] = token
        logger.info("[AUTH] SUITE_PAT supplied to git via GIT_ASKPASS.")

    cflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if sys.platform == "win32" else 0
    try:
        result = subprocess.run(
            ["git", "clone", repo_url, output_dir],
            creationflags=cflags,
            env=git_env,
            check=False,
        )
        if result.returncode != 0:
            logger.error("git clone failed with exit code %s", result.returncode)
        return result.returncode
    finally:
        if askpass_dir and os.path.exists(askpass_dir):
            try:
                shutil.rmtree(askpass_dir)
            except OSError as err:
                logger.debug("Failed to remove temporary askpass directory: %s", err)


def main() -> int:
    """Clone a GitHub repository into the output directory (CLI entry point)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_url", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    return clone_repository(args.repo_url, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
