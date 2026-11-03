"""Google Drive source: download a shared file or folder with gdown.

Run from the project root as ``python -m sources.gd``.
"""
import argparse
import logging
import os

try:
    import gdown
except ImportError:
    gdown = None

logger = logging.getLogger(__name__)


def _gdown_errors() -> tuple[type[BaseException], ...]:
    """Exceptions gdown is expected to raise for a failed/unauthorised download."""
    errors: list[type[BaseException]] = [OSError, ValueError, RuntimeError]
    exc_mod = getattr(gdown, "exceptions", None)
    for name in ("FileURLRetrievalError", "FolderContentsMaximumLimitError"):
        cls = getattr(exc_mod, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            errors.append(cls)
    return tuple(errors)

# 2026 Resilience: Force ASCII progress bars globally to prevent Unicode Mojibake in PowerShell
# Removed TQDM_ASCII override to allow Unicode block characters

def main():
    """Download a Google Drive file or folder (CLI entry point)."""
    parser = argparse.ArgumentParser(description="LemGendary Google Drive Downloader")
    parser.add_argument("--repo_id", type=str, required=True, help="Google Drive File or Folder ID")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory path (will be converted to .zip in parent dir)",
    )
    parser.add_argument("--action", type=str, default="download")
    args = parser.parse_args()

    if gdown is None:
        raise RuntimeError(
            "gdown package is required for Google Drive operations. Please install via pip: pip"
            " install gdown"
        )

    if args.action == "download":
        print(f"STATUS:GD-PULLING {args.repo_id}")
        # The output_dir passed from hub is a folder like raw-sets/slug
        parent_dir = os.path.dirname(args.output_dir)
        slug = os.path.basename(args.output_dir)
        zip_path = os.path.join(parent_dir, f"{slug}.zip")
        gdown_errors = _gdown_errors()

        try:
            print(f"Attempting to download {args.repo_id} as a file archive to {zip_path}...")
            # fuzzy=True helps if the URL is provided instead of an ID
            res = gdown.download(id=args.repo_id, output=zip_path, quiet=False)
            if res is None:
                raise RuntimeError("Failed to download as a file.")
            print("STATUS:DOWNLOADED")
        except gdown_errors as e:
            logger.warning(
                "Google Drive file download failed for %s; trying folder download: %s",
                args.repo_id,
                e,
            )
            print(f"WARNING: File download failed ({e}). Attempting folder download...")
            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except OSError as exc:
                    print(f"[DEBUG] Could not remove partial zip file {zip_path}: {exc}")

            try:
                res = gdown.download_folder(id=args.repo_id, output=args.output_dir, quiet=False)
                if res is None:
                    raise RuntimeError("Failed to download as a folder.") from e
                print("STATUS:COMPLETED")
            except gdown_errors as e2:
                logger.error("Could not download Google Drive source %s: %s", args.repo_id, e2)
                print(f"ERROR: Could not download Google Drive source {args.repo_id}: {e2}")
                print("STATUS:FAILED")

if __name__ == "__main__":
    main()
