"""
LemGendary Kaggle Dataset Manager CLI
=====================================
Unified interface for uploading, downloading, extracting, and monitoring Kaggle manifolds.

Run from the project root as ``python -m sources.kaggle``. Running it as a
module keeps ``sources/`` off ``sys.path``, so this file cannot shadow the
PyPI ``kaggle`` package.
"""
import argparse
import logging
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from core.common_sync import (
    DownloadRequest,
    perform_dataset_download,
    perform_dataset_upload,
    setup_kaggle_auth,
    track_kaggle_dataset_status,
)
from core.kaggle_identity import default_owner
from core.source_manifolds import SourceManifold, find_source_manifold
from tools.sync_kaggle_metadata import sync_metadata

logger = logging.getLogger(__name__)
logging.getLogger("kagglehub").setLevel(logging.ERROR)


def _source_manifold(clean_repo_id: str) -> SourceManifold | None:
    """The registry entry that packs ``clean_repo_id`` as its source, else None."""
    try:
        return find_source_manifold(clean_repo_id)
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Source manifold lookup for %s failed: %s", clean_repo_id, exc)
        return None


def _registry_manifold_name(slug: str) -> str | None:
    """Modernized folder of the registry entry whose ``kaggle_ref`` contains ``slug``."""
    try:
        yd_path = Path(__file__).parent.parent / "unified_data.yaml"
        if not yd_path.exists():
            yd_path = Path(__file__).parent / "unified_data.yaml"
        if not yd_path.exists():
            return None
        with open(yd_path, "r", encoding="utf-8") as yf:
            ydata = yaml.safe_load(yf) or {}
        prefix = ydata.get("_registry_metadata", {}).get("name_prefix", "LemGendized")
        suffix = ydata.get("_registry_metadata", {}).get("name_suffix", "")
        for entry in ydata.get("datasets", {}).values():
            if slug.lower() in entry.get("kaggle_ref", "").lower():
                return entry.get("modernized_folder") or f"{prefix}{entry.get('name', '')}{suffix}"
    except (OSError, UnicodeDecodeError, yaml.YAMLError, AttributeError) as exc:
        logger.debug("Kaggle manifold name lookup fallback: %s", exc)
        print(f"[DEBUG] Kaggle manifold name lookup fallback: {exc}")
    return None


def _resolve_manifold_paths(
    clean_repo_id: str, output_dir: str, source: SourceManifold | None = None
) -> tuple[Path, Path, str, str]:
    """Resolve target manifold directory and root datasets directory.

    A registered source manifold (``source``) lands in its entry's modernized folder.
    """
    target_dir = Path(output_dir).resolve()
    slug = clean_repo_id.split("/")[-1]
    if target_dir.name.startswith("LemGendized"):
        return target_dir, target_dir.parent, target_dir.name, slug
    if source is not None:
        manifold_name = source.modernized_folder
    else:
        manifold_name = _registry_manifold_name(slug) or slug
    return target_dir / manifold_name, target_dir, manifold_name, slug


@dataclass(frozen=True)
class KaggleRequest:
    """One Kaggle operation: the action, its dataset handle and target folder."""

    action: str
    repo_id: str
    output_dir: str = ""
    is_competition: bool = False
    all_datasets: bool = False
    no_wait: bool = False


def _download(req: KaggleRequest, clean_repo_id: str) -> bool:
    print(f"PULLING: {clean_repo_id}")
    source = _source_manifold(clean_repo_id)
    target_dir, root_datasets_dir, manifold_name, slug = _resolve_manifold_paths(
        clean_repo_id, req.output_dir, source
    )
    return perform_dataset_download(
        DownloadRequest(
            clean_repo_id=clean_repo_id,
            target_dir=target_dir,
            root_datasets_dir=root_datasets_dir,
            manifold_name=manifold_name,
            slug=slug,
            is_competition=req.is_competition,
            label_json=source.label_json if source else None,
        )
    )


def _upload(req: KaggleRequest, clean_repo_id: str) -> bool:
    print(f"PUSHING: {clean_repo_id}")
    return perform_dataset_upload(
        src_path=Path(req.output_dir).resolve(), clean_repo_id=clean_repo_id, no_wait=req.no_wait
    )


def run_kaggle_action(req: KaggleRequest) -> int:
    """Authenticate and run one Kaggle download, upload, status check or metadata push.

    Returns the exit code (0 on success). ``download`` and ``upload`` need
    ``output_dir``; a missing one raises ``ValueError``.
    """
    if req.action in ("download", "upload") and not req.output_dir:
        raise ValueError(f"output_dir is required when action is '{req.action}'.")
    socket.setdefaulttimeout(60.0)
    clean_repo_id = req.repo_id.replace("kaggle://", "")
    owner = clean_repo_id.split("/")[0] if "/" in clean_repo_id else default_owner()
    setup_kaggle_auth(default_user=owner)

    if req.action == "download":
        return 0 if _download(req, clean_repo_id) else 1
    if req.action == "upload":
        return 0 if _upload(req, clean_repo_id) else 1
    if req.action == "status":
        return 0 if track_kaggle_dataset_status(clean_repo_id) else 1
    if req.action == "metadata":
        return sync_metadata(
            repo_id=None if req.all_datasets else clean_repo_id,
            metadata_dir=req.output_dir or None,
            all_datasets=req.all_datasets,
        )
    raise ValueError(f"Unknown Kaggle action '{req.action}'.")


def main():
    """Main CLI entrypoint for Kaggle dataset operations."""
    parser = argparse.ArgumentParser(description="LemGendary Kaggle Manager")
    parser.add_argument("--repo_id", required=True, help="Kaggle dataset handle or URL")
    parser.add_argument("--output_dir", default="", help="Target local directory")
    parser.add_argument("--is_competition", action="store_true", help="Download from competition")
    parser.add_argument(
        "--action",
        default="download",
        choices=["download", "upload", "status", "metadata"],
        help="Action to perform",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Update metadata for all datasets in unified_data.yaml (metadata action only)",
    )
    parser.add_argument(
        "--no-wait", action="store_true", help="Skip monitoring server-side extraction after upload"
    )
    args = parser.parse_args()

    if args.action in ["download", "upload"] and not args.output_dir:
        parser.error(f"--output_dir is required when --action is '{args.action}'.")

    code = run_kaggle_action(KaggleRequest(
        action=args.action,
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        is_competition=args.is_competition,
        all_datasets=args.all_datasets,
        no_wait=args.no_wait,
    ))
    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
