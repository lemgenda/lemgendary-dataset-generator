"""
Remote Kaggle dataset queries and placeholder creation for the sync core.

Reads a dataset's version list and server-side status (Kaggle SDK first,
Kaggle CLI as fallback) and creates the README-only placeholder that must
exist before metadata updates or uploads can address a new slug. The
placeholder is always private and carries the manifold's own license; the
metadata push that follows applies the registry's license and visibility
(``is_private``), so nothing is ever public before its metadata says so.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict

from kagglehub.clients import build_kaggle_client
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

from core.kaggle_errors import KAGGLE_CLIENT_ERRORS
from core.lazy_imports import load_kaggle_api
from core.sync_auth import (
    is_kaggle_large_manifold,
    kaggle_cli_creationflags,
    setup_kaggle_auth,
    split_owner_slug,
)
from core.sync_metadata_settings import FALLBACK_LICENSE

logger = logging.getLogger(__name__)


class DatasetVersionInfo(TypedDict):
    """Current and latest version numbers of a Kaggle dataset, with each version's status."""
    current_version: int
    latest_version: int
    versions: dict[int, str]


def get_dataset_version_info(repo_id: str) -> DatasetVersionInfo:
    """Retrieve current version number, latest version number, and per-version status."""
    clean_handle = repo_id.replace("kaggle://", "")
    owner, slug = split_owner_slug(clean_handle)
    setup_kaggle_auth(default_user=owner)

    try:
        with build_kaggle_client() as client:
            req = ApiGetDatasetRequest()
            req.owner_slug = owner
            req.dataset_slug = slug
            resp = client.datasets.dataset_api_client.get_dataset(req)
            curr = resp.current_version_number or 0
            valid_versions = [
                v for v in (resp.versions or [])
                if v is not None and v.version_number is not None
            ]
            versions_dict: dict[int, str] = {
                v.version_number: v.status or ""
                for v in valid_versions
                if v.version_number is not None
            }
            return {
                "current_version": curr,
                "latest_version": max(
                    (v.version_number for v in valid_versions), default=curr
                ),
                "versions": versions_dict,
            }
    except KAGGLE_CLIENT_ERRORS as exc:
        logger.warning("Failed retrieving dataset version info for %s: %s", repo_id, exc)
        return {
            "current_version": 0,
            "latest_version": 0,
            "versions": {},
        }


def _classify_cli_status(out: str, returncode: int) -> str | None:
    """Map ``kaggle datasets status`` output to ready / creating / error / raw last word."""
    if "ready" in out:
        return "ready"
    if "creating" in out or "pending" in out or "queued" in out:
        return "creating"
    if "error" in out or "failed" in out:
        return "error"
    if returncode == 0 and out:
        return out.split()[-1]
    return None


def get_dataset_status(repo_id: str) -> str | None:
    """Query current server-side status of a dataset."""
    clean_handle = repo_id.replace("kaggle://", "")
    owner, _ = split_owner_slug(clean_handle)
    setup_kaggle_auth(default_user=owner)

    try:
        api = load_kaggle_api()()
        api.authenticate()
        return api.dataset_status(clean_handle)
    except KAGGLE_CLIENT_ERRORS as exc:
        logger.debug("Kaggle API dataset_status failed, falling back to the Kaggle CLI: %s", exc)

    try:
        res = subprocess.run(
            ["kaggle", "datasets", "status", clean_handle],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            creationflags=kaggle_cli_creationflags(),
        )
        return _classify_cli_status(res.stdout.strip().lower(), res.returncode)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Kaggle CLI status check failed: %s", exc)

    return None


def _write_placeholder_payload(
    tmp_dir: Path, clean_repo_id: str, title: str, readme_text: str, license_name: str
) -> None:
    """Write the README.md and dataset-metadata.json that dataset_create_new requires."""
    owner, slug = split_owner_slug(clean_repo_id)
    placeholder_readme = readme_text or (
        f"# {title}\n\n"
        "This dataset is currently being compiled and will be populated shortly.\n\n"
        "The full dataset will replace this placeholder upon upload completion."
    )
    (tmp_dir / "README.md").write_text(placeholder_readme, encoding="utf-8")

    safe_title = title[:50]
    if len(safe_title) < 6:
        safe_title = safe_title.ljust(6, " ")
    meta = {
        "id": f"{owner}/{slug}",
        "title": safe_title,
        "licenses": [{"name": license_name}],
        "isPrivate": True,
    }
    (tmp_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _create_placeholder_dataset(clean_repo_id: str, tmp_dir: Path) -> bool:
    """Push the placeholder folder as a new private dataset; "already in use" counts as success."""
    try:
        api = load_kaggle_api()()
        api.authenticate()
        resp = api.dataset_create_new(
            folder=str(tmp_dir),
            public=False,
            quiet=False,
            convert_to_csv=False,
            dir_mode="skip",
        )
    except KAGGLE_CLIENT_ERRORS as exc:
        err_str = str(exc)
        if "already in use" in err_str or "409" in err_str:
            print(f"[INIT] Dataset '{clean_repo_id}' already exists on Kaggle. Proceeding.")
            return True
        print(f"[ERROR] Failed to create Kaggle dataset placeholder: {exc}")
        return False

    if resp and getattr(resp, "status", None) == "error":
        err_msg = getattr(resp, "error", str(resp))
        if "already in use" in str(err_msg):
            print(
                f"[INIT] Dataset '{clean_repo_id}' already exists on Kaggle (concurrent"
                " creation). Proceeding."
            )
            return True
        print(f"[ERROR] Failed to create Kaggle dataset '{clean_repo_id}': {err_msg}")
        return False
    print(
        f"[OK] [INIT] Kaggle dataset '{clean_repo_id}' created successfully. Ready for data"
        " upload."
    )
    return True


def ensure_kaggle_dataset_exists(
    clean_repo_id: str,
    title: str,
    readme_text: str = "",
    license_name: str = FALLBACK_LICENSE,
) -> bool:
    """Create the Kaggle dataset with a README placeholder if it does not yet exist.

    This prevents the 403 Forbidden error when calling UpdateDatasetMetadata on a
    dataset that has never been created. A minimal private dataset containing only
    README.md is pushed first, making the slug addressable before the actual upload;
    the metadata push then sets the registry license and visibility.
    """
    if is_kaggle_large_manifold(clean_repo_id):
        print(
            "[SAFEGUARD] Refusing to create placeholder for legacy *large manifold:"
            f" {clean_repo_id}"
        )
        return False

    version_info = get_dataset_version_info(clean_repo_id)
    if version_info["latest_version"] > 0:
        return True  # Dataset already exists

    print(
        f"[INIT] Dataset '{clean_repo_id}' not found on Kaggle. Creating empty private"
        " placeholder..."
    )

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        _write_placeholder_payload(tmp_dir, clean_repo_id, title, readme_text, license_name)
        return _create_placeholder_dataset(clean_repo_id, tmp_dir)
