"""Publish a containerized manifold to Kaggle as a new dataset version."""

from __future__ import annotations

import json
import logging

import kagglehub
import kagglehub.exceptions
import kagglehub.gcs_upload

from core.common_sync import (
    get_dataset_version_info,
    push_kaggle_dataset_metadata,
    setup_kaggle_auth,
    track_kaggle_dataset_status,
)
from tools.modernize.queue_items import ManifoldQueueItem

logger = logging.getLogger(__name__)


def push_manifold_to_kaggle(
    item: ManifoldQueueItem,
    version_notes: str | None = None,
    no_wait: bool = False,
) -> bool:
    """Upload containerized manifold to Kaggle as a new version with synchronized metadata."""
    if not item.kaggle_ref:
        print(f"[SKIP] No kaggle_ref configured for {item.name}.")
        return False

    clean_handle = item.kaggle_ref.replace("kaggle://", "").strip()
    if "/" not in clean_handle:
        clean_handle = f"lemtreursi/{clean_handle}"

    target_dir = item.target_dir
    if not target_dir or not target_dir.exists():
        print(f"[ERROR] Target directory {target_dir} does not exist.")
        return False

    setup_kaggle_auth()
    v_info = get_dataset_version_info(clean_handle)
    current_ver = v_info.get("latest_version") or 0
    next_ver = current_ver + 1
    notes = (
        version_notes
        or f"v{next_ver}: Modernized {item.canonical_format} format with WebP transcoding"
    )

    print(
        f"\n[KAGGLE-PUSH] Uploading {item.name} to {clean_handle} (target version: v{next_ver})..."
    )
    print(f"[KAGGLE-PUSH] Payload directory: {target_dir}")

    # Synchronize dataset-metadata.json id field with clean_handle
    meta_path = target_dir / "dataset-metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f_meta:
                m_data = json.load(f_meta)
            m_data["id"] = clean_handle
            m_data["title"] = item.title
            m_data["subtitle"] = item.subtitle
            with open(meta_path, "w", encoding="utf-8") as f_meta:
                json.dump(m_data, f_meta, indent=2)
            print(f"[METADATA] Synchronized {meta_path.name} handle to {clean_handle}")
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Could not update %s id: %s", meta_path, exc)
            print(f"[WARN] Could not update dataset-metadata.json id: {exc}")

    # kagglehub annotates MAX_FILES_TO_UPLOAD as Literal[50]; setattr bypasses the
    # static type constraint while still performing the runtime assignment correctly.
    kagglehub.gcs_upload.MAX_FILES_TO_UPLOAD = 5000

    try:
        kagglehub.dataset_upload(clean_handle, str(target_dir), version_notes=notes)
        print(f"[SUCCESS] Upload payload transferred to Kaggle for {clean_handle}.")
    except (
        OSError,
        ValueError,
        RuntimeError,
        kagglehub.exceptions.BackendError,
        kagglehub.exceptions.CredentialError,
        kagglehub.exceptions.UnauthenticatedError,
        kagglehub.exceptions.KaggleEnvironmentError,
        kagglehub.exceptions.NotFoundError,
    ) as exc:
        logger.error("Kaggle upload failed for %s: %s", clean_handle, exc)
        print(f"[ERROR] Kaggle upload failed for {clean_handle}: {exc}")
        return False

    # Push metadata & column descriptors
    if meta_path.exists():
        try:
            print("[METADATA] Pushing dataset settings and licensing to Kaggle API...")
            push_kaggle_dataset_metadata(clean_handle, meta_path)
            print(f"[SUCCESS] Kaggle metadata successfully synchronized for {clean_handle}.")
        except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
            logger.warning("Kaggle metadata push failed for %s: %s", clean_handle, exc)
            print(f"[WARN] Kaggle metadata push notice: {exc}")

    if not no_wait:
        print(f"[TRACK] Waiting for Kaggle processing completion for v{next_ver}...")
        ok = track_kaggle_dataset_status(clean_handle, target_version=next_ver, timeout=3600)
        return ok

    return True
