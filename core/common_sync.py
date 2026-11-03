"""
LemGendary Dataset Synchronization & Cloud Transfer Core
========================================================
Public facade for Kaggle authentication, archive staging, status tracking,
and resilient transfer utilities shared across dataset managers.

The implementation lives in:
  - core.sync_auth              credentials, CA bundle, handle helpers,
                                the *large manifold safeguard
  - core.sync_transfer          byte-metered copies, staging cleanup
  - core.sync_remote            version/status queries, placeholder creation
  - core.sync_tracking          server-side extraction monitor
  - core.sync_metadata          Dataset Settings API push with repair retries
  - core.sync_metadata_settings dataset-metadata.json -> DatasetSettings
  - core.sync_upload            stage, upload, metadata, wait
  - core.sync_download          resumable download, stream or extract

Patch implementation modules (e.g. ``core.sync_tracking.get_dataset_version_info``)
in tests; rebinding a name on this facade does not affect the implementation.
"""

from core.sync_auth import (
    DEFAULT_KAGGLE_OWNER,
    is_kaggle_large_manifold,
    normalize_kaggle_handle,
    setup_kaggle_auth,
    split_owner_slug,
)
from core.sync_download import DownloadRequest, perform_dataset_download
from core.sync_metadata import push_kaggle_dataset_metadata
from core.sync_remote import (
    DatasetVersionInfo,
    ensure_kaggle_dataset_exists,
    get_dataset_status,
    get_dataset_version_info,
)
from core.sync_tracking import track_kaggle_dataset_status
from core.sync_transfer import (
    CHUNK_SIZE,
    cleanup_temp_archives,
    copy_tree_with_progress,
    copy_with_progress,
    robust_staging_cleanup,
)
from core.sync_upload import perform_dataset_upload

__all__ = [
    "CHUNK_SIZE",
    "DEFAULT_KAGGLE_OWNER",
    "DatasetVersionInfo",
    "DownloadRequest",
    "cleanup_temp_archives",
    "copy_tree_with_progress",
    "copy_with_progress",
    "ensure_kaggle_dataset_exists",
    "get_dataset_status",
    "get_dataset_version_info",
    "is_kaggle_large_manifold",
    "normalize_kaggle_handle",
    "perform_dataset_download",
    "perform_dataset_upload",
    "push_kaggle_dataset_metadata",
    "robust_staging_cleanup",
    "setup_kaggle_auth",
    "split_owner_slug",
    "track_kaggle_dataset_status",
]
