"""
Kaggle dataset metadata push for the sync core.

Sends title, license, keywords, header image and column descriptors through
the Dataset Settings API without re-uploading data, repairing settings Kaggle
rejects (invalid keywords, missing files, title length) and retrying.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kagglesdk.datasets.types.dataset_api_service import ApiUpdateDatasetMetadataRequest

from core.kaggle_errors import KAGGLE_REMOTE_ERRORS
from core.lazy_imports import load_kaggle_api
from core.sync_auth import (
    is_kaggle_large_manifold,
    normalize_kaggle_handle,
    setup_kaggle_auth,
    split_owner_slug,
)
from core.sync_metadata_settings import (
    build_dataset_settings,
    build_file_settings,
    dedupe_keywords,
    header_image_name,
)
from core.sync_remote import ensure_kaggle_dataset_exists

logger = logging.getLogger(__name__)

_MAX_REPAIR_RETRIES = 5


def _kaggle_upload_header_image(api: Any, meta_folder: Path, image_name: str) -> Any:
    """Adapter over the Kaggle client's private ``_upload_dataset_image_file``.

    The public KaggleApi exposes no header-image upload, so this is the single
    place that reaches into the private method; a client upgrade that renames
    it only needs this function updated.
    """
    upload = getattr(api, "_upload_dataset_image_file")
    return upload(
        metadata_file_path=str(meta_folder),
        relative_image_file_path=image_name,
        quiet=True,
    )


def _attach_header_image(
    api: Any, settings: Any, meta: dict[str, Any], meta_folder: Path, clean_handle: str
) -> None:
    """Upload the header/cover image (if any) and store its token on ``settings``."""
    img_name = header_image_name(meta, meta_folder)
    if not img_name or not (meta_folder / img_name).exists():
        return
    try:
        cropped_upload = _kaggle_upload_header_image(api, meta_folder, img_name)
        if cropped_upload:
            settings.image = cropped_upload
            print(f"[INFO] Uploaded header image {img_name} for {clean_handle}")
    except KAGGLE_REMOTE_ERRORS as img_err:
        print(f"[WARN] Could not upload header image {img_name} for {clean_handle}: {img_err}")


def _list_remote_files(api: Any, clean_handle: str) -> set[str]:
    """File paths Kaggle currently stores for the dataset (empty when unavailable)."""
    try:
        remote_file_list = api.dataset_list_files(clean_handle)
        raw_files = getattr(remote_file_list, "files", None)
        if raw_files is not None:
            return {
                f.name for f in raw_files
                if f is not None and getattr(f, "name", None) is not None
            }
    except KAGGLE_REMOTE_ERRORS as exc:
        logger.warning("Could not list remote files for %s: %s", clean_handle, exc)
    return set()


def _has_errors(resp: Any) -> bool:
    """True when an UpdateDatasetMetadata response carries validation errors."""
    return bool(resp and getattr(resp, "errors", None) and len(resp.errors) > 0)


def _fix_invalid_keywords(settings: Any, err: str) -> bool:
    """Drop the keywords Kaggle named as invalid."""
    invalid_kws = re.findall(r'"([^"]+)"', err)
    if invalid_kws and settings.keywords:
        settings.keywords = [k for k in settings.keywords if k not in invalid_kws]
        return True
    return False


def _fix_missing_file(settings: Any, err: str) -> bool:
    """Drop the file entry Kaggle cannot find (or all file entries if unnamed)."""
    missing_match = re.search(r"contain a file named '([^']+)'", err)
    if missing_match and settings.data:
        missing_name = missing_match.group(1)
        settings.data = [f for f in settings.data if getattr(f, "name", None) != missing_name]
        if not settings.data:
            settings.data = None
        return True
    if settings.data:
        settings.data = None
        return True
    return False


def _fix_title_length(settings: Any, _err: str) -> bool:
    """Trim an over-long title at a word boundary."""
    if len(settings.title) > 50:
        settings.title = settings.title[:50].rsplit(" ", 1)[0]
        return True
    return False


# Ordered (error substring, repair) pairs; the first matching repair handles an error.
_ERROR_REPAIRS = (
    ("The following keywords are invalid:", _fix_invalid_keywords),
    ("contain a file named", _fix_missing_file),
    ("The title must be between 6 and 50 characters", _fix_title_length),
)


def _repair_rejected_settings(settings: Any, errors: list[str]) -> bool:
    """Apply a repair for every recognized error; True when any setting changed."""
    should_retry = False
    for err in list(errors):
        for needle, repair in _ERROR_REPAIRS:
            if needle in err:
                if repair(settings, err):
                    should_retry = True
                break
    return should_retry


def _print_synced(file_count: int) -> None:
    print(
        "[SUCCESS] Kaggle dataset metadata & column descriptors synced successfully"
        f" ({file_count} file definitions registered)."
    )


@dataclass
class _MetadataUpdate:
    """One prepared UpdateDatasetMetadata call plus its retry/recovery policy."""
    client: Any
    req: Any
    settings: Any
    file_count: int
    clean_handle: str

    def send(self) -> Any:
        """Send the request as it currently stands."""
        return self.client.datasets.dataset_api_client.update_dataset_metadata(self.req)

    def send_with_repairs(self) -> Any:
        """Send, repairing rejected settings and resending up to five times."""
        resp = self.send()
        for _retry in range(_MAX_REPAIR_RETRIES):
            if not _has_errors(resp) or not _repair_rejected_settings(self.settings, resp.errors):
                break
            # Image upload tokens are single-use; drop the image on retry to avoid a 500 /
            # JSONDecodeError
            self.settings.image = None
            resp = self.send()
        return resp

    def retry_without_image(self) -> bool:
        """Resend once with the consumed image token dropped; True when accepted."""
        if getattr(self.settings, "image", None) is None:
            return False
        self.settings.image = None
        try:
            return not _has_errors(self.send())
        except KAGGLE_REMOTE_ERRORS as img_retry_exc:
            logger.debug("Retry without image token also encountered: %s", img_retry_exc)
            return False

    def recover(self, exc: Exception) -> bool:
        """Recover from a raised update: retry without image, then without column descriptors."""
        # A consumed image token raises (e.g. JSONDecodeError): retry once without the
        # image, keeping the file descriptions
        if self.retry_without_image():
            _print_synced(self.file_count)
            return True

        if not self.file_count:
            print(f"[ERROR] Kaggle metadata update call failed for {self.clean_handle}: {exc}")
            return False
        print(
            f"[WARN] Metadata call failed with file schemas ({exc}); retrying without column"
            " descriptors..."
        )
        self.settings.data = None
        try:
            resp = self.send()
        except KAGGLE_REMOTE_ERRORS as retry_exc:
            print(
                f"[ERROR] Kaggle metadata update call failed for {self.clean_handle}: {retry_exc}"
            )
            return False
        return self._report(resp, file_count=0)

    def _report(self, resp: Any, file_count: int) -> bool:
        """Print the final rejected/synced line for ``resp``; True when accepted."""
        if _has_errors(resp):
            print(
                f"[ERROR] Kaggle metadata update rejected for {self.clean_handle}: {resp.errors}"
            )
            return False
        _print_synced(file_count)
        return True

    def submit(self) -> bool:
        """Send with repair/recovery retries; True when Kaggle accepted the metadata."""
        try:
            resp = self.send_with_repairs()
        except KAGGLE_REMOTE_ERRORS as exc:
            return self.recover(exc)
        return self._report(resp, self.file_count)


def _load_metadata(metadata_path: Path) -> tuple[Path, dict[str, Any]] | None:
    """Locate and parse dataset-metadata.json (file path or its folder)."""
    meta_file = (
        metadata_path if metadata_path.is_file() else metadata_path / "dataset-metadata.json"
    )
    if not meta_file.exists():
        print(f"[ERROR] Metadata file not found at {meta_file}")
        return None
    return meta_file, json.loads(meta_file.read_text(encoding="utf-8"))


def push_kaggle_dataset_metadata(repo_id: str, metadata_path: Path) -> bool:
    """Pushes complete Kaggle metadata including single license and column descriptors."""
    clean_handle = normalize_kaggle_handle(repo_id)
    if is_kaggle_large_manifold(clean_handle):
        print(f"[SAFEGUARD] Refusing to touch legacy *large manifold: {clean_handle}")
        return False

    owner, slug = split_owner_slug(clean_handle)
    setup_kaggle_auth(default_user=owner)

    loaded = _load_metadata(metadata_path)
    if loaded is None:
        return False
    meta_file, meta = loaded

    # The dataset must exist before its metadata is updated, or Kaggle answers 403;
    # a missing dataset is created with a minimal README placeholder.
    if not ensure_kaggle_dataset_exists(clean_handle, title=meta.get("title", slug)):
        print(
            f"[WARN] Could not ensure Kaggle dataset placeholder exists for '{clean_handle}'."
            " Attempting metadata update anyway..."
        )

    api = load_kaggle_api()()
    api.authenticate()

    req = ApiUpdateDatasetMetadataRequest()
    req.owner_slug = owner
    req.dataset_slug = slug
    settings = build_dataset_settings(meta, slug)
    _attach_header_image(api, settings, meta, meta_file.parent, clean_handle)
    if meta.get("keywords"):
        settings.keywords = dedupe_keywords(meta["keywords"])

    # Discover actual files on Kaggle for accurate path mapping
    remote_files = _list_remote_files(api, clean_handle)
    files_data = build_file_settings(meta, meta_file.parent, remote_files)
    if files_data:
        settings.data = files_data
    req.settings = settings

    return _MetadataUpdate(
        client=api.build_kaggle_client(),
        req=req,
        settings=settings,
        file_count=len(files_data),
        clean_handle=clean_handle,
    ).submit()
