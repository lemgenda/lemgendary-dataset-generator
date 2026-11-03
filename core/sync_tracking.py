"""
Kaggle server-side extraction monitor for the sync core.

Polls a dataset version until Kaggle reports it ready or failed, rendering a
tqdm bar plus newline-terminated ``KAGGLE EXTRACTION: ...`` lines that the API
job runner parses as progress.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from kagglehub.clients import build_kaggle_client
from kagglesdk.datasets.types.dataset_api_service import (
    ApiGetDatasetFilesSummaryRequest,
)
from tqdm import tqdm

from core.kaggle_errors import KAGGLE_CLIENT_ERRORS
from core.sync_auth import setup_kaggle_auth, split_owner_slug
from core.sync_remote import get_dataset_status, get_dataset_version_info

logger = logging.getLogger(__name__)

_FAILED_STATES = ("error", "failed")


def _count_remote_files(handle: str) -> int:
    """Number of files Kaggle currently lists for ``handle`` (0 when unknown)."""
    owner, slug = split_owner_slug(handle)
    try:
        with build_kaggle_client() as client:
            sreq = ApiGetDatasetFilesSummaryRequest()
            sreq.owner_slug = owner
            sreq.dataset_slug = slug
            sresp = client.datasets.dataset_api_client.get_dataset_files_summary(sreq)
            summary_files = (
                getattr(sresp, "files", None)
                or getattr(sresp, "dataset_files", None)
                or []
            )
            return len(summary_files)
    except KAGGLE_CLIENT_ERRORS as exc:
        logger.debug("Failed querying Kaggle dataset files summary: %s", exc)
        return 0


@dataclass
class _ExtractionMonitor:
    """Progress state for one extraction watch: handle, version, bar and last percentage."""
    handle: str
    target_ver: int
    expected_files: int
    pbar: Any
    start_time: float
    last_pct: float = 0.0

    def elapsed(self) -> int:
        """Whole seconds since monitoring started."""
        return int(time.time() - self.start_time)

    def advance_to(self, pct: float) -> None:
        """Move the bar forward to ``pct`` (never backwards)."""
        if pct > self.last_pct:
            self.pbar.update(pct - self.last_pct)
            self.last_pct = pct

    def simulated_pct(self, ceiling: float, ramp_seconds: float, ramp_pct: float) -> float:
        """Time-based progress estimate used when Kaggle gives no file counts."""
        return min(ceiling, (time.time() - self.start_time) / ramp_seconds * ramp_pct)

    def finish_ready(self, success_msg: str) -> bool:
        """Close the bar at 100%, print the completion lines and report success."""
        self.pbar.n = 100
        self.pbar.set_postfix_str("COMPLETE")
        self.pbar.refresh()
        self.pbar.close()
        print(
            f"KAGGLE EXTRACTION: {self.handle} - 100% [{self.elapsed()}s, COMPLETE]",
            flush=True,
        )
        print(success_msg, flush=True)
        return True

    def finish_failed(self, error_msg: str) -> bool:
        """Close the bar as FAILED, print the failure lines and report failure."""
        self.pbar.set_postfix_str("FAILED")
        self.pbar.refresh()
        self.pbar.close()
        print(f"KAGGLE EXTRACTION: {self.handle} - 0% [{self.elapsed()}s, FAILED]", flush=True)
        print(error_msg, flush=True)
        return False

    def print_progress(self) -> None:
        """Emit one pipe-friendly progress line for API job consumers."""
        status_postfix = self.pbar.postfix or "PROCESSING"
        print(
            f"KAGGLE EXTRACTION: {self.handle} - {int(self.pbar.n)}% [{self.elapsed()}s,"
            f" {status_postfix}]",
            flush=True,
        )


def _poll_listed_version(mon: _ExtractionMonitor, status_val: str) -> bool | None:
    """Handle a poll where the target version is listed; returns the outcome or None."""
    if status_val == "ready":
        return mon.finish_ready(
            f"[SUCCESS] Kaggle server-side extraction complete! Dataset '{mon.handle}'"
            f" version {mon.target_ver} is ready."
        )
    if status_val in _FAILED_STATES:
        return mon.finish_failed(
            "\n[ERROR] Kaggle reports extraction/processing failed for version"
            f" {mon.target_ver}."
        )

    file_pct = None
    if mon.expected_files > 0:
        cur_files = _count_remote_files(mon.handle)
        if cur_files > 0:
            file_pct = min(99.0, (cur_files / mon.expected_files) * 100.0)

    if file_pct is not None and file_pct > mon.last_pct:
        mon.advance_to(file_pct)
        mon.pbar.set_postfix_str(f"{status_val.upper()} ({int(file_pct)}% files)")
    else:
        mon.advance_to(mon.simulated_pct(95.0, 120.0, 80.0))
        mon.pbar.set_postfix_str(status_val.upper())
    return None


def _poll_unlisted_version(mon: _ExtractionMonitor) -> bool | None:
    """Handle a poll where the version is not listed yet, using the dataset status."""
    raw_status = get_dataset_status(mon.handle)
    status_lower = raw_status.lower() if raw_status else ""
    if status_lower == "ready":
        return mon.finish_ready(
            f"[SUCCESS] Kaggle server-side extraction complete! Dataset '{mon.handle}' is"
            " ready."
        )
    if status_lower in _FAILED_STATES:
        return mon.finish_failed(
            f"\n[ERROR] Kaggle reports processing failed for dataset '{mon.handle}'."
        )

    mon.advance_to(mon.simulated_pct(90.0, 180.0, 75.0))
    mon.pbar.set_postfix_str((raw_status or "QUEUED / UNPACKING").upper())
    return None


def _latest_version_if_pending(clean_handle: str) -> int | None:
    """Latest version number, or None when that version is already ready."""
    info = get_dataset_version_info(clean_handle)
    latest = info["latest_version"] or 1
    versions = info["versions"]
    if latest in versions and versions[latest].lower() == "ready":
        print(f"[KAGGLE STATUS] Dataset '{clean_handle}' version {latest} is READY.")
        return None
    return latest


def track_kaggle_dataset_status(
    repo_id: str,
    target_version: int | None = None,
    expected_files: int = 0,
    timeout: int = 1800,
    poll_interval: int = 4
) -> bool:
    """Monitor Kaggle server-side extraction/processing with real-time tqdm progress tracking."""
    clean_handle = repo_id.replace("kaggle://", "")
    owner, _ = split_owner_slug(clean_handle)
    setup_kaggle_auth(default_user=owner)

    if target_version is None:
        target_version = _latest_version_if_pending(clean_handle)
        if target_version is None:
            return True

    print(
        f"\n[KAGGLE] Monitoring server-side extraction for '{clean_handle}' (Target Version:"
        f" {target_version})..."
    )
    mon = _ExtractionMonitor(
        handle=clean_handle,
        target_ver=target_version,
        expected_files=expected_files,
        start_time=time.time(),
        pbar=tqdm(
            total=100,
            desc=f"KAGGLE EXTRACTION (v{target_version})",
            unit="%",
            colour="cyan",
            dynamic_ncols=False,
            bar_format="{desc} - {percentage:3.0f}% [{elapsed}, {postfix}]",
        ),
    )

    while time.time() - mon.start_time < timeout:
        versions = get_dataset_version_info(clean_handle)["versions"]
        if target_version in versions:
            outcome = _poll_listed_version(mon, versions[target_version].lower())
        else:
            outcome = _poll_unlisted_version(mon)
        if outcome is not None:
            return outcome
        mon.print_progress()
        time.sleep(poll_interval)

    mon.pbar.close()
    print(
        f"\n[TIMEOUT] Monitoring timed out after {timeout} seconds. Dataset may still be processing"
        " on Kaggle."
    )
    return False
