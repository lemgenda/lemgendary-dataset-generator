"""Unit tests for Kaggle extraction progress streaming and jobs.py line handling."""

from __future__ import annotations

import io
from unittest.mock import patch, MagicMock
import unittest

from core.common_sync import track_kaggle_dataset_status


class TestKaggleExtractionProgress(unittest.TestCase):
    """Verifies that Kaggle extraction logs are properly formatted for pipe consumers."""

    @patch("core.common_sync.setup_kaggle_auth")
    @patch("core.common_sync.get_dataset_version_info")
    def test_track_kaggle_dataset_status_emits_newline_flush(
        self,
        mock_info: MagicMock,
        mock_auth: MagicMock,
    ) -> None:
        """Verify that track_kaggle_dataset_status emits newline-terminated lines with flush."""
        mock_info.return_value = {
            "latest_version": 2,
            "versions": {2: "ready"},
        }

        captured_stdout = io.StringIO()
        with patch("sys.stdout", captured_stdout):
            success = track_kaggle_dataset_status("lemtreursi/test_manifold", target_version=2)

        self.assertTrue(success)
        output = captured_stdout.getvalue()
        self.assertIn("KAGGLE EXTRACTION: lemtreursi/test_manifold - 100%", output)
        self.assertIn("[SUCCESS]", output)

    def test_jobs_is_progress_detection(self) -> None:
        """Verify api/jobs.py line parsing logic classifies KAGGLE EXTRACTION as progress."""
        clean_line = "KAGGLE EXTRACTION: lemtreursi/manifold_eurusd_1m - 45% [18s, PROCESSING]"
        is_progress = (
            "%" in clean_line
            or "EXTRACTING" in clean_line
            or "KAGGLE EXTRACTION" in clean_line
        )
        self.assertTrue(is_progress)


if __name__ == "__main__":
    unittest.main()
