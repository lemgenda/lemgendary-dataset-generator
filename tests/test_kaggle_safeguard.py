"""Unit tests for Kaggle *large manifold safeguard."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from core.common_sync import (
    ensure_kaggle_dataset_exists,
    is_kaggle_large_manifold,
    perform_dataset_upload,
    push_kaggle_dataset_metadata,
)
from tools.sync_kaggle_metadata import is_large_manifold


class TestKaggleLargeManifoldSafeguard(unittest.TestCase):
    """Verifies that Kaggle metadata and upload utilities never target *large manifolds."""

    def test_is_large_manifold_identification(self) -> None:
        """Verify detection of legacy *large manifold handles, URLs, and slugs."""
        large_targets = [
            "https://www.kaggle.com/datasets/lemtreursi/lemgendizedforexuniverselarge",
            "https://www.kaggle.com/datasets/lemtreursi/lemgendizedclassificationmastermanifoldlarge",
            "kaggle://lemtreursi/lemgendizedprofessionalmultitaskrestorationlarge",
            "lemtreursi/lemgendizedforexuniverselarge",
            "lemtreursi/lemgendizedclassificationmastermanifoldlarge",
            "lemtreursi/lemgendizedprofessionalmultitaskrestorationlarge",
            "lemtreursi/lemgendizedretinafacemobilenetlarge",
            "lemtreursi/test_dataset_large",
            "lemtreursi/test-dataset-large",
        ]
        for target in large_targets:
            self.assertTrue(
                is_large_manifold(target),
                f"Failed to detect *large manifold in sync_kaggle_metadata: {target}",
            )
            self.assertTrue(
                is_kaggle_large_manifold(target),
                f"Failed to detect *large manifold in common_sync: {target}",
            )

    def test_canonical_manifolds_are_permitted(self) -> None:
        """Verify that canonical production datasets are not flagged as *large."""
        canonical_targets = [
            "lemtreursi/lemgendizedforexuniverse",
            "lemtreursi/lemgendizedclassificationmaster",
            "lemtreursi/lemgendizedmultitaskrestorationpro",
            "lemtreursi/lemgendizedretinaface",
            "lemtreursi/lemgendizednimaaesthetic",
            "lemtreursi/lemgendizedupnv2",
            "lemtreursi/lemgendizedfilmrestorer",
            "lemtreursi/lemgendizedcodeformer",
            "lemtreursi/lemgendizedparsenet",
            "lemtreursi/lemgendizedffanetindoor",
            "lemtreursi/lemgendizedmirnetlowlight",
            "lemtreursi/lemgendizedaihelperlemgend",
        ]
        for target in canonical_targets:
            self.assertFalse(
                is_large_manifold(target),
                f"Incorrectly flagged canonical manifold in sync_kaggle_metadata: {target}",
            )
            self.assertFalse(
                is_kaggle_large_manifold(target),
                f"Incorrectly flagged canonical manifold in common_sync: {target}",
            )

    def test_push_kaggle_dataset_metadata_rejects_large_manifold(self) -> None:
        """Verify push_kaggle_dataset_metadata returns False immediately for *large handles."""
        with tempfile.TemporaryDirectory() as td:
            meta_file = Path(td) / "dataset-metadata.json"
            meta_file.write_text('{"title": "Test Large"}', encoding="utf-8")

            res_handle = push_kaggle_dataset_metadata(
                "lemtreursi/lemgendizedforexuniverselarge", meta_file
            )
            self.assertFalse(res_handle)

            res_url = push_kaggle_dataset_metadata(
                "https://www.kaggle.com/datasets/lemtreursi/lemgendizedclassificationmastermanifoldlarge",
                meta_file,
            )
            self.assertFalse(res_url)

    def test_ensure_kaggle_dataset_exists_rejects_large_manifold(self) -> None:
        """Verify ensure_kaggle_dataset_exists returns False immediately for *large handles."""
        res = ensure_kaggle_dataset_exists(
            "lemtreursi/lemgendizedforexuniverselarge", title="Forex Large"
        )
        self.assertFalse(res)

    def test_perform_dataset_upload_rejects_large_manifold(self) -> None:
        """Verify perform_dataset_upload returns False immediately for *large handles."""
        with tempfile.TemporaryDirectory() as td:
            dummy_dir = Path(td) / "dummy"
            dummy_dir.mkdir()
            res = perform_dataset_upload(
                dummy_dir, "lemtreursi/lemgendizedforexuniverselarge"
            )
            self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
