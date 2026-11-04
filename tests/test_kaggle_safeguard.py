"""Unit tests for Kaggle *large manifold safeguard."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.common_sync import (
    ensure_kaggle_dataset_exists,
    is_kaggle_large_manifold,
    perform_dataset_upload,
    push_kaggle_dataset_metadata,
)
from core.ecosystem import ecosystem
from tools.sync_kaggle_metadata import is_large_manifold

OWNER = ecosystem().kaggle_owner


class TestKaggleLargeManifoldSafeguard(unittest.TestCase):
    """Verifies that Kaggle metadata and upload utilities never target *large manifolds."""

    def test_is_large_manifold_identification(self) -> None:
        """Verify detection of legacy *large manifold handles, URLs, and slugs."""
        large_targets = [
            f"https://www.kaggle.com/datasets/{OWNER}/lemgendizedforexuniverselarge",
            (
                f"https://www.kaggle.com/datasets/{OWNER}/"
                "lemgendizedclassificationmastermanifoldlarge"
            ),
            f"kaggle://{OWNER}/lemgendizedprofessionalmultitaskrestorationlarge",
            f"{OWNER}/lemgendizedforexuniverselarge",
            f"{OWNER}/lemgendizedclassificationmastermanifoldlarge",
            f"{OWNER}/lemgendizedprofessionalmultitaskrestorationlarge",
            f"{OWNER}/lemgendizedretinafacemobilenetlarge",
            f"{OWNER}/test_dataset_large",
            f"{OWNER}/test-dataset-large",
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
            f"{OWNER}/lemgendizedforexuniverse",
            f"{OWNER}/lemgendizedclassificationmaster",
            f"{OWNER}/lemgendizedmultitaskrestorationpro",
            f"{OWNER}/lemgendizedretinaface",
            f"{OWNER}/lemgendizednimaaesthetic",
            f"{OWNER}/lemgendizedupnv2",
            f"{OWNER}/lemgendizedfilmrestorer",
            f"{OWNER}/lemgendizedcodeformer",
            f"{OWNER}/lemgendizedparsenet",
            f"{OWNER}/lemgendizedffanetindoor",
            f"{OWNER}/lemgendizedmirnetlowlight",
            f"{OWNER}/lemgendizedaihelperlemgend",
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
                f"{OWNER}/lemgendizedforexuniverselarge", meta_file
            )
            self.assertFalse(res_handle)

            res_url = push_kaggle_dataset_metadata(
                f"https://www.kaggle.com/datasets/{OWNER}/"
                "lemgendizedclassificationmastermanifoldlarge",
                meta_file,
            )
            self.assertFalse(res_url)

    def test_ensure_kaggle_dataset_exists_rejects_large_manifold(self) -> None:
        """Verify ensure_kaggle_dataset_exists returns False immediately for *large handles."""
        res = ensure_kaggle_dataset_exists(
            f"{OWNER}/lemgendizedforexuniverselarge", title="Forex Large"
        )
        self.assertFalse(res)

    def test_perform_dataset_upload_rejects_large_manifold(self) -> None:
        """Verify perform_dataset_upload returns False immediately for *large handles."""
        with tempfile.TemporaryDirectory() as td:
            dummy_dir = Path(td) / "dummy"
            dummy_dir.mkdir()
            res = perform_dataset_upload(
                dummy_dir, f"{OWNER}/lemgendizedforexuniverselarge"
            )
            self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
