"""Unit tests for LemGendary Dataset Documentation Generator.

Verifies modular core.docs subpackage, separate generator scripts for each document asset,
and backward-compatible core.doc_generator facade.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from core import doc_generator, ecosystem
from core.docs import (
    FOREX_COLUMN_FIELDS,
    clean_readme,
    format_source,
    generate_category_txt,
    generate_classes_txt,
    generate_cover_image,
    generate_dataset_docs,
    generate_dataset_info_yaml,
    generate_index_json,
    generate_kaggle_metadata,
    generate_readme_md,
    resolve_manifold_context,
)
from core.docs.metadata import MODELS_META
from core.docs.templates import build_models_markdown


def _read_text(case: unittest.TestCase, path: Path) -> str:
    """Assert ``path`` exists and return its UTF-8 text."""
    case.assertTrue(path.exists(), f"{path.name} was not generated")
    return path.read_text(encoding="utf-8")


def _read_json(case: unittest.TestCase, path: Path) -> Any:
    """Assert ``path`` exists and return its parsed JSON."""
    return json.loads(_read_text(case, path))


def _read_yaml(case: unittest.TestCase, path: Path) -> Any:
    """Assert ``path`` exists and return its parsed YAML."""
    return yaml.safe_load(_read_text(case, path))


class TestDocumentationGenerator(unittest.TestCase):
    """Test suite for modular documentation generator and individual script generators."""

    def test_format_source_mapping(self) -> None:
        """Verify format_source normalizes known and generic source names."""
        self.assertEqual(format_source("celebamask"), "CelebAMask")
        self.assertEqual(format_source("df2k"), "DF2K-OST")
        self.assertEqual(format_source("coco"), "COCO 2017")
        self.assertEqual(format_source("koniq10k"), "KonIQ-10k")
        self.assertEqual(format_source("gopro"), "GoPro Deblurring Dataset")
        self.assertEqual(format_source("custom-noise-set"), "Custom Noise Set")

    def test_clean_readme(self) -> None:
        """Verify clean_readme eliminates consecutive blank lines and trailing whitespace."""
        raw = "# Title\n\n\n\nParagraph 1.\n\n\nParagraph 2.\n\n\n"
        cleaned = clean_readme(raw)
        self.assertEqual(cleaned, "# Title\n\nParagraph 1.\n\nParagraph 2.\n")

    def test_vision_dataset_docs_generation(self) -> None:
        """Verify documentation generation for a vision manifold."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir) / "LemGendizedNafNetDebluring"
            out_root.mkdir(parents=True, exist_ok=True)

            sample_index = [
                {"name": "img_001", "source": "gopro", "split": "train", "task": "restoration"},
                {"name": "img_002", "source": "gopro", "split": "train", "task": "restoration"},
                {"name": "img_003", "source": "gopro", "split": "val", "task": "restoration"},
            ]

            sample_count = generate_dataset_docs(
                out_root, sample_index, "LemGendizedNafNetDebluring"
            )
            self.assertEqual(sample_count, 3)

            # Check index.json
            self.assertEqual(len(_read_json(self, out_root / "index.json")), 3)

            # Check dataset_info.yaml
            loaded_info = _read_yaml(self, out_root / "dataset_info.yaml")
            self.assertEqual(loaded_info["task"], "restoration")
            self.assertEqual(loaded_info["count"], 3)
            self.assertIn("GoPro Deblurring Dataset", loaded_info["original_sources"])

            # Check category.txt
            self.assertTrue(len(_read_text(self, out_root / "category.txt").strip()) > 0)

            # Check classes.txt
            self.assertEqual(_read_text(self, out_root / "classes.txt").strip(), "restoration")

            # Check README.md
            readme_text = _read_text(self, out_root / "README.md")
            self.assertIn("# LemGendizedNafNetDebluring", readme_text)
            self.assertIn("GoPro Deblurring Dataset", readme_text)
            self.assertIn("Total Samples", readme_text)

            # Check dataset-metadata.json (Kaggle)
            meta_payload = _read_json(self, out_root / "dataset-metadata.json")
            self.assertEqual(meta_payload["licenses"][0]["name"], "CC-BY-NC-4.0")
            self.assertEqual(
                meta_payload["id"], f"{ecosystem.kaggle_owner()}/lemgendizednafnetdebluring"
            )
            self.assertFalse(meta_payload["isPrivate"])
            self.assertTrue(len(meta_payload["keywords"]) >= 5)

    def test_forex_dataset_docs_generation(self) -> None:
        """Verify documentation generation for a forex temporal manifold."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir) / "LemGendizedForexUniverse"
            out_root.mkdir(parents=True, exist_ok=True)

            overrides = {
                "dataset_type": "forex",
                "pairs": ["EURUSD", "GBPUSD", "USDJPY"],
                "timeframe_rungs": [1, 5, 15, 60, 240, 1440],
                "start_date": "2019-01-01",
                "lookback_bars": 168,
                "category": "Forex & Financial Time-Series",
            }

            sample_count = generate_dataset_docs(
                out_root, None, "LemGendizedForexUniverse", overrides
            )
            self.assertIsInstance(sample_count, int)

            # Check dataset_info.yaml
            loaded_info = _read_yaml(self, out_root / "dataset_info.yaml")
            self.assertEqual(loaded_info["task"], "forex")
            self.assertEqual(loaded_info["dataset_type"], "forex")
            self.assertEqual(loaded_info["pairs"], ["EURUSD", "GBPUSD", "USDJPY"])
            self.assertEqual(loaded_info["timeframe_rungs"], [1, 5, 15, 60, 240, 1440])
            self.assertEqual(loaded_info["lookback_bars"], 168)
            self.assertEqual(loaded_info["format"], "parquet")

            # Check classes.txt
            self.assertEqual(
                _read_text(self, out_root / "classes.txt").strip(), "SELL\nHOLD\nBUY"
            )

            # Check dataset-metadata.json
            meta_payload = _read_json(self, out_root / "dataset-metadata.json")
            self.assertTrue(len(meta_payload["resources"]) > 0)
            first_resource = meta_payload["resources"][0]
            self.assertEqual(first_resource["schema"]["fields"], FOREX_COLUMN_FIELDS)

    def test_separate_generator_scripts(self) -> None:
        """Each document generator runs on its own and produces the expected output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifold_dir = Path(tmp_dir) / "LemGendizedCodeFormer"
            manifold_dir.mkdir(parents=True, exist_ok=True)

            ctx = resolve_manifold_context(manifold_dir, pascal_name="LemGendizedCodeFormer")

            # 1. category_txt
            cat_file = generate_category_txt(ctx)
            self.assertEqual(_read_text(self, cat_file).strip(), "Image Restoration")

            # 2. classes_txt
            self.assertEqual(
                _read_text(self, generate_classes_txt(ctx)).strip(), "restoration"
            )

            # 3. cover_image
            img_file = generate_cover_image(ctx)
            self.assertTrue(img_file.exists())
            with Image.open(img_file) as im:
                self.assertEqual(im.size, (564, 284))

            # 4. dataset_info_yaml
            info = _read_yaml(self, generate_dataset_info_yaml(ctx))
            self.assertEqual(info["task"], "restoration")
            self.assertEqual(info["name"], "LemGendizedCodeFormer")

            # 5. index_json
            sample_idx = [
                {
                    "id": 0,
                    "name": "sample_0",
                    "source": "FFHQ",
                    "split": "train",
                    "task": "restoration",
                }
            ]
            idx_file = generate_index_json(ctx, final_index=sample_idx)
            self.assertEqual(len(_read_json(self, idx_file)), 1)

            # 6. readme_md
            readme_file, readme_text = generate_readme_md(ctx)
            self.assertTrue(readme_file.exists())
            self.assertIn("# LemGendizedCodeFormer", readme_text)
            self.assertIn("## Dataset Overview", readme_text)

            # 7. kaggle_metadata
            meta_data = generate_kaggle_metadata(ctx, readme_content=readme_text)
            self.assertTrue((manifold_dir / "dataset-metadata.json").exists())
            self.assertEqual(meta_data["title"], "LemGendized CodeFormer")
            self.assertEqual(meta_data["id"], f"{ecosystem.kaggle_owner()}/lemgendizedcodeformer")
            self.assertTrue(20 <= len(meta_data["subtitle"]) <= 80)
            self.assertEqual(meta_data["headerImage"], img_file.name)
            self.assertTrue(len(meta_data["resources"]) >= 6)

    def test_regeneration_keeps_recorded_label_fields(self) -> None:
        """label_json, classes and landmark_source written by a repair survive a
        dataset_info.yaml regeneration; classes.txt follows the recorded classes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifold_dir = Path(tmp_dir) / "LemGendizedCodeFormer"
            manifold_dir.mkdir(parents=True, exist_ok=True)
            recorded = {"label_json": "face_landmarks", "classes": ["face"],
                        "landmark_source": "sfhq-110pt"}
            (manifold_dir / "dataset_info.yaml").write_text(
                yaml.safe_dump({"count": 1, **recorded}), encoding="utf-8")
            ctx = resolve_manifold_context(manifold_dir, pascal_name="LemGendizedCodeFormer")
            info = _read_yaml(self, generate_dataset_info_yaml(ctx))
            self.assertEqual({k: info[k] for k in recorded}, recorded)
            self.assertEqual(_read_text(self, generate_classes_txt(ctx)).split(), ["face"])

    def test_facade_parity(self) -> None:
        """Verify core.doc_generator facade re-exports match core.docs."""
        self.assertIs(doc_generator.generate_dataset_docs, generate_dataset_docs)
        self.assertIs(doc_generator.format_source, format_source)
        self.assertEqual(doc_generator.FOREX_COLUMN_FIELDS, FOREX_COLUMN_FIELDS)


class TestModelTargets(unittest.TestCase):
    """SOTA target tables of the models bound to a manifold."""

    def test_retinaface_targets_are_measured_metrics_marked_provisional(self) -> None:
        """RetinaFace lists box IoU, AP50, AP50-95 and NME (lower is better), provisional."""
        block = build_models_markdown("LemGendizedRetinaFace", MODELS_META)
        for row in ("| **Box IoU** |", "| **AP50** |", "| **AP50-95** |",
                    "| **Landmark NME** | < 0.15 | < 0.12 | **< 0.1** |"):
            self.assertIn(row, block)
        self.assertNotIn("mAP", block)
        self.assertIn("**Provisional targets**", block)


if __name__ == "__main__":
    unittest.main()
