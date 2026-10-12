"""Unit tests for LemGendary Dataset Documentation Generator.

Verifies modular core.docs subpackage, separate generator scripts for each document asset,
and backward-compatible core.doc_generator facade.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
import yaml

from core import doc_generator
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

            sample_count = generate_dataset_docs(out_root, sample_index, "LemGendizedNafNetDebluring")
            self.assertEqual(sample_count, 3)

            # Check index.json
            idx_file = out_root / "index.json"
            self.assertTrue(idx_file.exists())
            with open(idx_file, "r", encoding="utf-8") as f:
                loaded_idx = json.load(f)
            self.assertEqual(len(loaded_idx), 3)

            # Check dataset_info.yaml
            info_file = out_root / "dataset_info.yaml"
            self.assertTrue(info_file.exists())
            with open(info_file, "r", encoding="utf-8") as f:
                loaded_info = yaml.safe_load(f)
            self.assertEqual(loaded_info["task"], "restoration")
            self.assertEqual(loaded_info["count"], 3)
            self.assertIn("GoPro Deblurring Dataset", loaded_info["original_sources"])

            # Check category.txt
            cat_file = out_root / "category.txt"
            self.assertTrue(cat_file.exists())
            with open(cat_file, "r", encoding="utf-8") as f:
                cat_text = f.read().strip()
            self.assertTrue(len(cat_text) > 0)

            # Check classes.txt
            classes_file = out_root / "classes.txt"
            self.assertTrue(classes_file.exists())
            with open(classes_file, "r", encoding="utf-8") as f:
                classes_text = f.read().strip()
            self.assertEqual(classes_text, "restoration")

            # Check README.md
            readme_file = out_root / "README.md"
            self.assertTrue(readme_file.exists())
            with open(readme_file, "r", encoding="utf-8") as f:
                readme_text = f.read()
            self.assertIn("# LemGendizedNafNetDebluring", readme_text)
            self.assertIn("GoPro Deblurring Dataset", readme_text)
            self.assertIn("Total Samples", readme_text)

            # Check dataset-metadata.json (Kaggle)
            meta_file = out_root / "dataset-metadata.json"
            self.assertTrue(meta_file.exists())
            with open(meta_file, "r", encoding="utf-8") as f:
                meta_payload = json.load(f)
            self.assertEqual(meta_payload["licenses"][0]["name"], "CC-BY-NC-4.0")
            self.assertEqual(meta_payload["id"], "lemtreursi/lemgendizednafnetdebluring")
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

            sample_count = generate_dataset_docs(out_root, None, "LemGendizedForexUniverse", overrides)
            self.assertIsInstance(sample_count, int)

            # Check dataset_info.yaml
            info_file = out_root / "dataset_info.yaml"
            self.assertTrue(info_file.exists())
            with open(info_file, "r", encoding="utf-8") as f:
                loaded_info = yaml.safe_load(f)
            self.assertEqual(loaded_info["task"], "forex")
            self.assertEqual(loaded_info["dataset_type"], "forex")
            self.assertEqual(loaded_info["pairs"], ["EURUSD", "GBPUSD", "USDJPY"])
            self.assertEqual(loaded_info["timeframe_rungs"], [1, 5, 15, 60, 240, 1440])
            self.assertEqual(loaded_info["lookback_bars"], 168)
            self.assertEqual(loaded_info["format"], "parquet")

            # Check classes.txt
            classes_file = out_root / "classes.txt"
            self.assertTrue(classes_file.exists())
            with open(classes_file, "r", encoding="utf-8") as f:
                classes_text = f.read().strip()
            self.assertEqual(classes_text, "SELL\nHOLD\nBUY")

            # Check dataset-metadata.json
            meta_file = out_root / "dataset-metadata.json"
            self.assertTrue(meta_file.exists())
            with open(meta_file, "r", encoding="utf-8") as f:
                meta_payload = json.load(f)
            self.assertTrue(len(meta_payload["resources"]) > 0)
            first_resource = meta_payload["resources"][0]
            self.assertEqual(first_resource["schema"]["fields"], FOREX_COLUMN_FIELDS)

    def test_separate_generator_scripts(self) -> None:
        """Verify each separate generator script runs independently and produces 100% accurate output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifold_dir = Path(tmp_dir) / "LemGendizedCodeFormer"
            manifold_dir.mkdir(parents=True, exist_ok=True)

            ctx = resolve_manifold_context(manifold_dir, pascal_name="LemGendizedCodeFormer")

            # 1. category_txt
            cat_file = generate_category_txt(ctx)
            self.assertTrue(cat_file.exists())
            self.assertEqual(cat_file.read_text(encoding="utf-8").strip(), "Image Restoration")

            # 2. classes_txt
            cls_file = generate_classes_txt(ctx)
            self.assertTrue(cls_file.exists())
            self.assertEqual(cls_file.read_text(encoding="utf-8").strip(), "restoration")

            # 3. cover_image
            img_file = generate_cover_image(ctx)
            self.assertTrue(img_file.exists())
            with Image.open(img_file) as im:
                self.assertEqual(im.size, (564, 284))

            # 4. dataset_info_yaml
            yaml_file = generate_dataset_info_yaml(ctx)
            self.assertTrue(yaml_file.exists())
            with open(yaml_file, "r", encoding="utf-8") as yf:
                info = yaml.safe_load(yf)
            self.assertEqual(info["task"], "restoration")
            self.assertEqual(info["name"], "LemGendizedCodeFormer")

            # 5. index_json
            sample_idx = [{"id": 0, "name": "sample_0", "source": "FFHQ", "split": "train", "task": "restoration"}]
            idx_file = generate_index_json(ctx, final_index=sample_idx)
            self.assertTrue(idx_file.exists())
            with open(idx_file, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            self.assertEqual(len(data), 1)

            # 6. readme_md
            readme_file, readme_text = generate_readme_md(ctx)
            self.assertTrue(readme_file.exists())
            self.assertIn("# LemGendizedCodeFormer", readme_text)
            self.assertIn("## Dataset Overview", readme_text)

            # 7. kaggle_metadata
            meta_data = generate_kaggle_metadata(ctx, readme_content=readme_text)
            meta_file = manifold_dir / "dataset-metadata.json"
            self.assertTrue(meta_file.exists())
            self.assertEqual(meta_data["title"], "LemGendized CodeFormer")
            self.assertEqual(meta_data["id"], "lemtreursi/lemgendizedcodeformer")
            self.assertTrue(20 <= len(meta_data["subtitle"]) <= 80)
            self.assertEqual(meta_data["headerImage"], img_file.name)
            self.assertTrue(len(meta_data["resources"]) >= 6)

    def test_facade_parity(self) -> None:
        """Verify core.doc_generator facade re-exports match core.docs."""
        self.assertIs(doc_generator.generate_dataset_docs, generate_dataset_docs)
        self.assertIs(doc_generator.format_source, format_source)
        self.assertEqual(doc_generator.FOREX_COLUMN_FIELDS, FOREX_COLUMN_FIELDS)


if __name__ == "__main__":
    unittest.main()
