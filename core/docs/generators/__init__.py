"""LemGendary Dataset Documentation Generators.

Separate, modular generator scripts for each document asset required by
the LemGendary Manifold Specification:
- category.txt
- classes.txt
- <ModelName>.jpg (Cover Image)
- dataset_info.yaml
- index.json
- README.md
- dataset-metadata.json (Kaggle Frictionless Metadata)
"""

from __future__ import annotations

from core.docs.generators.category_txt import generate_category_txt
from core.docs.generators.classes_txt import generate_classes_txt
from core.docs.generators.cover_image import generate_cover_image
from core.docs.generators.dataset_info_yaml import generate_dataset_info_yaml
from core.docs.generators.index_json import generate_index_json
from core.docs.generators.kaggle_metadata import generate_kaggle_metadata
from core.docs.generators.readme_md import generate_readme_md

__all__ = [
    "generate_category_txt",
    "generate_classes_txt",
    "generate_cover_image",
    "generate_dataset_info_yaml",
    "generate_index_json",
    "generate_kaggle_metadata",
    "generate_readme_md",
]
