"""LemGendary Dataset Documentation Orchestrator.

Master coordinator invoking individual document generator scripts in sequence:
- index.json
- dataset_info.yaml
- category.txt
- classes.txt
- <ModelName>.jpg (Cover Image)
- README.md
- dataset-metadata.json (Kaggle Frictionless Metadata)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

_MODULE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from core.docs.context import resolve_manifold_context
from core.docs.generators import (
    generate_category_txt,
    generate_classes_txt,
    generate_cover_image,
    generate_dataset_info_yaml,
    generate_index_json,
    generate_kaggle_metadata,
    generate_readme_md,
)
from core.docs.metadata import UNIFIED_DATA

logger = logging.getLogger(__name__)


def generate_dataset_docs(
    output_root: str | Path,
    final_index: list[dict[str, Any]] | None = None,
    pascal_name: str | None = None,
    overrides: dict[str, Any] | None = None,
    force_cover_image: bool = False,
) -> int:
    """Generate all standard documentation and manifest files for a dataset manifold.

    Invokes separate modular generator scripts in proper sequence:
    1. index.json
    2. dataset_info.yaml
    3. category.txt
    4. classes.txt
    5. <ModelName>.jpg (Cover Image)
    6. README.md
    7. dataset-metadata.json (Kaggle metadata)

    Returns:
        int: Total number of samples in the manifold.
    """
    output_root = Path(output_root).resolve()
    ctx = resolve_manifold_context(
        output_root=output_root,
        pascal_name=pascal_name,
        overrides=overrides,
        final_index=final_index,
    )

    # 1. index.json
    generate_index_json(ctx, final_index=final_index)

    # 2. dataset_info.yaml
    generate_dataset_info_yaml(ctx, overrides=overrides)

    # 3. category.txt
    generate_category_txt(ctx)

    # 4. classes.txt
    generate_classes_txt(ctx)

    # 5. Cover Image (<ModelName>.jpg)
    generate_cover_image(ctx, force_regenerate=force_cover_image)

    # 6. README.md
    _readme_path, readme_content = generate_readme_md(ctx)

    # 7. dataset-metadata.json (Kaggle API metadata)
    generate_kaggle_metadata(ctx, readme_content=readme_content)

    return ctx.total_samples


def regenerate_all_docs(
    datasets_dir: str | Path | None = None,
    force_cover_image: bool = False,
) -> int:
    """Regenerate documentation for all existing manifolds discovered in workspace."""
    if datasets_dir is None:
        cand = Path(__file__).resolve().parent.parent.parent.parent / "LemGendaryDatasets"
        datasets_dir = cand if cand.exists() else Path(__file__).resolve().parent.parent.parent / "LemGendaryDatasets"
    datasets_dir = Path(datasets_dir).resolve()

    if not datasets_dir.exists():
        print(f"[ERROR] Manifolds directory does not exist: {datasets_dir}")
        return 0

    print(f"Scanning manifolds in {datasets_dir}...")

    # Identify all active folders in LemGendaryDatasets
    active_manifold_dirs: list[Path] = [
        p for p in sorted(datasets_dir.iterdir())
        if p.is_dir() and not p.name.startswith(".")
    ]

    print(f"Found {len(active_manifold_dirs)} active manifold directories to process.")
    count = 0

    for m_dir in active_manifold_dirs:
        print(f"Regenerating docs for {m_dir.name}...")
        try:
            samples = generate_dataset_docs(
                output_root=m_dir,
                final_index=None,
                pascal_name=m_dir.name,
                force_cover_image=force_cover_image,
            )
            print(f"  Success: {m_dir.name} -> Total Samples: {samples:,}")
            count += 1
        except Exception as exc:
            print(f"  Error on {m_dir.name}: {exc}")
            logger.exception("Failed to regenerate docs for %s", m_dir.name)

    print(f"Regeneration complete for {count} of {len(active_manifold_dirs)} manifolds.")
    return count


regenerate_all_non_forex = regenerate_all_docs


def main() -> None:
    """Command-line entrypoint for LemGendary Dataset Documentation Orchestrator."""
    parser = argparse.ArgumentParser(description="LemGendary Dataset Doc Orchestrator")
    parser.add_argument("--all", action="store_true", help="Regenerate documentation for all manifolds")
    parser.add_argument("--manifold", type=str, default=None, help="Specific manifold folder name or path")
    parser.add_argument("--force-images", action="store_true", help="Force regenerate cover images")
    args = parser.parse_args()

    cand_root = Path(__file__).resolve().parent.parent.parent.parent / "LemGendaryDatasets"
    datasets_root = cand_root if cand_root.exists() else Path(__file__).resolve().parent.parent.parent / "LemGendaryDatasets"

    if args.manifold:
        m_path = Path(args.manifold)
        if not m_path.exists() and (datasets_root / args.manifold).exists():
            m_path = datasets_root / args.manifold
        generate_dataset_docs(m_path, None, m_path.name, force_cover_image=args.force_images)
        print(f"Successfully generated documentation for {m_path.name}")
    else:
        regenerate_all_docs(datasets_root, force_cover_image=args.force_images)


if __name__ == "__main__":
    main()
