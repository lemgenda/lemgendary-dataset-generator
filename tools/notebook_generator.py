"""LemGendary Datasets — Notebook Generator Facade.

Delegates all notebook generation to the modular tools.notebooks engine,
enforcing clean separation of concerns and guaranteeing 100% parity
between dataset compiler notebooks and training suite notebooks.
"""

import argparse
import os
import sys
from typing import Any

from tools.notebooks import (
    generate_colab_inference_notebook,
    generate_colab_training_notebook,
    generate_colab_usage_notebook,
    generate_inference_notebook,
    generate_training_notebook,
    generate_usage_notebook,
    load_registry,
)
from tools.notebooks.cells.env import (
    _build_env_var_lines,
    _load_runtime_env,
)

__all__ = [
    "_load_runtime_env",
    "_build_env_var_lines",
    "generate_inference_notebook",
    "generate_usage_notebook",
    "generate_training_notebook",
    "generate_colab_inference_notebook",
    "generate_colab_usage_notebook",
    "generate_colab_training_notebook",
    "main",
]


def main(argv: list[str] | None = None) -> int:
    """CLI orchestrator for dataset manifold and model notebooks."""
    import yaml

    parser = argparse.ArgumentParser(description="LemGendary Dataset Notebook Orchestrator (v16.2.9 Nuclear)")
    parser.add_argument("--dataset", type=str, help="Dataset key for single notebook generation.")
    parser.add_argument("--model", type=str, help="Model key for single notebook generation.")
    parser.add_argument("--all", action="store_true", help="Regenerate the entire Notebook Matrix for all datasets.")
    parser.add_argument("--output", type=str, help="Override output path (for single) or export root (for all).")
    parser.add_argument("--dir", type=str, help="Alias for --output directory.")
    args = parser.parse_args(argv)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.dirname(base_dir) if os.path.basename(base_dir) == "tools" else base_dir

    models_registry: dict[str, Any] = {}
    for candidate in (
        os.path.join(proj_root, "unified_models_v2.yaml"),
        os.path.join(proj_root, "..", "lemgendary-training-suite", "unified_models_v2.yaml"),
        os.path.join(proj_root, "..", "lemgendary-training-suite", "unified_models.yaml"),
    ):
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        models_registry = loaded
                break
            except OSError:
                pass

    registry_path = os.path.join(proj_root, "unified_data.yaml")
    datasets: dict[str, Any] = {}
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                loaded_ds = yaml.safe_load(f)
                if isinstance(loaded_ds, dict):
                    datasets = loaded_ds.get("datasets", {})
        except OSError:
            pass

    dataset_to_models: dict[str, list[str]] = {
        "nima_aesthetic": ["nima_aesthetic_mobile", "nima_aesthetic_efficientnet", "nima_aesthetic_pro"],
        "classification_master_manifold": ["universal_nsfw_classification"],
        "professional_multitask_restoration": ["professional_multitask_restoration"],
        "forex_universe": ["forex_predictor"],
        "retinaface_mobilenet": ["retinaface"],
    }

    target_output = args.output if args.output else (args.dir if args.dir else None)
    export_root = target_output if target_output else os.path.abspath(os.path.join(proj_root, "../LemGendaryModels"))

    def _emit_for_model(m_key: str, d_info: dict[str, Any]) -> None:
        m_dir = os.path.join(export_root, m_key)
        os.makedirs(m_dir, exist_ok=True)
        generate_inference_notebook(m_key, m_dir, unified_models_registry=models_registry, config=d_info)
        generate_usage_notebook(m_key, m_dir, unified_models_registry=models_registry, config=d_info)
        generate_colab_inference_notebook(m_key, m_dir, unified_models_registry=models_registry, config=d_info)
        generate_colab_usage_notebook(m_key, m_dir, unified_models_registry=models_registry, config=d_info)

    if args.all:
        print(f"[NUCLEAR] Initiating Global Notebook Refresh for {len(datasets)} datasets...")
        for d_key, d_info in datasets.items():
            models = dataset_to_models.get(d_key, [d_key])
            for m_key in models:
                _emit_for_model(m_key, d_info if isinstance(d_info, dict) else {})
        print("\n[SUCCESS] Notebook Matrix Synchronized.")
        return 0

    if args.dataset and args.model:
        d_info = datasets.get(args.dataset, {})
        _emit_for_model(args.model, d_info if isinstance(d_info, dict) else {})
        return 0

    if args.model:
        _emit_for_model(args.model, {})
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())