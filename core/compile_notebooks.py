"""Kaggle and Colab training notebooks written next to every compiled manifold.

The generators live in the sibling ``lemgendary-training-suite`` checkout
(Single Source of Truth); ``core.manifold_compile`` documents the import.
"""

import logging
import re
import sys
from pathlib import Path

from core.lazy_imports import load_training_notebooks

logger = logging.getLogger(__name__)

# Fix-ups from the CamelCase split back to the training suite's model keys.
_MODEL_KEY_FIXES = (("naf_net", "nafnet"), ("upn_v_2", "upn_v2"))


def get_training_suite_notebook_generators():
    """Return the notebook generators from lemgendary-training-suite (Single Source of Truth).

    Equivalent to ``from training.notebooks import generate_training_notebook,
    generate_colab_training_notebook`` once the sibling checkout is on
    ``sys.path``; the import goes through ``core.lazy_imports`` because the
    training suite is an optional sibling repository.
    """
    suite_dir = Path(__file__).resolve().parents[2] / "lemgendary-training-suite"
    if suite_dir.exists() and str(suite_dir) not in sys.path:
        sys.path.insert(0, str(suite_dir))
    notebooks = load_training_notebooks()
    return notebooks.generate_training_notebook, notebooks.generate_colab_training_notebook


def _resolve_model_key(model_key: str, pascal_name: str) -> str:
    """The training-suite model key for a manifold (derived from its name without a key)."""
    resolved_model = model_key
    if not resolved_model:
        clean_name = (
            pascal_name.replace("LemGendized", "")
            .replace("KaggleReady", "")
            .replace("Large", "")
            .replace("Mini", "")
        )
        resolved_model = re.sub(r'(?<!^)(?=[A-Z])', '_', clean_name).lower()

    for wrong, right in _MODEL_KEY_FIXES:
        if wrong in resolved_model:
            resolved_model = resolved_model.replace(wrong, right)
    return resolved_model


def write_manifold_notebooks(model_key: str, pascal_name: str, output_root: Path) -> None:
    """Write the training notebooks of a sample-based manifold; failures are logged."""
    try:
        gen_nb, gen_colab_nb = get_training_suite_notebook_generators()
        resolved_model = _resolve_model_key(model_key, pascal_name)
        gen_nb(
            pascal_name, resolved_model, output_root / f"{resolved_model}_kaggle_training.ipynb"
        )
        gen_colab_nb(
            pascal_name, resolved_model, output_root / f"{resolved_model}_colab_training.ipynb"
        )
    except (ImportError, OSError, ValueError) as e:
        logger.error("Could not generate training notebooks for %s: %s", pascal_name, e)


def write_forex_notebooks(model_key: str, pascal_name: str, output_root: Path) -> None:
    """Write the training notebooks of a forex manifold; failures are logged."""
    try:
        gen_train_nb, gen_colab_nb = get_training_suite_notebook_generators()
        target_model = "forex_predictor" if model_key == "forex_universe" else model_key
        gen_train_nb(
            pascal_name,
            target_model,
            str(output_root / f"{target_model}_kaggle_training.ipynb"),
        )
        gen_colab_nb(
            pascal_name,
            target_model,
            str(output_root / f"{target_model}_colab_training.ipynb"),
        )
    except (ImportError, OSError, ValueError) as e:
        logger.warning("Forex notebook generation skipped: %s", e)
