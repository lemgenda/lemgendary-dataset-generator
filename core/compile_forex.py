"""Forex manifolds: MT5 acquisition through ``forex.pipeline`` instead of image samples."""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from core.compile_notebooks import write_forex_notebooks
from core.compile_outcome import ManifoldOutcome
from core.doc_generator import generate_dataset_docs
from core.error_types import DOC_GENERATION_ERRORS
from core.lazy_imports import load_forex_module

logger = logging.getLogger(__name__)

_DEFAULT_TIMEFRAMES = [1, 5, 15, 60, 240, 1440]


def is_forex(model_config: dict[str, Any]) -> bool:
    """True for manifolds compiled from MT5 time series rather than images."""
    return (
        model_config.get("dataset_type") == "forex"
        or model_config.get("acquisition_mode") == "mt5_terminal"
    )


def _forex_settings(model_config: dict[str, Any]) -> dict[str, Any]:
    """The acquisition settings of a forex manifold, with their defaults."""
    return {
        'pairs': model_config.get('pairs', []),
        'timeframe_rungs': model_config.get('timeframe_rungs', _DEFAULT_TIMEFRAMES),
        'start_date': model_config.get('start_date', '2019-01-01'),
        'lookback_bars': model_config.get('lookback_bars', 168),
    }


def _run_acquisition(full_name: str, settings: dict[str, Any], output_root: Path) -> None:
    """Download the symbols and timeframes through the forex pipeline."""
    print(
        f" -> Engaging MT5 Auto-Acquisition Bridge for {len(settings['pairs'])} symbols across"
        f" {len(settings['timeframe_rungs'])} timeframes..."
    )
    try:
        dataset_defs = [{
            "name": full_name,
            "pairs": settings['pairs'],
            "timeframes": settings['timeframe_rungs'],
            "start_date": settings['start_date'],
        }]

        load_forex_module("pipeline").run_download_pipeline(
            dataset_defs=dataset_defs,
            out_dir=str(output_root.parent),
            login=None, password=None, server=None
        )
    except Exception:
        logger.exception("Temporal compilation of %s dropped", full_name)
        raise


def _write_dataset_info(model_config: dict[str, Any], pascal_name: str,
                        settings: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Write ``dataset_info.yaml`` and return its content."""
    yaml_info = {
        'name': pascal_name,
        'dataset_type': 'forex',
        'category': model_config.get('category', 'Forex & Financial Time-Series'),
        **settings,
        'last_processed': datetime.now().isoformat()
    }
    with open(output_root / "dataset_info.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_info, f, default_flow_style=False, explicit_start=True)
    return yaml_info


def compile_forex_manifold(model_key: str, model_config: dict[str, Any],
                           output_root: Path, full_name: str) -> ManifoldOutcome:
    """Acquire a forex manifold, write its notebooks, info file and docs."""
    pascal_name = model_config.get("name", model_key.replace("_", ""))
    print(f"\n[FOREX MANIFOLD] Compiling Foundation Matrix: {output_root.name}...")
    output_root.mkdir(parents=True, exist_ok=True)

    for empty_dir in [
        output_root / "images",
        output_root / "labels",
        output_root / "masks",
        output_root / "targets",
    ]:
        if empty_dir.exists():
            shutil.rmtree(empty_dir)

    settings = _forex_settings(model_config)
    _run_acquisition(full_name, settings, output_root)
    write_forex_notebooks(model_key, pascal_name, output_root)
    yaml_info = _write_dataset_info(model_config, pascal_name, settings, output_root)

    try:
        generate_dataset_docs(
            output_root, final_index=None, pascal_name=pascal_name, overrides=yaml_info
        )
    except DOC_GENERATION_ERRORS as e:
        logger.warning("Documentation engine skipped for %s: %s", pascal_name, e)

    print(
        "[SUCCESS] Temporal Chunking compilation successfully staged under storage path!\n"
    )
    return ManifoldOutcome(model_key, output_root, samples=0, sample_based=False)
