"""Compiler configuration: CONFIG defaults, unified_data.yaml metadata and CLI state.

Owns the process-wide settings every compiler stage reads. ``CONFIG`` is a
mutable dict shared by reference, so overrides applied by ``configure()``
are visible to every module that imported it.
"""

import argparse
import json
import logging
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.cli_args import build_parser
from core.config_schema import UnifiedData, load_unified_data

logger = logging.getLogger(__name__)


# ─── Config ─────────────────────────────────────────────────────────────────
CONFIG_PATH = Path("./config.json")

DEFAULT_CONFIG: dict[str, Any] = {
    "train_split": 0.8,
    "num_workers": max(1, multiprocessing.cpu_count() - 2),
    "diffusion_size": 1024,
    "black_threshold": 0.1,
    "nima_threshold": 4.0,
    "enable_dedup": False,
    "enable_perceptual_dedup": False,   # Phase 2
    "audit_debug_rejects": False,       # Phase 2
    "strict_ground_truth": False,
    "image_format": "webp",             # Phase 3
    "image_quality": 92,                # Phase 3
    "target_quality": 95,               # Phase 3
    "mask_format": "webp-lossless",     # Phase 3
    "label_strategy": None,             # Phase 5 — None => infer from task
    "prompt_strategy": None,            # Phase 5
    "mask_strategy": None,              # Phase 5 — None => infer from task
}

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
        CONFIG: dict[str, Any] = {**DEFAULT_CONFIG, **json.load(_f)}
else:
    CONFIG = dict(DEFAULT_CONFIG)


ROOT = (
    Path(__file__).resolve().parent.parent
    if Path(__file__).parent.name == "core"
    else Path(__file__).resolve().parent
)

try:
    _manifest_path = ROOT / "unified_data.yaml"
    if not _manifest_path.exists():
        _manifest_path = Path("./unified_data.yaml")
    _UNIFIED: UnifiedData = load_unified_data(_manifest_path)
    YAML_DATA: dict[str, Any] = _UNIFIED.to_legacy_dict()
except FileNotFoundError as e:
    print(f"[FATAL] {e}")
    sys.exit(3)
except (ValueError, yaml.YAMLError) as e:
    logger.error("unified_data.yaml failed schema validation: %s", e)
    print("[FATAL] unified_data.yaml failed schema validation:")
    print(f"  {type(e).__name__}: {e}")
    print("[REMEDY] Run `python -m core.config_schema` to see detailed field diagnostics.")
    sys.exit(2)

META: dict[str, Any] = YAML_DATA.get("_registry_metadata", {})
VERSION: str = META.get("version", "4.2.0")


# ─── CLI configuration (Phase 1.5: SSOT in cli_args.py) ─────────────────────
# NOTE: Nothing here runs at import time. The entry point must call
# configure() explicitly before any worker function is invoked.
@dataclass
class _CliState:
    """Holds the Namespace installed by configure() or set_args()."""

    args: argparse.Namespace | None = None


_CLI = _CliState()


def configure(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and apply CONFIG overrides.

    Must be called once from the process entry point (typically
    manifold_compile.py) BEFORE any worker function or `get_args()` use.

    Calling this more than once is safe but overwrites the previous
    Namespace and re-applies CLI overrides to CONFIG (idempotent for the
    same argv).
    """
    return install_args(build_parser().parse_args(argv))


def install_args(ns: argparse.Namespace) -> argparse.Namespace:
    """Make ``ns`` the active Namespace and apply its CONFIG overrides.

    For entry points that parse the CLI themselves (``core.manifold_compile``
    applies preset defaults first, ``services.run_compile`` builds argv from typed
    options): without this call ``get_args()`` raises in the workers and
    ``--workers`` / ``--image-format`` / strategy flags never reach CONFIG.
    """
    _CLI.args = ns
    _apply_cli_overrides(ns)
    return ns


def set_args(ns: argparse.Namespace | None) -> None:
    """Inject a pre-parsed Namespace (``None`` clears it) without touching CONFIG.

    Used by spawned worker processes (Windows spawn) where sys.argv in the
    child is not the parent's argv. The parent should pass
    `get_args()` as the 4th element of `initargs` to the Pool.
    """
    _CLI.args = ns


def get_args() -> argparse.Namespace:
    """Return the active Namespace, raising if the module isn't configured."""
    if _CLI.args is None:
        raise RuntimeError(
            "compiler_core has not been configured. Call "
            "compiler_core.configure() from your entry point before using "
            "worker functions, or pass args=... to init_worker() in spawned "
            "worker processes."
        )
    return _CLI.args


def _apply_cli_overrides(ns: argparse.Namespace) -> None:
    """Apply CLI flags to CONFIG.

    Extracted verbatim from the previously module-level block; only the
    trigger point moved.
    """
    if getattr(ns, "workers", None):
        CONFIG["num_workers"] = ns.workers

    # Phase 3: CLI overrides for transcode policy.
    if getattr(ns, "image_format", None):
        CONFIG["image_format"] = ns.image_format
    if getattr(ns, "image_quality", None) is not None:
        CONFIG["image_quality"] = ns.image_quality
    if getattr(ns, "target_quality", None) is not None:
        CONFIG["target_quality"] = ns.target_quality
    if getattr(ns, "mask_format", None):
        CONFIG["mask_format"] = ns.mask_format

    # Phase 5: CLI overrides for generation strategies.
    if getattr(ns, "label_strategy", None):
        CONFIG["label_strategy"] = ns.label_strategy
    if getattr(ns, "prompt_strategy", None):
        CONFIG["prompt_strategy"] = ns.prompt_strategy
    if getattr(ns, "mask_strategy", None):
        CONFIG["mask_strategy"] = ns.mask_strategy




# Static paths. None of these depend on CLI args — they were already
# constants before the refactor.
INPUT_ROOT = Path("./raw-sets")
OUT_PARENT = Path(META.get("output_folder_name", "../LemGendaryDatasets"))
CATEGORY_MAP_PATH = ROOT / "converters" / "category_map.json"
if not CATEGORY_MAP_PATH.exists():
    CATEGORY_MAP_PATH = ROOT / "category_map.json"

if CATEGORY_MAP_PATH.exists():
    with open(CATEGORY_MAP_PATH, "r", encoding="utf-8") as _f:
        CATEGORY_MAP: dict[str, int] = json.load(_f)
else:
    CATEGORY_MAP = {}

DATASETS_META: dict[str, Any] = YAML_DATA.get("datasets", {})
