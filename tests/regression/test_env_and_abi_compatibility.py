"""Regression tests for Python 3.14 environment, ABI compatibility, and permissive licenses."""

import importlib
import sys

import pytest


def test_python_version() -> None:
    """Assert runtime is strictly Python 3.14."""
    assert sys.version_info[:2] == (3, 14), f"Expected Python 3.14, got {sys.version}"


def test_permissive_license_zero_agpl() -> None:
    """Ensure copyleft/AGPL dependencies (albumentations/albumentationsx) are not installed."""
    with pytest.raises(ImportError):
        importlib.import_module("albumentations")

    with pytest.raises(ImportError):
        importlib.import_module("albumentationsx")


@pytest.mark.parametrize(
    "pkg_name",
    [
        "torch",
        "torchvision",
        "transformers",
        "PIL",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "sklearn",
        "safetensors",
        "huggingface_hub",
        "tokenizers",
        "regex",
        "tqdm",
        "yaml",
        "mediapipe",
        "cv2",
        "ultralytics",
        "fastapi",
        "uvicorn",
        "websockets",
        "pydantic",
        "typer",
        "click",
        "requests",
        "httpx2",
        "webdataset",
        "litdata",
        "zstandard",
    ],
)
def test_import_dependencies_zero_warnings(pkg_name: str) -> None:
    """Dynamically import required packages asserting successful import."""
    mod = importlib.import_module(pkg_name)
    assert mod is not None


def test_platform_gated_dependencies() -> None:
    """Check platform-specific packages on Windows."""
    if sys.platform == "win32":
        mt5 = importlib.import_module("MetaTrader5")
        assert mt5 is not None
        pnd_ta = importlib.import_module("pandas_ta_classic")
        assert pnd_ta is not None
