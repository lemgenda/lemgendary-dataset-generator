"""Loaders for heavy or optional dependencies.

Single responsibility: every package that is too slow to import eagerly, is
optional in ``requirements.txt`` or must not load before
``bootstrap_runtime()`` has run is imported here and nowhere else.

Each loader imports through ``importlib.import_module`` when it is called, so
importing a caller never pulls in pandas, pyarrow, kaggle, mediapipe,
mosaicml-streaming or the torch-backed model modules. A missing package
raises ``ImportError`` from the loader, which callers already handle.

torch itself is loaded by ``runtime.environment.load_torch()``.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, NamedTuple

from runtime.environment import bootstrap_runtime


class MediaPipeTasks(NamedTuple):
    """The mediapipe modules needed to build a face landmarker."""

    mp: ModuleType
    python: ModuleType
    vision: ModuleType


def load_pandas() -> ModuleType:
    """Return the ``pandas`` module."""
    return importlib.import_module("pandas")


def load_parquet() -> ModuleType:
    """Return the ``pyarrow.parquet`` module."""
    return importlib.import_module("pyarrow.parquet")


def load_kaggle_api() -> Any:
    """Return the ``KaggleApi`` class.

    ``import kaggle`` authenticates on import and raises ``OSError`` when no
    credentials exist, so it is deferred until a Kaggle call is made.
    """
    module = importlib.import_module("kaggle.api.kaggle_api_extended")
    return module.KaggleApi


def load_mediapipe() -> MediaPipeTasks:
    """Return the mediapipe package with its ``tasks.python`` and ``vision`` modules."""
    return MediaPipeTasks(
        mp=importlib.import_module("mediapipe"),
        python=importlib.import_module("mediapipe.tasks.python"),
        vision=importlib.import_module("mediapipe.tasks.python.vision"),
    )


def load_streaming() -> ModuleType:
    """Return the ``streaming`` (mosaicml-streaming) module."""
    return importlib.import_module("streaming")


def load_model_module(name: str) -> ModuleType:
    """Return ``models.<name>`` after the runtime environment is applied.

    The model modules import torch, transformers or ultralytics at module
    level; torch reads its CUDA settings when it loads, so the environment
    must be bootstrapped first.
    """
    bootstrap_runtime()
    return importlib.import_module(f"models.{name}")


def load_forex_module(name: str) -> ModuleType:
    """Return ``forex.<name>``; the forex package imports pyarrow and pandas."""
    return importlib.import_module(f"forex.{name}")


def load_manifold_compile() -> ModuleType:
    """Return ``core.manifold_compile`` (imports pandas, sklearn and webdataset)."""
    bootstrap_runtime()
    return importlib.import_module("core.manifold_compile")


def load_manifold_reduce() -> ModuleType:
    """Return ``tools.manifold_reduce`` (imports the compile stack)."""
    bootstrap_runtime()
    return importlib.import_module("tools.manifold_reduce")


def load_training_notebooks() -> ModuleType:
    """Return ``training.notebooks`` from lemgendary-training-suite.

    The caller puts the training-suite checkout on ``sys.path`` first.
    """
    return importlib.import_module("training.notebooks")
