"""
LemGendary Dataset Compiler — Source Fetchers Package

Consolidates the four fetch backends previously scattered at the repo root:

    hf_manager.py      ->  sources/hf.py
    gh_manager.py      ->  sources/gh.py
    gd_manager.py      ->  sources/gd.py
    kaggle_manager.py  ->  sources/kaggle.py

Each submodule is CLI-invocable and exposes a `main()` entry point. The
hub PowerShell script invokes them by full path (see lem_gendary_datasets_hub.ps1
ScriptBlocks $HuggingFaceSB / $GHSourceSB / $GDriveSB / $DownloadSB).

Phase 1.2 of the 2026 modernization roadmap.

The pre-1.2 top-level managers (hf_manager.py, gh_manager.py, gd_manager.py,
kaggle_manager.py) were removed in Phase 1.5.7 after verifying that every
caller had migrated to `sources.X`.
"""

from __future__ import annotations

__all__ = ["hf", "gh", "gd", "kaggle", "base"]

from . import base
from . import hf
from . import gh
from . import gd
from . import kaggle