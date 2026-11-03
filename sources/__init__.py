"""
LemGendary Dataset Compiler — Source Fetchers Package

Holds the four fetch backends:

    sources/hf.py      Hugging Face
    sources/gh.py      GitHub
    sources/gd.py      Google Drive
    sources/kaggle.py  Kaggle

Each submodule is a standalone CLI with a `main()` entry point, run as
``python -m sources.<name>`` (the hub PowerShell script does this in its
ScriptBlocks $HuggingFaceSB / $GHSourceSB / $GDriveSB / $DownloadSB).
The package imports none of them, so starting one fetcher does not load the
dependencies of the other three.
"""
