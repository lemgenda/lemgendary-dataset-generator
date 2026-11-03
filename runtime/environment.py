"""
Process-level runtime bootstrap.

Single responsibility: establish a known, deterministic process environment
before any heavy dependency is imported.

Two categories of setup:

1. Environment variables — sourced from the LemGendary Environment Manager's
   ``runtime_env.yaml`` (SSOT). If that file is not present (fresh checkout
   without env-manager cloned), a hardcoded fallback mirroring the YAML is
   used, and a one-time warning is printed.

2. In-process patches — cannot be externalized because they mutate live
   Python objects owned by this process:
       - sys.stdout / sys.stderr UTF-8 reconfigure (Windows)
       - torch.cuda capability check short-circuits (P100 compatibility)

Idempotent: calling ``bootstrap_runtime()`` multiple times has no additional
effect beyond the first invocation.
"""

from __future__ import annotations

import functools
import importlib
import logging
import os
import sys
from pathlib import Path
from types import ModuleType

import yaml
from yaml import YAMLError

logger = logging.getLogger(__name__)

# Fallback values mirror lem-gendary-env-manager/requirements/runtime_env.yaml.
# When the YAML is present, its values take precedence and this dict is
# bypassed. Kept in sync manually; used only when env-manager is absent.
_RUNTIME_ENV_FALLBACK: dict[str, str] = {
    "PYTHONUTF8": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8",
    "FOR_DISABLE_CONSOLE_CTRL_HANDLER": "1",
    "FOR_IGNORE_EXCEPTIONS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_FORCE_PTX_JIT": "1",
    "TORCH_CUDA_ARCH_LIST": "6.0;7.0;7.5;8.0;8.6;9.0",
}

_WINDOWS_ONLY_VARS = frozenset({
    "FOR_DISABLE_CONSOLE_CTRL_HANDLER",
    "FOR_IGNORE_EXCEPTIONS",
})


def _find_runtime_env_yaml() -> Path | None:
    """Locate env-manager's runtime_env.yaml relative to this file.

    Layout assumed:
        model-training/
        ├── lemgendary-datasets/runtime/environment.py     (this file)
        └── lemgendary-env-manager/requirements/runtime_env.yaml
    """
    candidate = (
        Path(__file__).parent.parent.parent
        / "lemgendary-env-manager"
        / "requirements"
        / "runtime_env.yaml"
    )
    return candidate if candidate.exists() else None


def _load_env_from_yaml(yaml_path: Path) -> dict[str, str] | None:
    """Parse runtime_env.yaml into a flat name->value dict.

    Returns None on any read or parse failure (caller falls back to defaults).
    """
    try:
        raw = yaml_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("runtime env YAML %s not readable; using defaults: %s", yaml_path, exc)
        return None

    try:
        data = yaml.safe_load(raw) or {}
    except YAMLError as exc:
        logger.warning("Malformed runtime env YAML %s; using defaults: %s", yaml_path, exc)
        return None

    result: dict[str, str] = {}
    for name, spec in (data.get("variables") or {}).items():
        if isinstance(spec, dict) and "value" in spec:
            result[str(name)] = str(spec["value"])
    return result if result else None


def _apply_env_vars(vars_: dict[str, str]) -> None:
    """Set env vars, scoping Windows-only vars to Windows hosts."""
    is_windows = os.name == "nt"
    for name, value in vars_.items():
        if name in _WINDOWS_ONLY_VARS and not is_windows:
            continue
        os.environ[name] = value


def _apply_stdio_reconfigure() -> None:
    """Force stdout/stderr to UTF-8 on Windows.

    Uses getattr + callable() to guard against alternate stream
    implementations (pytest capture, some IDEs) that lack ``reconfigure``.
    No suppression required.
    """
    if os.name != "nt":
        return
    stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
    stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
    if callable(stdout_reconfigure):
        stdout_reconfigure(encoding="utf-8")
    if callable(stderr_reconfigure):
        stderr_reconfigure(encoding="utf-8")


def _drop_queued_capability_checks(cuda) -> None:
    """Adapter over torch.cuda's private lazy-init queue (no public API exists).

    Removes queued ``_check_capability`` / ``_check_cubins`` calls in place so
    they never run when CUDA lazily initialises. The list object is mutated
    rather than reassigned, so torch keeps the same reference.
    """
    queued = getattr(cuda, "_queued_calls", None)
    if not isinstance(queued, list):
        return
    queued[:] = [
        c for c in queued
        if not (
            isinstance(c, tuple)
            and len(c) > 0
            and getattr(c[0], "__name__", "") in ("_check_capability", "_check_cubins")
        )
    ]


def _apply_torch_patches() -> None:
    """Short-circuit torch.cuda capability checks for P100 compatibility.

    These mutate live objects on the torch module; they cannot be expressed
    as environment variables and must run inside this process. No-op if
    torch is not importable.
    """
    try:
        torch = _import_torch()
    except ImportError as exc:
        logger.debug("torch not importable; skipping CUDA patches: %s", exc)
        return

    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return

    _drop_queued_capability_checks(cuda)

    if hasattr(cuda, "_check_capability"):
        setattr(cuda, "_check_capability", lambda *a, **k: None)
    if hasattr(cuda, "_check_cubins"):
        setattr(cuda, "_check_cubins", lambda *a, **k: None)


@functools.cache
def bootstrap_runtime() -> None:
    """Apply all runtime environment mutations once; later calls are no-ops."""
    yaml_path = _find_runtime_env_yaml()
    if yaml_path is not None:
        env_vars = _load_env_from_yaml(yaml_path)
        if env_vars is not None:
            _apply_env_vars(env_vars)
        else:
            print(
                f"[WARN] Could not parse {yaml_path.name}; "
                f"falling back to hardcoded defaults.",
                file=sys.stderr,
            )
            _apply_env_vars(_RUNTIME_ENV_FALLBACK)
    else:
        print(
            "[WARN] LemGendary Environment Manager not found at "
            "../lemgendary-env-manager/. Using hardcoded runtime env defaults.",
            file=sys.stderr,
        )
        _apply_env_vars(_RUNTIME_ENV_FALLBACK)

    _apply_stdio_reconfigure()
    _apply_torch_patches()


def _import_torch() -> ModuleType:
    """Import torch. The only import site of torch in the compiler."""
    return importlib.import_module("torch")


def load_torch() -> ModuleType:
    """Return the torch module, applying the runtime environment before its first import.

    torch reads TORCH_CUDA_ARCH_LIST, CUDA_FORCE_PTX_JIT and
    PYTORCH_CUDA_ALLOC_CONF when it loads, so every module that needs torch
    gets it from here instead of importing it directly. Raises ImportError
    if torch is not installed.
    """
    bootstrap_runtime()
    return _import_torch()


def get_device_info() -> str:
    """Return a human-readable description of the active compute device."""
    try:
        torch = load_torch()
        if torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(0)})"
    except (ImportError, OSError, RuntimeError, AssertionError) as exc:
        logger.debug("PyTorch device detection unavailable: %s", exc)
    return "CPU (No CUDA detected)"
