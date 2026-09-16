"""
Process-level runtime bootstrap.

Consolidates the environment mutations previously scattered at the top of
compiler_core.py. Call ``bootstrap_runtime()`` once from any entry point
that needs the following guarantees:

  * stdout/stderr forced to UTF-8 on Windows (prevents UnicodeEncodeError)
  * Fortran runtime error dialogs disabled (FOR_DISABLE_CONSOLE_CTRL_HANDLER)
  * PyTorch CUDA JIT enabled with an expanded arch list (Pascal sm_60 support)
  * Torch CUDA capability checks short-circuited (fixes P100 on cu121)

Idempotent: calling it multiple times has no additional effect.

Zero Suppressions Policy (Phase 1.5.7): this module contains no
`# type: ignore` comments. Dynamic stdio method resolution uses
``getattr(..., None)`` + ``callable()``, which is type-safe without
suppression.
"""

from __future__ import annotations

import os
import sys

_BOOTSTRAPPED = False


def bootstrap_runtime() -> None:
    """Apply all runtime environment mutations. Safe to call repeatedly."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    # ── UTF-8 stdio on Windows ──────────────────────────────────────────
    # ``sys.stdout.reconfigure`` is a TextIOWrapper method unavailable on
    # some alternate stream implementations (e.g. pytest capture, certain
    # IDEs). Resolve it dynamically and call it if present. Using
    # getattr()/callable() avoids the need for # type: ignore[attr-defined].
    if os.name == "nt":
        stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
        stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
        if callable(stdout_reconfigure):
            stdout_reconfigure(encoding="utf-8")
        if callable(stderr_reconfigure):
            stderr_reconfigure(encoding="utf-8")
        os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
        os.environ["FOR_IGNORE_EXCEPTIONS"] = "1"

    # ── CUDA JIT + arch targets ────────────────────────────────────────
    os.environ["CUDA_FORCE_PTX_JIT"] = "1"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;7.0;7.5;8.0;8.6;9.0"

    # ── Torch CUDA capability guard (deferred import — torch is heavy) ──
    try:
        import torch
    except ImportError:
        _BOOTSTRAPPED = True
        return

    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        # Filter queued capability checks — the CUDA runtime shims that
        # trigger these are not compatible with older arch lists.
        queued = getattr(cuda, "_queued_calls", None)
        if isinstance(queued, list):
            cuda._queued_calls = [
                c for c in queued
                if getattr(getattr(c, "__getitem__", lambda _: None), "__name__", "") not in
                   ("_check_capability", "_check_cubins")
                if not (isinstance(c, tuple) and len(c) > 0 and
                        getattr(c[0], "__name__", "") in ("_check_capability", "_check_cubins"))
            ]
        # Bypass the capability / cubin checks entirely on P100 hardware.
        if hasattr(cuda, "_check_capability"):
            setattr(cuda, "_check_capability", lambda *a, **k: None)
        if hasattr(cuda, "_check_cubins"):
            setattr(cuda, "_check_cubins", lambda *a, **k: None)

    _BOOTSTRAPPED = True


def get_device_info() -> str:
    """Return a human-readable description of the active compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(0)})"
    except ImportError:
        pass
    return "CPU (No CUDA detected)"