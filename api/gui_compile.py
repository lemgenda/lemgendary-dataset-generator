"""Request-to-job translation for the desktop GUI compile endpoints.

Builds the ``services.compiler_service`` invocations for ``/gui/quick-compile`` and
``/gui/custom-compile``, normalises custom-manifold source URLs into registry
refs and validates / persists the custom manifold, raising the HTTP errors the
routes return.
"""

from __future__ import annotations

import logging
import re
from typing import Any, NamedTuple

from fastapi import HTTPException, status
from pydantic import ValidationError

from api.models import CustomCompileRequest, QuickCompileRequest
from core import presets, unified_registry
from services.compiler_service import compile_invocation
from services.invocation import ServiceInvocation

__all__ = [
    "CustomManifoldNames",
    "build_custom_entry",
    "clean_source_refs",
    "custom_compile_command",
    "custom_compile_parameters",
    "derive_manifold_names",
    "normalize_source_ref",
    "persist_custom_manifold",
    "quick_compile_command",
    "resolve_custom_preset",
]

logger = logging.getLogger("lemgendary.api.routes.gui.compile")

# (URL marker, ref scheme, strip ".git") in match order; drive.google.com keeps the full URL.
_REPO_URL_RULES: tuple[tuple[str, str, bool], ...] = (
    ("kaggle.com/datasets/", "kaggle", False),
    ("huggingface.co/datasets/", "hf", False),
    ("huggingface.co/", "hf", False),
    ("github.com/", "gh", True),
)


def quick_compile_command(req: QuickCompileRequest) -> ServiceInvocation:
    """``services.compiler_service`` invocation for a preset-driven quick compile.

    ``shard_ext`` is forwarded like ``CompileJobRequest.shard_ext``; unset, the
    compiler's own default applies.
    """
    return compile_invocation({
        "model": req.model,
        "preset": req.preset,
        "max_gb": req.max_gb,
        "workers": req.workers,
        "shard_ext": req.shard_ext,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "label_strategy": req.label_strategy,
    })


def resolve_custom_preset(requested: str | None) -> str:
    """Return a preset name that exists in presets.yaml (the single source of truth).

    An omitted preset resolves to presets.yaml `default_preset`; an unknown
    preset is rejected with 422 and the list of valid names.
    """
    try:
        available = sorted(presets.list_presets())
        name = requested if requested else presets.get_default_preset_name()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Compiler presets are unusable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compiler presets are unusable: {exc}",
        ) from exc
    if name not in available:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown compiler preset '{name}'. Valid presets: {', '.join(available)}",
        )
    return name


def persist_custom_manifold(model_key: str, entry: dict[str, Any]) -> None:
    """Add the custom manifold to unified_data.yaml or raise the matching HTTP error."""
    try:
        unified_registry.add_dataset_entry(model_key, entry)
    except unified_registry.RegistryEntryExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Custom manifold '{model_key}' fails the registry schema: {exc}",
        ) from exc
    except (OSError, unified_registry.RegistryWriteError) as exc:
        logger.error("Failed persisting custom manifold '%s'; job not started: %s", model_key, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not write unified_data.yaml; job not started: {exc}",
        ) from exc


class CustomManifoldNames(NamedTuple):
    """PascalCase name, snake_case registry key and output folder of a custom manifold."""

    pascal_name: str
    model_key: str
    modernized_folder: str


def derive_manifold_names(custom_name: str) -> CustomManifoldNames:
    """Derive the registry names for a (stripped, non-empty) custom dataset name."""
    pascal = "".join(w.capitalize() for w in re.split(r"[^a-zA-Z0-9]", custom_name) if w)
    pascal_name = pascal or "CustomManifold"
    return CustomManifoldNames(
        pascal_name=pascal_name,
        model_key=re.sub(r"(?<!^)(?=[A-Z])", "_", pascal_name).lower(),
        modernized_folder=f"LemGendized{pascal_name}",
    )


def normalize_source_ref(src: str) -> str:
    """Turn a Kaggle / Hugging Face / GitHub / Google Drive URL into a registry ref."""
    for marker, scheme, strip_git in _REPO_URL_RULES:
        if marker in src:
            repo = src.split(marker)[-1].split("?")[0]
            if strip_git:
                repo = repo.replace(".git", "")
            return f"{scheme}://{repo.strip('/')}"
    if "drive.google.com" in src:
        return f"gd://{src}"
    return src


def clean_source_refs(sources: list[str]) -> list[str]:
    """Normalised refs for every non-blank source entry, in request order."""
    return [normalize_source_ref(s.strip()) for s in sources if s.strip()]


def build_custom_entry(
    names: CustomManifoldNames, req: CustomCompileRequest, cleaned_refs: list[str]
) -> dict[str, Any]:
    """The unified_data.yaml dataset entry for a custom manifold."""
    return {
        "name": names.pascal_name,
        "title": f"LemGendized {names.pascal_name}",
        "canonical_format": req.canonical_format,
        "task": req.task,
        "modernized_folder": names.modernized_folder,
        "refs": [{"ref": r} for r in cleaned_refs],
        "container": {"primary": req.canonical_format},
    }


def custom_compile_command(
    model_key: str, preset_name: str, purge_loose_images: bool
) -> ServiceInvocation:
    """``services.compiler_service`` invocation for a freshly registered custom manifold."""
    return compile_invocation(
        {"model": model_key, "preset": preset_name, "cleanup": purge_loose_images}
    )


def custom_compile_parameters(
    req: CustomCompileRequest,
    names: CustomManifoldNames,
    preset_name: str,
    cleaned_refs: list[str],
) -> dict[str, Any]:
    """Job parameters recorded for a custom compile."""
    return {
        "custom_name": req.custom_name,
        "model_key": names.model_key,
        "pascal_name": names.pascal_name,
        "preset": preset_name,
        "canonical_format": req.canonical_format,
        "shard_size": req.shard_size,
        "sources": cleaned_refs,
        "purge_loose_images": req.purge_loose_images,
    }
