"""FastAPI route handlers for LLM dataset compilation and validation."""

from __future__ import annotations

import json
import logging
import unittest
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from api.jobs import job_manager, submit_job
from api.models import JobType
from api.routes.datasets import _resolve_output_root
from services.llm_service import compile_llm_manifold, llm_compile_invocation
from tools.llm_helper.quality_validator import QualityValidator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["llm"])


class LLMCompileRequest(BaseModel):
    """Request payload for compiling an LLM instruction dataset."""

    dataset_key: str = Field(
        ...,
        description="Target dataset key from unified_data.yaml (e.g. test_llm, ai_helper_lemgend)",
    )
    target_samples: int | None = Field(None, description="Override target sample count")
    async_mode: bool = Field(False, description="Run compilation as a background worker job")


class LLMValidationRequest(BaseModel):
    """Request payload for validating an LLM dataset JSONL file."""
    file_path: str | None = Field(
        None,
        description=(
            "Path to a .jsonl file inside the dataset root, preferably relative to it "
            "(e.g. ai_helper_lemgend/data/ai_helper_lemgend_train.jsonl). Absolute paths "
            "are accepted only when they resolve inside the dataset root."
        ),
    )
    records: list[dict[str, Any]] | None = Field(
        None, description="Inline list of records to validate"
    )


@router.post("/compile")
async def compile_llm_dataset(req: LLMCompileRequest) -> dict[str, Any]:
    """Compile an LLM instruction manifold using BaseLLMCompiler and unified_data.yaml.

    The route is ``async`` so the job runner receives the server's running
    event loop (its log broadcasts are scheduled on it). The synchronous
    compile path runs in a worker thread so it never blocks that loop.
    """
    if req.async_mode:
        job = submit_job(
            job_manager,
            JobType.LLM,
            llm_compile_invocation(req.dataset_key),
            {"dataset_key": req.dataset_key, "target_samples": req.target_samples},
        )
        return {
            "status": "enqueued",
            "job_id": job.id,
            "dataset_key": req.dataset_key,
            "message": f"LLM compilation enqueued for {req.dataset_key}",
        }

    try:
        result = await run_in_threadpool(compile_llm_manifold, req.dataset_key)
    except (OSError, KeyError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        logger.exception("LLM manifold compile failed for %s", req.dataset_key)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to compile LLM manifold: {exc}",
        ) from exc
    return {
        "status": "success",
        "dataset_key": req.dataset_key,
        "details": result,
    }


def resolve_corpus_path(raw_path: str) -> Path:
    """Resolve ``raw_path`` to a file that lies inside the dataset root.

    Relative paths are taken relative to the dataset root. The path is fully
    resolved (``..`` collapsed, symlinks followed) before the containment check,
    so traversal and symlinks pointing outside the root are refused.

    Raises:
        HTTPException: 403 when the path resolves outside the dataset root,
            404 when it is inside the root but is not an existing file.
    """
    root = _resolve_output_root().resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        logger.warning("Refused LLM validate path outside the dataset root: %s", raw_path)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Path is outside the dataset root and cannot be validated: {raw_path}",
        )
    if not resolved.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"File not found: {raw_path}"
        )
    return resolved


@router.post("/validate")
def validate_llm_corpus(req: LLMValidationRequest) -> dict[str, Any]:
    """Run QualityValidator on an LLM dataset file inside the dataset root or on inline records."""
    records: list[dict[str, Any]] = []

    if req.records is not None:
        records = req.records
    elif req.file_path is not None:
        corpus_path = resolve_corpus_path(req.file_path)
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse JSONL file: {exc}"
            ) from exc
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either file_path or records",
        )

    validator = QualityValidator()
    is_valid, issues = validator.validate_corpus(records)

    return {
        "status": "valid" if is_valid else "invalid",
        "total_records": len(records),
        "issues_count": len(issues),
        "issues": [
            {
                "record_id": iss.record_id,
                "field": iss.field,
                "severity": iss.severity,
                "issue": iss.issue,
            }
            for iss in issues
        ],
    }


class DatasetTestSuiteRequest(BaseModel):
    """Payload for triggering in-process test discovery sweep."""
    pattern: str = Field("test_*.py", description="Test discovery file pattern")


@router.post("/test-suite")
def run_dataset_tests(req: DatasetTestSuiteRequest) -> dict[str, Any]:
    """Execute dataset test suite in-process and return structured test outcome."""
    project_root = Path(__file__).resolve().parent.parent.parent
    tests_dir = project_root / "tests"
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern=req.pattern)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    return {
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "failures_count": len(result.failures),
        "errors_count": len(result.errors),
        "failures": [str(f[0]) for f in result.failures],
        "errors": [str(e[0]) for e in result.errors],
    }
