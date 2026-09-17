"""Pydantic data models for LemGendary Dataset Compiler API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    COMPILE = "compile"
    DEGRADE = "degrade"
    REDUCE = "reduce"
    MODERNIZE = "modernize"
    LABEL = "label"
    PROMPT = "prompt"
    MASK = "mask"
    AUDIT = "audit"
    SYNC = "sync"
    CUSTOM = "custom"


class CompileJobRequest(BaseModel):
    model: Optional[str] = Field(None, description="Dataset/model key in unified_data.yaml")
    max_gb: Optional[float] = Field(None, description="Max size override in gigabytes")
    suffix: Optional[str] = Field(None, description="Manifold name suffix override")
    workers: Optional[int] = Field(None, description="Parallel worker process count")
    no_vetting: bool = Field(False, description="Bypass NIMA aesthetic vetting gate")
    no_labeling: bool = Field(False, description="Bypass YOLO auto-labeling")
    no_hash: bool = Field(False, description="Bypass deduplication hash")
    image_format: Optional[str] = Field("webp", description="Target image format (webp, jpeg, png, keep)")
    image_quality: Optional[int] = Field(92, description="Target image quality (1-100)")
    target_quality: Optional[int] = Field(95, description="Target quality for restoration ground truth")
    mask_format: Optional[str] = Field("webp-lossless", description="Target mask format")
    also_format: Optional[str] = Field(None, description="Comma-separated container formats (mds, litdata, wds, parquet)")
    force_duplicate: bool = Field(False, description="Bypass hardlink gate and duplicate bytes")
    accept_space_loss: bool = Field(False, description="Accept storage growth for container writes")
    label_strategy: Optional[str] = Field(None, description="Auto-labeling strategy")
    prompt_strategy: Optional[str] = Field(None, description="Diffusion prompt template")
    mask_strategy: Optional[str] = Field(None, description="Segmentation mask strategy")


class DegradeJobRequest(BaseModel):
    source: str = Field(..., description="Source clean dataset path or registered manifold name")
    output: str = Field(..., description="Destination synthetic manifold folder name")
    profile: str = Field("motion-blur+iso-noise", description="Degradation profile expression or preset alias")
    intensity: str = Field("medium", description="Intensity preset (low, medium, high)")
    pairs: Optional[int] = Field(None, description="Maximum sample pairs to synthesize")
    val_split: float = Field(0.12, description="Validation split ratio")
    seed: int = Field(42, description="Deterministic seed")
    image_format: str = Field("webp", description="Format for degraded images (webp, jpeg, png)")
    workers: Optional[int] = Field(None, description="Worker thread count")
    dry_run: bool = Field(False, description="Simulate without writing files")


class GenericJobRequest(BaseModel):
    job_type: JobType = Field(JobType.CUSTOM, description="Type of job to execute")
    command: List[str] = Field(..., description="Command and argument list")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")


class JobResponse(BaseModel):
    id: str = Field(..., description="Unique job identifier (UUID)")
    job_type: JobType = Field(..., description="Job category")
    state: JobState = Field(..., description="Current job lifecycle state")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    started_at: Optional[str] = Field(None, description="ISO 8601 execution start timestamp")
    completed_at: Optional[str] = Field(None, description="ISO 8601 completion timestamp")
    exit_code: Optional[int] = Field(None, description="Process exit code")
    error_message: Optional[str] = Field(None, description="Error details if failed")
    log_file: str = Field(..., description="Relative or absolute path to execution log file")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Job configuration parameters")


class JobListResponse(BaseModel):
    jobs: List[JobResponse]
    total: int


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "LemGendary Dataset Compiler API"
    version: str
    uptime_seconds: float
    active_jobs: int


class HardwareInfo(BaseModel):
    cpu_count: int
    ram_total_gb: float
    ram_available_gb: float
    cuda_available: bool
    device_name: str
    active_workers: int


class FullHealthResponse(BaseModel):
    status: str
    service: str
    version: str
    uptime_seconds: float
    active_jobs: int
    hardware: HardwareInfo


class DatasetInfoResponse(BaseModel):
    name: str
    path: str
    task: str
    sample_count: int
    size_bytes: int
    formats: List[str]
    has_hardlinks: bool
    hardlink_ratio: float


class DatasetListResponse(BaseModel):
    datasets: List[DatasetInfoResponse]
    total: int


class EnvPassthroughResponse(BaseModel):
    status: str
    returncode: int
    output: str
