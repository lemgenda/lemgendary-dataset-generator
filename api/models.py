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
    preset: Optional[str] = Field(None, description="Compiler preset profile name from presets.yaml")
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


class DatasetFormatStats(BaseModel):
    webp: int = 0
    jpg: int = 0
    png: int = 0
    parquet: int = 0
    other: int = 0


class DatasetDetailStats(BaseModel):
    name: str
    path: str
    task: str
    sample_count: int
    size_bytes: int
    size_gb: float
    formats: DatasetFormatStats
    has_hardlinks: bool
    hardlink_ratio: float


class DatasetStatsListResponse(BaseModel):
    datasets: List[DatasetDetailStats]
    total: int
    total_size_bytes: int
    total_size_gb: float


class ActiveJobTelemetry(BaseModel):
    id: str
    job_type: str
    state: str
    created_at: str
    started_at: Optional[str] = None
    progress_percent: float = 0.0
    current_sample: int = 0
    total_samples: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: Optional[float] = None
    fps: float = 0.0


class ActiveJobsResponse(BaseModel):
    active_jobs: List[ActiveJobTelemetry]
    count: int


class CompilerPresetModel(BaseModel):
    name: str
    title: str
    description: str
    image_format: str
    image_quality: int
    target_quality: Optional[int] = None
    mask_format: Optional[str] = None
    min_resolution: Optional[int] = None
    resampling: Optional[str] = None
    vetting_enabled: bool = False
    labeling_enabled: bool = False
    hardlink_gate: bool = True
    containers: List[str] = Field(default_factory=list)


class PresetListResponse(BaseModel):
    presets: Dict[str, CompilerPresetModel]
    total: int


class QuickCompileRequest(BaseModel):
    model: str = Field(..., description="Dataset or model key in unified_data.yaml")
    preset: str = Field("quality-vision", description="Preset name from presets.yaml")
    max_gb: Optional[float] = Field(None, description="Max size override in gigabytes")
    workers: Optional[int] = Field(None, description="Parallel worker process count")


class GuiStateResponse(BaseModel):
    service: str = "LemGendary Dataset Compiler API"
    version: str
    uptime_seconds: float
    active_jobs_count: int
    total_datasets_count: int
    total_storage_bytes: int
    total_storage_gb: float
    hardware: HardwareInfo
    presets: List[str]

