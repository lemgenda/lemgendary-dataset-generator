"""Pydantic data models for LemGendary Dataset Compiler API."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobState(str, Enum):
    """Lifecycle state of a background job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Category of work a background job performs."""
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
    LLM = "llm"


class CompileJobRequest(BaseModel):
    """Request body for a dataset compile job."""
    model: str | None = Field(None, description="Dataset/model key in unified_data.yaml")
    preset: str | None = Field(
        None, description="Compiler preset profile name from presets.yaml"
    )
    max_gb: float | None = Field(None, description="Max size override in gigabytes")
    suffix: str | None = Field(None, description="Manifold name suffix override")
    workers: int | None = Field(None, description="Parallel worker process count")
    no_vetting: bool = Field(False, description="Bypass NIMA aesthetic vetting gate")
    no_labeling: bool = Field(False, description="Bypass YOLO auto-labeling")
    no_hash: bool = Field(False, description="Bypass deduplication hash")
    image_format: str | None = Field(
        "webp", description="Target image format (webp, jpeg, png, keep)"
    )
    image_quality: int | None = Field(92, description="Target image quality (1-100)")
    target_quality: int | None = Field(
        95, description="Target quality for restoration ground truth"
    )
    mask_format: str | None = Field("webp-lossless", description="Target mask format")
    also_format: str | None = Field(
        None, description="Comma-separated container formats (mds, litdata, wds, parquet)"
    )
    force_duplicate: bool = Field(False, description="Bypass hardlink gate and duplicate bytes")
    accept_space_loss: bool = Field(False, description="Accept storage growth for container writes")
    label_strategy: str | None = Field(None, description="Auto-labeling strategy")
    prompt_strategy: str | None = Field(None, description="Diffusion prompt template")
    mask_strategy: str | None = Field(None, description="Segmentation mask strategy")
    shard_ext: str | None = Field(
        "data", description="WebDataset shard file extension without dot (default: 'data')"
    )


class DegradeJobRequest(BaseModel):
    """Request body for a degradation (synthetic pair) job."""
    source: str = Field(..., description="Source clean dataset path or registered manifold name")
    output: str = Field(..., description="Destination synthetic manifold folder name")
    profile: str = Field(
        "motion-blur+iso-noise", description="Degradation profile expression or preset alias"
    )
    intensity: str = Field("medium", description="Intensity preset (low, medium, high)")
    pairs: int | None = Field(None, description="Maximum sample pairs to synthesize")
    val_split: float = Field(0.12, description="Validation split ratio")
    seed: int = Field(42, description="Deterministic seed")
    image_format: str = Field("webp", description="Format for degraded images (webp, jpeg, png)")
    workers: int | None = Field(None, description="Worker thread count")
    dry_run: bool = Field(False, description="Simulate without writing files")


class GenericJobRequest(BaseModel):
    """Request body for a job that runs an arbitrary command."""
    job_type: JobType = Field(JobType.CUSTOM, description="Type of job to execute")
    command: list[str] = Field(..., description="Command and argument list")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")


class JobResponse(BaseModel):
    """A background job and its current state."""
    id: str = Field(..., description="Unique job identifier (UUID)")
    job_type: JobType = Field(..., description="Job category")
    state: JobState = Field(..., description="Current job lifecycle state")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    started_at: str | None = Field(None, description="ISO 8601 execution start timestamp")
    completed_at: str | None = Field(None, description="ISO 8601 completion timestamp")
    exit_code: int | None = Field(None, description="Process exit code")
    error_message: str | None = Field(None, description="Error details if failed")
    log_file: str = Field(..., description="Relative or absolute path to execution log file")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Job configuration parameters"
    )


class JobListResponse(BaseModel):
    """A list of jobs with their total count."""
    jobs: list[JobResponse]
    total: int


class HealthResponse(BaseModel):
    """Basic liveness response."""
    status: str = "ok"
    service: str = "LemGendary Dataset Compiler API"
    version: str
    uptime_seconds: float
    active_jobs: int


class HardwareInfo(BaseModel):
    """Host CPU, memory and CUDA details."""
    cpu_count: int
    ram_total_gb: float
    ram_available_gb: float
    cuda_available: bool
    device_name: str
    active_workers: int


class FullHealthResponse(BaseModel):
    """Liveness response with host hardware details."""
    status: str
    service: str
    version: str
    uptime_seconds: float
    active_jobs: int
    hardware: HardwareInfo


class DatasetInfoResponse(BaseModel):
    """Summary of one compiled dataset on disk."""
    name: str
    path: str
    task: str
    sample_count: int
    size_bytes: int
    formats: list[str]
    has_hardlinks: bool
    hardlink_ratio: float


class DatasetListResponse(BaseModel):
    """A list of compiled datasets with their total count."""
    datasets: list[DatasetInfoResponse]
    total: int


class EnvPassthroughResponse(BaseModel):
    """Result of a command passed through to the environment manager."""
    status: str
    returncode: int
    output: str


class DatasetFormatStats(BaseModel):
    """Per-format file counts for a dataset."""
    webp: int = 0
    jpg: int = 0
    png: int = 0
    parquet: int = 0
    other: int = 0


class DatasetSourceInfo(BaseModel):
    """One upstream source a dataset was compiled from."""
    name: str
    ref: str | None = None
    count: int | None = None
    type: str | None = "SOURCE"


class DatasetDetailStats(BaseModel):
    """Detailed statistics for one compiled dataset."""
    name: str
    key: str | None = None
    display_name: str | None = None
    path: str
    task: str
    sample_count: int
    size_bytes: int
    size_gb: float
    format: str | None = "directory"
    canonical_format: str | None = "webdataset"
    formats: DatasetFormatStats
    shards_count: int = 0
    is_compiled: bool = False
    has_hardlinks: bool
    hardlink_ratio: float
    sources: list[DatasetSourceInfo] = []
    kaggle_ref: str | None = None


class DatasetStatsListResponse(BaseModel):
    """Detailed statistics for all compiled datasets with size totals."""
    datasets: list[DatasetDetailStats]
    total: int
    total_size_bytes: int
    total_size_gb: float


class ActiveJobTelemetry(BaseModel):
    """Progress telemetry for one running job."""
    id: str
    job_type: str
    state: str
    created_at: str
    started_at: str | None = None
    progress_percent: float = 0.0
    current_sample: int = 0
    total_samples: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    fps: float = 0.0


class ActiveJobsResponse(BaseModel):
    """The running jobs with their count."""
    active_jobs: list[ActiveJobTelemetry]
    count: int


class CompilerPresetModel(BaseModel):
    """A compiler preset as exposed by the API."""
    name: str
    title: str
    description: str
    image_format: str
    image_quality: int
    target_quality: int | None = None
    mask_format: str | None = None
    min_resolution: int | None = None
    resampling: str | None = None
    vetting_enabled: bool = False
    labeling_enabled: bool = False
    hardlink_gate: bool = True
    containers: list[str] = Field(default_factory=list)


class PresetListResponse(BaseModel):
    """All compiler presets keyed by name."""
    presets: dict[str, CompilerPresetModel]
    total: int


class QuickCompileRequest(BaseModel):
    """Request body for compiling a dataset with a preset."""
    model: str = Field(..., description="Dataset or model key in unified_data.yaml")
    preset: str = Field("quality-vision", description="Preset name from presets.yaml")
    max_gb: float | None = Field(None, description="Max size override in gigabytes")
    workers: int | None = Field(None, description="Parallel worker process count")


class GuiStateResponse(BaseModel):
    """Dashboard summary for the GUI."""
    service: str = "LemGendary Dataset Compiler API"
    version: str
    uptime_seconds: float
    active_jobs_count: int
    total_datasets_count: int
    total_storage_bytes: int
    total_storage_gb: float
    hardware: HardwareInfo
    presets: list[str]


class KaggleDatasetRegistryItem(BaseModel):
    """A Kaggle-published dataset from the unified registry."""
    key: str
    title: str
    name: str
    modernized_folder: str
    kaggle_ref: str
    clean_repo_id: str
    is_local_present: bool = False
    canonical_format: str = "webdataset"
    sample_count: int = 0
    size_gb: float = 0.0


class KaggleDownloadRequest(BaseModel):
    """Request body for downloading a Kaggle dataset."""
    kaggle_ref: str = Field(..., description="Kaggle dataset handle, kaggle:// URI, or URL")
    target_folder: str | None = Field(
        None, description="Destination folder name under datasets root"
    )
    force: bool = Field(False, description="Force overwrite existing folder")


class KaggleUploadRequest(BaseModel):
    """Request body for uploading a manifold to Kaggle."""
    manifold: str = Field(..., description="Local manifold folder name or registry key")
    kaggle_ref: str | None = Field(None, description="Optional target Kaggle dataset handle")
    no_wait: bool = Field(True, description="Do not wait for server-side extraction verification")


class CustomCompileRequest(BaseModel):
    """Request body for compiling a custom multi-source dataset."""

    custom_name: str = Field(
        ..., description="Unique dataset manifold name (e.g. CustomFaceEnhance)"
    )
    task: str = Field(
        "vision", description="Task type (restoration, segmentation, detection, quality, vision)"
    )
    preset: str | None = Field(
        None,
        description=(
            "Compiler preset ID from presets.yaml; omitted means presets.yaml `default_preset`"
        ),
    )
    canonical_format: str = Field("webdataset", description="Target container format")
    shard_size: int = Field(5000, description="Samples per shard container")
    sources: list[str] = Field(
        default_factory=list,
        description="List of source URLs or refs (kaggle://, hf://, gd://, gh://, or URLs)",
    )
    purge_loose_images: bool = Field(
        True,
        description=(
            "After a successful compile, purge loose images of manifolds packed into a container"
        ),
    )
