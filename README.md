# LemGendary Dataset Compiler Suite

> Industrial-standard manifold compiler for Vision, Restoration, and Time-Series datasets.
>
> **Function reference and architecture details:** [Whitepaper (PAPER_DATASET_COMPILER.md)](./lemgendary-docs/MD-Papers/PAPER_DATASET_COMPILER.md) · [Documentation Hub](https://lemgenda.github.io/ai-training-whitepapers/index.html)

---

## Current Status

| | |
| --- | --- |
| **Version** | `v16.6.0-MODERNIZED` |
| **Phase** | Phases 0, 1, 2, 3, 4, 5, 6, 7, 8 complete (9/9 roadmap phases) |
| **Next** | Production Modernization Complete — Ecosystem Ready |
| **Verified Manifolds** | 20 production manifolds, 1.4M+ sample stability |
| **Roadmap** | [modernization_roadmap.md](./modernization_roadmap.md) |

---

## Changelog

### v16.6.0 — CPA Integration Prep (Phase 8)

Prepared the dataset compiler sidecar service and CLI for seamless consumption by the LemGendary AI Studio Desktop GUI (`lemgendary-ai-studio-gui`) and Cross-Project Automations (CPA):

- **Canonical Compiler Presets (`presets.yaml`, `presets.py`)** — Defined standard compilation profiles (`quality-vision`, `restoration-hardlinked`, `detection-variable`, `cloud-archival`) encapsulating format, quality floors, vetting gates, auto-labeling, and container targets.
- **Desktop GUI Aggregation Endpoints (`api/routes/gui.py`)** — Added `/api/gui/state` (consolidated state snapshot), `/api/gui/datasets/with-stats` (deep format breakdown, file counts, storage footprints, and hardlink metrics), `/api/gui/jobs/active` (live job telemetry), `/api/gui/presets` (parameter schemas), and `/api/gui/quick-compile` (fast job dispatch).
- **Contract Freezing (`openapi.json`)** — Exported frozen OpenAPI 3.1 schema specification for zero-drift TypeScript client generation in desktop GUI applications.
- **CLI Enhancements (`cli.py`, `cli_args.py`, `manifold_compile.py`)** — Added `--preset` flag to `lemgendary compile` and introduced `lemgendary presets list` command rendering formatted parameter matrices.
- **Environment Manager Compatibility** — Audited and aligned ecosystem integration with `lemgendary-env-manager` (port 8000), verified 100% compliance under zero-emoji, zero-suppression, and zero-silent-failure rules.

### v16.5.0 — API + CLI Unification (Phase 7)

Introduced high-performance REST and WebSocket sidecar service (`api/`) and unified hybrid CLI routing:

- **`api/server.py`** — FastAPI application running on `127.0.0.1:8100` with CORS middleware, lifespan event queue draining, PID tracking in `.lgd_server/server.pid`, and interactive OpenAPI docs at `/docs`.
- **`api/jobs.py`** — SQLite persistent job tracking in `.lgd_server/jobs.db`, background subprocess execution, restart recovery marking orphaned jobs `interrupted`, and disk buffering in `.lgd_server/logs/<job_id>.log`.
- **`api/auth.py`** — Token-based security requiring `X-API-Key` or `Authorization: Bearer <token>` for modifying operations, with persistent local master key in `.lgd_server/token`.
- **`api/events.py`** — `ConnectionManager` handling WebSocket broadcast channels and job-specific log streaming.
- **`api/routes/`** — Modular endpoints for `/api/health` (liveness and hardware sensors), `/api/config` (inspect/validate `unified_data.yaml`), `/api/jobs` (list, submit compile/degrade, cancel, WS log stream), `/api/datasets` (manifold inspection), `/api/sources` (raw sets), `/api/kaggle` (sync triggers), `/api/gates` (NTFS hardlinks), and `/api/env` (delegating to `lem-env`).
- **Hybrid CLI Dispatch** — `lemgendary compile` and `lemgendary degrade` automatically detect an active server at `127.0.0.1:8100`, submit jobs via HTTP POST, and render live WebSocket logs to Rich Console in real time, falling back to direct in-process execution when the server is stopped or when `--no-server` is passed.
- **Server CLI Commands** — Added `lemgendary server start` (supporting `--background`), `lemgendary server stop`, and `lemgendary server status` reporting health and hardware telemetry.

### v16.4.2 — Zero-Suppression & Zero-Silent-Failure Hardening

Comprehensive audit and hardening across all 85 Python modules achieving absolute zero-diagnostic compliance without any suppressions:

- **Zero Suppressions (100% Clean)** — Completely eliminated all `# type: ignore`, `# pylint: disable`, and `# noqa` across the entire project. Root causes fixed directly via clean type narrowing, dynamic module imports, and proper typing.
- **Zero Silent Failures** — Replaced all 69 bare/swallowed `except ...: pass` blocks across 20 modules with structured contextual `logger.debug` and `logger.warning` handling.
- **Pyright Static Type Checking** — 0 errors, 0 warnings, 0 informations across all 85 Python files.
- **Pylint Clean Pass** — 0 errors across the codebase.
- **Bytecode Compilation** — 100% clean compilation via `py_compile` with `doraise=True`.
- **Full Compliance Validation** — Verified clean pass under `env_manager.cli validate -p lemgendary-datasets`.

### v16.4.1 — Degradation Engine (Phase 6)

Implemented pure NumPy/SciPy/PIL compiler-time synthetic manifold derivation and training-time on-the-fly augmentation engine.

- **`degrade/base.py`** — `DegradationProfile` protocol, `CompositeProfile`, `DynamicDegrader`, `parse_profile()` supporting functional tokens and presets (`motion-blur+iso-noise`, `lowlight-noise`, `rainy-haze`, `vintage-film`, `compression-artifacts`, `super-resolution-x4`, `full-spectrum-restoration`)
- **`degrade/blur.py`** — `GaussianBlur`, `MotionBlur` (directional linear), `DefocusBlur` (disk aperture), `BoxBlur`
- **`degrade/noise.py`** — `GaussianNoise`, `PoissonNoise` (photon shot), `SaltPepperNoise`, `ISOCalibratedNoise` (heteroscedastic Poisson + Gaussian readout)
- **`degrade/haze.py`** — `AtmosphericHaze` based on dark channel atmospheric scattering model
- **`degrade/rain.py`** — `RainStreaks` (directional wind streaks) and `RainMist`
- **`degrade/jpeg.py`** — `JPEGCompression` simulating 8x8 DCT quantization artifacts
- **`degrade/lowlight.py`** — `LowLight` combining non-linear gamma curve darkening, shadow readout noise, and color temperature tinting
- **`degrade/downsample.py`** — `Downsample` resolution reduction via bicubic, bilinear, lanczos, or nearest modes
- **`degrade/film.py`** — `FilmGrain`, `FilmScratches`, `FilmDust`, `ColorFade`
- **`generate_degrade.py`** — Compiler-side synthetic manifold generator writing clean-degraded pairs, exact quantitative JSON parameter logs to `labels/<split>/<name>.json`, and registry provenance
- **`cli.py` Integration** — Added top-level `degrade` subcommand with full configuration options
- **`lemgendary-training-suite` Integration** — Added `DynamicOnTheFlyDegrader` in `data/dataset.py` for online training augmentation with parameter tracking

### v16.4.0 — Smart Generation Engine (Phase 5)

Implemented smart multi-modal label, prompt, and mask generation infrastructure with unified CLI commands.

- **`generators/base.py`** — Structural `GenerationResult` type and generator interface
- **`generators/labels.py`** — Multi-strategy `LabelGenerator` supporting `blip_caption`, `clip_zeroshot`, `yolo_detection`, `parsenet_segmentation`, and `nima_quality`
- **`generators/prompts.py`** — `PromptGenerator` generating structured prompts for diffusion manifolds (`diffusers-v1`, `sd-v1`, `flux-v1`, `minimal`)
- **`generators/masks.py`** — `MaskGenerator` for semantic and instance segmentation (`parsenet`, `sam`, `modnet`)
- **`generate_cli.py`** — Standalone generation execution engine over compiled manifolds with configurable devices and sample limits
- **`cli.py` Integration** — Added top-level `label`, `prompt`, and `mask` commands

### v16.3.10 — Container Format Layer (Phase 4)

Introduced modular multi-format writer architecture supporting SOTA deep-learning containers alongside canonical directory layouts.

- **`formats/base.py`** — Defined `Sample` NamedTuple, `Writer` protocol, `make_writer()` factory, and `parse_also_format()`
- **`formats/directory.py`** — Canonical directory writer and `DirectorySampleSource` streaming iterator
- **`formats/mds.py`** — MosaicML Streaming format (`MDSWriter`) with true global shuffling and mid-epoch resumption
- **`formats/litdata.py`** — PyTorch Lightning LitData format (`LitDataWriter`) for variable-shape bounding box and landmark workloads
- **`formats/webdataset.py`** — WebDataset tar shard writer (`WebDatasetWriter`)
- **`formats/parquet.py`** — Tabular Parquet + Zstd writer (`ParquetWriter`)
- **`migrate_manifold_format.py`** — Retroactive container format migration tool with automated hardlink gate pre-flight
- **`compiler_core.py` & `manifold_compile.py`** — Integrated `--also-format` parameter for concurrent multi-format emission during compile runs

### v16.3.9 — Image Transcoding Layer (Phase 3)

Integrated zero-intermediate WebP transcoding engine for massive disk and bandwidth reduction while preserving quality floors.

- **`formats/transcode.py`** — `ImageTranscoder` implementing WebP q=92 for images, WebP q=95 for restoration targets, and WebP lossless for segmentation masks
- **Alpha Channel Resilience** — Transparent PNGs transcode to WebP lossless; alpha JPEG candidates composited cleanly on white background
- **`migrate_manifold_image_format.py`** — In-place retroactive image format migration tool with atomic replacement and registry updates
- **In-flight Compile Integration** — `process_image()` directly emits transcoded byte formats without intermediate JPEG writes

### v16.3.8 — Audit & Deduplication Engine (Phase 2)

Comprehensive image integrity, bounding box/landmark validation, exact/perceptual deduplication, and NTFS hardlink fraction auditing.

- **`audit/vision_audit.py`** — `VisionAuditor` with magic byte sniffing (PNG, JPEG, WebP, TIFF), per-task resolution floors, black/white frame detection, aspect ratio limits, pair alignment, and 15+ canonical reject codes
- **`audit/dedup.py`** — `ExactHasher` (raw byte MD5) and `PerceptualHasher` (DCT-based 64-bit pHash + dHash with pure NumPy/SciPy)
- **`audit/reject_log.py`** — SQLite `reject_log` management in `manifold_registry.db` with export to `rejects.jsonl` and `dataset_info.yaml`
- **`audit/hardlinks.py`** — `audit_hardlinks()` computing filesystem hardlink percentages and evaluating tiered gates (`PROCEED`, `WARN`, `BLOCK`)
- **`cli.py` Integration** — Added `audit` command and `--no-hash` / `--dedup` compiler flags

### v16.3.7 — Zero Suppressions (Phase 1.5.7)

Enforced suppression policy across the codebase. Every remaining `# type: ignore` now carries a bracket code and a documented reason; the only two survivors are `[import-untyped]` on MetaTrader5 (which ships no PEP 561 stubs) and both live in `mt5_bridge.py` / `mt5_pipeline.py`.

- **`runtime/environment.py`** — Dynamic `sys.stdout.reconfigure` via `getattr` + `callable()` guard (no `[attr-defined]` needed)
- **`utils/image.py`** — `bool()` / `float()` casts for NumPy scalar narrowing; removed unused `img.was_converted` side-effect
- **`models/diffusion.py`, `models/encoder.py`** — `self.processor: Any` to bypass transformers 5.17 stub gaps on `return_tensors` / `padding` kwargs
- **`converters/xml.py`** — `_require_text()` helper replaces four inline `# type: ignore[arg-type]`
- **`converters/matlab.py`** — Explicit `spmatrix=False` selects non-deprecated `sio.loadmat` overload; `scipy-stubs` added to `requirements-datasets.txt`
- **`converters/*.py`** — Return-type annotations added to all seven parsers so `manifold_compile.py` casts target concrete types rather than `Unknown`
- **`manifold_compile.py`** — `_require_ann_path()` narrows `Path | None` from `detect_annotations()`; typed lookup dicts (`coco_file_to_id: dict[str, int]`, `parquet_map`, `matlab_map`)
- **`mt5_bridge.py`** (new) — Single import site for MT5; `_mt5: Any` bypasses Pyrefly's incomplete module introspection; all 20 MT5 call-site suppressions in `mt5_verify_samples.py` collapsed into one bridge
- **`compiler_core.py`** — Removed 3 suppressions; `setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)` replaces attribute-assignment + ignore; all `models.*` imports no longer suppress
- **`doc_generator.py`** — `_meta_data: dict[str, Any]` annotation removes 6 suppressions
- **`sources/gd.py`** — Two stale `gdown` suppressions removed (gdown 6.2 ships types)
- **Dead code** — Phase 1.2 shims (`hf_manager.py`, `gh_manager.py`, `gd_manager.py`, `kaggle_manager.py`) removed after confirming zero functional importers

### v16.3.6 — Unified CLI (Phase 1.5)

Typer-based `cli.py` provides a single entry point for every dataset-compiler operation. Existing scripts remain invocable directly — `cli.py` is a thin delegating shell so Phases 2–7 can add commands one at a time.

- **`cli_args.py`** (new) — SSOT for the shared argparse parser and for `lem-env` executable discovery (3-tier resolution: `PATH` → sibling venv → hub install)
- **`cli.py`** (new) — Typer app with subcommands: `compile`, `reduce`, `modernize`, `sync push|pull`, `docs regen|manifolds`, `config validate|show`, `env validate|status|install`, `version`
- **`env` sub-app** — Delegates to `lem-env` (LemGendary Environment Manager); `env validate` returns the environment manager's exit code verbatim
- **`compiler_core.py` + `manifold_compile.py`** — Both now call `build_parser()` from `cli_args.py`. Two-parser drift eliminated (previously had to stay manually in sync)

### v16.3.5 — SOLID Cleanup (Phase 1.5.5)

Extracted pure utility functions and shared infrastructure from `compiler_core.py`. Line count: ~1,300 → ~800.

- New `runtime/environment.py` — process-level bootstrap (UTF-8 stdio, CUDA JIT env vars, torch capability guards)
- New `utils/` package — `fs`, `geometry`, `hashing`, `image`, `math`, `naming`, `net`
- New `registry.py` — registry lifecycle, schema upgrades, migration helpers
- New `audit/ground_truth.py` — registry-pattern loader for 8 quality datasets (AVA, AADB, KonIQ, SPAQ, TID2013, LIVE, CSIQ, TAD66K)
- `compiler_core.py` re-exports the extracted symbols for backward compatibility
- Refactored `_Annotation` TypedDict with `data: list[Any]` (accepts both `int` and `float` numerics)

### v16.3.4 — `converters/` Package (Phase 1.4)

Extracted the 7 annotation parsers from `compiler_core.py` into a dedicated package.

- New `converters/` — `coco`, `parquet`, `xml`, `yolo`, `matlab`, `safetensors`, `dispatch`
- `parse_*` functions now live in their format-specific module; identical signatures and byte-parity output
- `compiler_core.py` re-exports preserve `from compiler_core import parse_coco` for `manifold_compile.py`

### v16.3.3 — Registry Promotion (Phase 1.3)

Registry databases now live inside each manifold folder at `<manifold>/manifold_registry.db`.

- Auto-migration from legacy `.cache/registry_<name>.db` on first open
- Schema expanded with `perceptual_hash`, `img_format`, `img_size_bytes`, `target_size_bytes`, `mask_size_bytes`, `is_hardlinked`, `reject_code`, `audit_trail`, `created_at`, `quality_dist`
- New `audit_events` and `reject_log` tables for Phase 2 pipeline
- New `migrate_registry.py` — offline migration tool with `--dry-run` and `--verify` modes

### v16.3.2 — Sources Package (Phase 1.2)

Consolidated the four fetch backends into a `sources/` package.

- `hf_manager.py` → `sources/hf.py`
- `gh_manager.py` → `sources/gh.py`
- `gd_manager.py` → `sources/gd.py`
- `kaggle_manager.py` → `sources/kaggle.py`
- New `sources/base.py` — shared auth, progress bar, and status emitters
- Top-level shims were created for the transition and removed in v16.3.7
- Hub PS1 now points at `sources\*.py`

### v16.3.1 — Config Schema (Phase 1.1)

`unified_data.yaml` now validates through a Pydantic v2 schema at import time.

- New `config_schema.py` — `UnifiedData`, `DatasetEntry`, `SourceRef`, `ImageFormatPolicy`, `ContainerPolicy`, `RegistryMetadata`, `GlobalConstraints`
- Malformed config aborts compilation with structured error before any filesystem mutation (exit code 2 for validation, 3 for missing file)
- `compiler_core.py` reads via `load_unified_data()`; `to_legacy_dict()` preserves the original `_registry_metadata` shape for downstream code

### v16.3.0 — Modernized Manifold Standard (Phase 0)

Retired the legacy `Large` suffix from manifold folder names.

- New `modernize_manifold.py` — renames folders, regenerates metadata, re-uploads to Kaggle, updates `unified_data.yaml` programmatically
- Hub menu option `4. [MODERNIZE]`
- `dataset_info.yaml` now emits **relative** `path:` (`../LemGendaryDatasets/<name>`)
- Forex fields (`pairs`, `timeframe_rungs`, `start_date`, `lookback_bars`) preserved through doc regeneration
- `MANIFOLD_TASK_MAP` carries both modern (suffix-free) and legacy entries
- `manifold_reduce.py` eligibility filter accepts both naming conventions
- New `regenerate_manifolds_md.py` — rebuilds top-level `manifolds.md` from live registry

### v16.2.9 — Notebook Hygiene

- Removed all P100 / `cu118` references from generated notebooks (Kaggle + Colab)
- Kaggle notebooks now **never download datasets** — they consume `/kaggle/input/` attachments only
- Colab notebooks scan `/kaggle/input` before any network fetch; support `notebook_no_download: true` config gate

### v16.2.8 — High-Fidelity Compiler Baseline

The pre-modernization reference state. Highlights:

- **O(1) physical skip-indexing** via flat `os.scandir` + hash set
- **Lanczos-3** resampling for diffusion/VLM; **512px** floor for quality, **224px** for restoration
- **ThreadPoolExecutor zero-IPC** path for I/O-bound workflows
- **1024px** diffusion baselines
- SQLite atomic registry resumption
- Kaggle/HF sync with real-time extraction tracking and bidirectional resumption
- **NTFS hardlink dedup** on restoration targets (recovered ~1.06 TB on UPNv2)
- **Parquet + Zstd** Forex pipeline (738 GB → 5.2 GB, 99.3% reduction)
- DPED mirroring, VRAM de-fragmentation, ParseNet/RetinaFace semantic extraction
- Decoupled `doc_generator.py`

---

## Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Modernize (suffix removal) | Done |
| 1.1 | Pydantic config schema | Done |
| 1.2 | `sources/` package | Done |
| 1.3 | Registry promotion | Done |
| 1.4 | `converters/` extraction | Done |
| 1.5.5 | SOLID cleanup | Done |
| 1.5 | Unified CLI skeleton | Done |
| 1.5.7 | Zero Suppressions pass | Done |
| 1.6 | Package skeletons (`formats/`, `generators/`, `degrade/`) | Done |
| 1.7 | Runtime env contract (env-manager SSOT) | Done |
| 2 | Audit & dedup | Done |
| 3 | Transcoding | Done |
| 4 | Format layer (MDS / LitData / WebDataset / Parquet) | Done |
| 5 | Smart generation (labels, prompts, masks) | Done |
| 6 | Degradation engine (`degrade/`) | Done |
| 7 | API + CLI unification (`api/`) | Done |
| 8 | CPA integration prep (`presets/`, `api/routes/gui.py`) | Done |

Full details: [modernization_roadmap.md](./modernization_roadmap.md)

---

## Developer Interface

### Unified CLI

The primary interface. Every operation is reachable through `cli.py`:

```bash
# Compiler Presets (Phase 8)
python cli.py presets list
python cli.py compile --model nima_aesthetic --preset quality-vision
python cli.py compile --model nafnet_deblurring --preset restoration-hardlinked

# Compile with transcoding and modern container formats
python cli.py compile --model nima_aesthetic --max-gb 50
python cli.py compile --model nima_aesthetic --image-format webp --image-quality 92 --also-format mds
python cli.py compile --model nima_technical --workers 16
python cli.py compile --model nima_aesthetic --no-labeling    # bypass YOLO

# Degradation synthesis (Phase 6)
python cli.py degrade --source raw-sets/div2k --output LemGendizedNafNetDebluringSynthetic --profile motion-blur+iso-noise
python cli.py degrade --source ../LemGendaryDatasets/LemGendizedNimaAesthetic --output LemGendizedLowLightSynthetic --profile lowlight-noise --intensity high

# Audit & Deduplication
python cli.py audit --model nima_aesthetic
python cli.py audit --manifold ../LemGendaryDatasets/LemGendizedNimaAesthetic --sample 500

# Transcoding
python cli.py transcode --model nima_technical --image-format webp --image-quality 92

# Container format write and migration
python cli.py format write --model nima_aesthetic --to mds
python cli.py format migrate --model nima_aesthetic --to mds --verify

# Smart multi-modal generation
python cli.py label --model parsenet --strategy parsenet_segmentation
python cli.py prompt --model diffusion_master --template diffusers-v1
python cli.py mask --model parsenet --strategy sam

# Reduce
python cli.py reduce --max-gb 10

# Modernize (retire `Large` suffix)
python cli.py modernize --dry-run
python cli.py modernize --all --yes
python cli.py modernize --datasets nima_technical,nima_aesthetic
python cli.py modernize --skip-kaggle      # rename locally, no re-upload

# Kaggle sync
python cli.py sync push --model nima_aesthetic
python cli.py sync pull --url username/lemgendizednimaaesthetic

# Documentation
python cli.py docs regen
python cli.py docs manifolds --check

# Configuration
python cli.py config validate
python cli.py config show

# Sidecar API Server (Phase 7)
python cli.py server start --background    # Launch daemon on 127.0.0.1:8100
python cli.py server status                # Probe health, uptime, and hardware sensors
python cli.py server stop                  # Gracefully terminate daemon

# Environment Manager passthrough
python cli.py env validate                 # -> lem-env validate --project lemgendary-datasets
python cli.py env status                   # -> lem-env audit --fast
python cli.py env install                  # -> lem-env install --project lemgendary-datasets

# Version
python cli.py version
```

### API Service & Sidecar Integration

The Dataset Compiler Suite exposes a high-throughput REST and WebSocket service on `127.0.0.1:8100` (`api/`), mirroring the conventions of the LemGendary Environment Manager for LemGendary AI Studio GUI sidecar operation:

- **Interactive Documentation**: Swagger UI at `http://127.0.0.1:8100/docs` and OpenAPI JSON at `http://127.0.0.1:8100/openapi.json`.
- **Token Security**: Protected endpoints require `X-API-Key` or `Authorization: Bearer <token>`, with automatic local key generation and persistence in `.lgd_server/token`.
- **Persistent Job Engine**: Backed by SQLite in `.lgd_server/jobs.db` with thread pool execution, restart recovery marking orphaned tasks `interrupted`, and disk buffering in `.lgd_server/logs/<job_id>.log`.
- **Desktop GUI Endpoints (`/api/gui`)**: Fast, aggregated endpoints designed for hydration in `lemgendary-ai-studio-gui`:
  - `GET /api/gui/state`: Consolidated compiler status, uptime, storage capacity, and hardware profile.
  - `GET /api/gui/datasets/with-stats`: Deep manifold inventory with per-format file distribution (WebP, JPEG, PNG, Parquet), exact byte sizes, and hardlink deduplication ratios.
  - `GET /api/gui/jobs/active`: Detailed execution telemetry (progress percentage, elapsed time, samples processed).
  - `GET /api/gui/presets`: Canonical parameter definitions for compiler preset profiles.
  - `POST /api/gui/quick-compile`: Fast job submission using preset templates.
- **WebSocket Streaming**: Live logs stream to subscribers via `ws://127.0.0.1:8100/api/ws/jobs/{id}/logs`.
- **Transparent Hybrid Routing**: `lemgendary compile` and `lemgendary degrade` automatically detect an active server, post tasks via HTTP, and stream logs live to Rich Console, falling back to in-process execution when the server is offline or when `--no-server` is specified.

### Interactive Hub

```powershell
./lemgendary_datasets_hub.ps1
```

| Option | Action |
| --- | --- |
| `1. [COMPILE]` | Build SOTA manifold (Vision, Forex, Restoration) |
| `2. [REDUCE]` | Create downsampled variants with custom fold/timeframe selection |
| `3. [SYNC]` | Kaggle push/pull with real-time extraction monitoring |
| `4. [MODERNIZE]` | Retire `Large` suffix, re-upload, update registry |
| `Q. [QUIT]` | Exit |

### Direct Script Access (Advanced)

Every underlying script remains invocable directly for automation and pipelines:

```bash
# Core compiler
python manifold_compile.py --model nima_aesthetic --max_gb 50 --also-format mds
python manifold_reduce.py --reduce --max_gb 10

# Retroactive migrations
python migrate_manifold_image_format.py --manifold ../LemGendaryDatasets/LemGendizedNimaTechnical --image-format webp
python migrate_manifold_format.py --manifold ../LemGendaryDatasets/LemGendizedNimaTechnical --to mds --verify
python migrate_registry.py --dry-run

# Smart generation runner
python generate_cli.py --manifold ../LemGendaryDatasets/LemGendizedNimaAesthetic --kind label --strategy blip_caption

# Kaggle sync (underlying)
python manifold_sync.py --action sync --model nima_aesthetic
python manifold_sync.py --action get --url username/slug
python sources/kaggle.py --action upload --repo_id username/dataset --output_dir <path>
python sources/kaggle.py --action download --repo_id username/dataset --output_dir <path>
python sources/kaggle.py --action status --repo_id username/dataset

# Documentation
python doc_generator.py --all
python regenerate_manifolds_md.py

# Config schema validation
python config_schema.py
```

### Hardware Resilience

- **CPU-GUARD** — auto-triggers High-Speed Mode on CPU-only hosts for massive datasets
- **CUDA-Sentry** — real-time GPU detection for NIMA/YOLO vetting

---

## Architecture

```text
lemgendary-datasets/
├── cli.py                         # Unified Typer entry point (Phase 1.5)
├── cli_args.py                    # SSOT for argparse + env-manager resolver
├── compiler_core.py               # Coordinator & pipeline dispatcher
├── manifold_compile.py            # Compilation engine
├── manifold_reduce.py             # Reduction engine
├── manifold_sync.py               # Kaggle sync orchestrator
├── modernize_manifold.py          # Suffix-removal tool (Phase 0)
├── migrate_registry.py            # Registry migration (Phase 1.3)
├── migrate_manifold_image_format.py # Retroactive WebP transcoding (Phase 3)
├── migrate_manifold_format.py     # Retroactive container migration (Phase 4)
├── generate_cli.py                # Standalone smart generation runner (Phase 5)
├── doc_generator.py               # Per-manifold docs
├── regenerate_manifolds_md.py      # Top-level matrix regeneration
├── config_schema.py               # Pydantic config (Phase 1.1)
├── registry.py                    # Registry lifecycle (Phase 1.3 / 1.5.5)
├── archive_manager.py             # Archive + resume
├── common_sync.py                 # Kaggle streaming core
├── notebook_generator.py          # Notebook matrix (Kaggle + Colab)
├── api/                           # REST & WebSocket API sidecar server (Phase 7)
│   ├── server.py, jobs.py, auth.py, events.py, models.py
│   └── routes/ (health, config, jobs, datasets, sources, kaggle, gates, env)
├── sources/                       # Fetch backends (Phase 1.2)
│   ├── hf.py, gh.py, gd.py, kaggle.py
│   └── base.py
├── converters/                    # Annotation parsers (Phase 1.4)
│   ├── coco.py, parquet.py, xml.py, yolo.py, matlab.py, safetensors.py
│   └── dispatch.py
├── audit/                         # Audit & Dedup engine (Phase 2)
│   ├── vision_audit.py, dedup.py, reject_log.py, hardlinks.py
│   └── ground_truth.py
├── formats/                       # Transcoding & Containers (Phase 3 & 4)
│   ├── transcode.py, base.py, directory.py, mds.py, litdata.py
│   └── webdataset.py, parquet.py
├── generators/                    # Smart generation (Phase 5)
│   ├── base.py, labels.py, prompts.py, masks.py
├── degrade/                       # Degradation engine skeleton (Phase 6)
│   ├── base.py, blur.py, noise.py, haze.py, rain.py, jpeg.py, lowlight.py, downsample.py, film.py
├── runtime/                       # Process bootstrap (Phase 1.5.5)
│   └── environment.py
├── utils/                         # Pure helpers (Phase 1.5.5)
│   ├── fs.py, geometry.py, hashing.py, image.py
│   ├── math.py, naming.py, net.py
├── models/                        # AI model wrappers (pre-existing)
│   ├── quality_scorer.py, detection.py, diffusion.py, encoder.py
│   └── nima.py
├── mt5_bridge.py                  # MT5 IPC bridge (Phase 1.5.7)
├── mt5_pipeline.py                # Forex MT5 pipeline
└── unified_data.yaml              # Registry manifest
```

---

## Output Topology

```text
raw-sets/                                          # Source datasets (Cleanup Guardian)
../LemGendaryDatasets/<manifold>/
├── images/{train,val}/                            # Standard structured images
├── labels/{train,val}/                            # NIMA 10-bin or YOLO vectors
├── targets/{train,val}/                           # Ground truth (SR/Restoration)
├── masks/{train,val}/                             # Segmentation masks
├── shards/                                        # WebDataset .tar (diffusion)
├── manifold_registry.db                           # SQLite registry (Phase 1.3)
├── dataset_info.yaml                              # Suite metadata (relative path:)
├── category.txt
├── classes.txt
├── index.json
└── README.md
```

---

## Project Ecosystem

| # | Project | Folder |
| --- | --- | --- |
| 1 | LemGendary Environment Manager | `.\lemgendary-env-manager\` |
| 2 | LemGendary Dataset Compiler Suite | `.\lemgendary-datasets\` |
| 3 | LemGendary Model Training Suite | `.\lemgendary-training-suite\` |
| 4 | LemGendary AI Studio GUI | `.\lemgendary-ai-studio-gui\` |
| 5 | LemGendary AI Documentation Hub | `.\lemgendary-docs\` |
| 6 | LemGendary Compiled Manifolds | `.\LemGendaryDatasets\` |
| 7 | LemGendary Trained Models | `.\LemGendaryModels\` |

---

LemGendary AI Suite — Advanced Agentic Coding 2026
