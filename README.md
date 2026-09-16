# LemGendary Dataset Compiler Suite

> Industrial-standard manifold compiler for Vision, Restoration, and Time-Series datasets.
>
> **Function reference and architecture details:** [Whitepaper (PAPER_DATASET_COMPILER.md)](./lemgendary-docs/MD-Papers/PAPER_DATASET_COMPILER.md) · [Documentation Hub](https://lemgenda.github.io/ai-training-whitepapers/index.html)

---

## Current Status

| | |
| --- | --- |
| **Version** | `v16.3.7-MODERNIZED` |
| **Phase** | 1.5.7 / 8 complete |
| **Next** | Phase 1.6 — Package skeletons (`formats/`, `generators/`, `degrade/`) |
| **Verified Manifolds** | 20 production manifolds, 1.4M+ sample stability |
| **Roadmap** | [modernization_roadmap.md](./modernization_roadmap.md) |

---

## Changelog

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
- **Stubs for Phases 2–7** — `audit`, `format`, `label`, `prompt`, `mask`, `degrade`, `server` print a phase pointer and exit 1
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
| **1.6** | **Package skeletons (`formats/`, `generators/`, `degrade/`)** | **Next** |
| 1.7 | Runtime env contract (env-manager SSOT) | Pending |
| 2 | Audit & dedup | Pending |
| 3 | Transcoding | Pending |
| 4 | Format layer (MDS / LitData / WebDataset) | Pending |
| 5 | Smart generation (labels, prompts, masks) | Pending |
| 6 | Degradation engine | Pending |
| 7 | API + CLI unification | Pending |
| 8 | CPA integration prep | Pending |

Full details: [modernization_roadmap.md](./modernization_roadmap.md)

---

## Developer Interface

### Unified CLI

The primary interface. Every operation is reachable through `cli.py`:

```bash
# Compile
python cli.py compile --model nima_aesthetic --max-gb 50
python cli.py compile --model nima_technical --workers 16
python cli.py compile --model nima_aesthetic --no-labeling    # bypass YOLO

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

# Environment Manager passthrough
python cli.py env validate                 # -> lem-env validate --project lemgendary-datasets
python cli.py env status                   # -> lem-env audit --fast
python cli.py env install                  # -> lem-env install --project lemgendary-datasets

# Version
python cli.py version
```

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
python manifold_compile.py --model nima_aesthetic --max_gb 50
python manifold_reduce.py --reduce --max_gb 10

# Registry migration
python migrate_registry.py --dry-run
python migrate_registry.py
python migrate_registry.py --verify

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
├── cli.py                    # Unified Typer entry point (Phase 1.5)
├── cli_args.py               # SSOT for argparse + env-manager resolver
├── compiler_core.py          # Coordinator (post-1.5.5 cleanup)
├── manifold_compile.py       # Compilation engine
├── manifold_reduce.py        # Reduction engine
├── manifold_sync.py          # Kaggle sync orchestrator
├── modernize_manifold.py     # Suffix-removal tool (Phase 0)
├── migrate_registry.py       # Registry migration (Phase 1.3)
├── doc_generator.py          # Per-manifold docs
├── regenerate_manifolds_md.py # Top-level matrix regeneration
├── config_schema.py          # Pydantic config (Phase 1.1)
├── registry.py               # Registry lifecycle (Phase 1.5.5)
├── archive_manager.py        # Archive + resume
├── common_sync.py            # Kaggle streaming core
├── notebook_generator.py     # Notebook matrix (Kaggle + Colab)
├── sources/                  # Fetch backends (Phase 1.2)
│   ├── hf.py, gh.py, gd.py, kaggle.py
│   └── base.py
├── converters/               # Annotation parsers (Phase 1.4)
│   ├── coco.py, parquet.py, xml.py, yolo.py, matlab.py, safetensors.py
│   └── dispatch.py
├── audit/                    # Audit pipeline (Phase 1.5.5 seed)
│   └── ground_truth.py
├── runtime/                  # Process bootstrap (Phase 1.5.5)
│   └── environment.py
├── utils/                    # Pure helpers (Phase 1.5.5)
│   ├── fs.py, geometry.py, hashing.py, image.py
│   ├── math.py, naming.py, net.py
├── models/                   # AI model wrappers (pre-existing)
│   ├── quality_scorer.py, detection.py, diffusion.py, encoder.py
│   └── nima.py
├── mt5_bridge.py             # MT5 IPC bridge (Phase 1.5.7)
├── mt5_pipeline.py           # Forex MT5 pipeline
└── unified_data.yaml         # Registry manifest
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
