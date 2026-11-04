# Lead Architect Review: LemGendary Dataset Compiler Suite

Scope: `./` (this repository), reviewed read-only against `../ROBOTS.md`,
`../Instructions.md` and
`../lemgendary-docs/MD-Papers/PAPER_DATASET_COMPILER.md`. No source,
configuration or existing documentation was modified. Paths are relative to
this repository unless prefixed with `../`.

Review date: 2026-11-02. Repository HEAD: `fdec07f`.

Evidence legend:

- **Verified**: reproduced or proven by tool output.
- **Read**: established by reading the code; not executed.
- **Inferred**: likely, but depends on runtime conditions that were not
  exercised.

## Executive Findings

### Strengths

- Clean extension contracts exist at the leaf level: the `Writer` Protocol
  (`formats/base.py:47`), the `Generator` Protocol (`generators/base.py:39`)
  and the `DegradationProfile` Protocol (`degrade/base.py:32`).
- `unified_data.yaml` is validated by a Pydantic schema
  (`core/config_schema.py:143`, `extra="forbid"`).
- A services layer exists (`services/`) and the CLI uses it for in-process
  execution.
- The job manager is persistent, with restart recovery
  (`api/jobs.py:67-85`). Token comparison is constant-time
  (`api/auth.py:66`).
- Code hygiene is good where it was checked:
  - No `# noqa`, `# type: ignore` or `# pylint: disable` in any `.py` file.
  - No `warnings.filterwarnings` and no bare `except:`.
  - Zero syntax errors, and zero undefined-name (F821) findings from ruff.

### Weaknesses

- **The CLI and the API do not share a service layer.**
  - The CLI calls `services/*`.
  - The API builds argv lists and spawns repository scripts as subprocesses
    (`api/routes/jobs.py:66`, `api/routes/kaggle.py:148`,
    `api/routes/modernize.py:107`, `api/routes/llm.py:38`).
  - `lemgendary_datasets_hub.ps1:878` is a third orchestration path.
  - This violates ROBOTS.md "CLI and In-Process Unification".
- **The services depend on scripts, not the other way round.**
  - `services/*` import `tools/*` and `core/*` entry points.
  - They then catch `SystemExit` to recover exit codes
    (`services/sync_service.py:34`, `services/doc_service.py:51`).
- **God modules** (line counts):
  - `core/compiler_core.py` (1346)
  - `core/common_sync.py` (1142)
  - `tools/modernization_queue.py` (1074)
  - `cli.py` (1052)
  - `core/manifold_compile.py` (956). Its `process_dataset` alone spans
    roughly lines 143 to 955.
  - `forex/pipeline.py` (836)
  - `api/routes/gui.py` (673)
  - `lemgendary_datasets_hub.ps1` (1253)
- **OCP is violated in the core paths.** The following are all hard-coded
  `if`/`elif` chains:
  - format writers
  - degradation tokens
  - annotation formats
  - task directory layouts
  - the forex branch inside the main compile loop
- **There is no SSOT for**:
  - the datasets root (more than 10 resolutions)
  - task mapping (4 copies)
  - image-quality defaults (5 copies)
  - port 8100 (9 definitions across 4 repos)
  - the Kaggle owner handle (9 hard-coded uses)

### Risks

- **Security:** most mutating endpoints are unauthenticated, and CORS is
  `allow_origins=["*"]` with `allow_credentials=True`
  (`api/server.py:82-83`). Any web page open in the user's browser can:
  - trigger Kaggle uploads
  - start modernization with `--purge-loose`
  - rewrite `unified_data.yaml`
  - run the test suite inside the server
- **Data loss:** `/api/modernize/run` drops the GUI's `dataset`, `dry_run` and
  `force` fields. A "dry run" for one dataset can therefore execute the full
  batch with purge (see D-05).
- **Registry corruption:** `/api/gui/custom-compile` round-trips the
  hand-written `unified_data.yaml` through `yaml.dump`. One invalid value makes
  every later compiler start-up `sys.exit(2)` (see D-03).

### Opportunities

- Use one `paths` and one `registry` module as the SSOT. Every route,
  generator, tool and CLI command can consume them.
- Plug-in registries for writers, degradation filters, annotation detectors
  and dataset-type compile strategies. This removes the core-loop edits that
  ROBOTS.md forbids.
- One typed job-dispatch entry point (`JobSpec` mapped to a service function).
  This lets the API, the CLI and the hub share one code path.
- Generate `openapi.json` from `create_app()`, and generate the GUI's
  TypeScript types from it. This removes the contract drift in D-07.

## Architecture Findings

### Current Architecture

```text
cli.py (Typer) ──► services/* ──► tools/*.py, core/manifold_compile.py
                                   (argparse mains, sys.exit, print)
api/server.py ──► api/routes/* ──► api/jobs.py ──► subprocess: core/*.py,
                                                   tools/*.py, sources/*.py,
                                                   cli.py llm compile
lemgendary_datasets_hub.ps1 ──► subprocess: core/manifold_compile.py,
                                 sources/*.py, tools/*.py
core/manifold_compile.py ──► sys.path import ../lemgendary-training-suite
```

Observations:

- **Import-time side effects.**
  - `core/compiler_core.py:49-51` calls `bootstrap_runtime()`.
  - `core/compiler_core.py:114-158` loads `config.json` and validates the YAML,
    then calls `sys.exit(2|3)`.
  - `api/jobs.py:428` instantiates `JobManager()`, which creates
    `.lgd_server/jobs.db`.
  - Importing `cli.py` imports the whole API graph through `api/auth.py` and
    `api/__init__.py:16`. That includes `kagglehub` via
    `api/routes/modernize.py:16` and `core/common_sync.py:18`. **Verified.**
- **Path hacks.** There are 30 `sys.path.insert` hacks, for example
  `sources/hf.py:5` and `core/docs/generators/*.py:16-22`. They are needed
  because the modules double as scripts.
- **Cross-repo coupling.** `core/manifold_compile.py:66-75` imports
  `training.notebooks` from the sibling repository by mutating `sys.path`.
- **Legacy shims.** These duplicate `forex/*`:
  - `tools/forex_schema.py`
  - `tools/mt5_bridge.py`
  - `tools/mt5_pipeline.py`
  - `tools/convert_forex_manifold.py`
  - `tools/embed_parquet_descriptions.py`

### Proposed Architecture

```text
lemgendary_datasets/
  paths.py            ProjectPaths: root, state dir, datasets root (from
                      registry), raw-sets; all Path(__file__)-anchored
  registry/           DatasetRegistry (typed, loaded once, ruamel round-trip
                      writer for safe edits); TaskResolver uses `task` field
  formats/registry.py @register_writer("mds") ...; make_writer() looks up dict
  degrade/registry.py @register_filter("blur.gauss") ...; parse_profile()
                      resolves tokens + presets.yaml-style preset table
  converters/registry.py ordered AnnotationDetector strategies
  pipeline/           CompilePipeline + strategies per dataset_type
                      (VisionStrategy, ForexStrategy, LLMStrategy)
  services/           pure functions returning typed Result / raising domain
                      errors; never sys.exit, never print (use logging)
  jobs/               JobSpec(kind, params) -> SERVICE_REGISTRY[kind];
                      worker entry `python -m lemgendary_datasets.worker`
  api/                thin routers, all mutating routes behind verify_token,
                      CORS restricted to Tauri + dev origins
  cli.py              thin Typer shell over services (no subprocess)
  tools/              thin argparse wrappers over services (or removed)
```

## Refactoring Plan

All of the following are proposals. None has been applied.

1. **`paths.py` (new)**
   - Change: replace these CWD-relative paths:
     - `core/compiler_core.py:114,253-254`
     - `api/auth.py:16`
     - `api/server.py:33`
     - `api/routes/modernize.py:55`
     - `api/routes/gui.py:271,620`
     - `api/routes/datasets.py:16`
     - `cli.py:765,842,877`
   - Also replace the copy-pasted `LemGendaryDatasets` fallbacks in
     `core/docs/generators/*.py`.
   - SOLID: SRP, DIP and SSOT.
2. **`registry/` (new) wrapping `core/config_schema.py`**
   - Change: move the dataset task and layout logic here:
     - `detect_task` (`core/compiler_core.py:336-365`)
     - `MANIFOLD_TASK_MAP` (`core/docs/metadata.py:50-97`)
     - the inline `mapping` (`core/manifold_compile.py:436-448`)
   - Read `task` and `task_override` from `unified_data.yaml`. Writes must
     preserve comments (ruamel) and validate before they are persisted.
   - SOLID: SSOT and SRP.
3. **`formats/base.py:58-77`**
   - Change: replace the `if` chain with a writer registry.
   - Remove the eager imports in `formats/__init__.py`. They defeat the lazy
     factory.
   - `api/routes/gui.py:70-120` should ask the registry for container
     directory names and extensions instead of hard-coding them.
   - SOLID: OCP and SSOT.
4. **`degrade/base.py` (`_build_preset`, `parse_profile`)**
   - Change: add a filter registry keyed by token.
   - Presets become data.
   - Expose the filters that are currently unreachable from expressions:
     `PoissonNoise`, `SaltPepperNoise`, `BoxBlur`, `RainMist`, `FilmDust`,
     `ColorFade`.
   - SOLID: OCP.
5. **`converters/dispatch.py` plus the `if fmt ==` chain in
   `core/manifold_compile.py`**
   - Change: add an ordered `AnnotationDetector` and parser registry.
   - SOLID: OCP and LSP.
6. **`core/manifold_compile.py`, `core/compiler_core.py`**
   - Change: split into pipeline stages and dataset-type strategies:
     - forex at `core/manifold_compile.py:259-320`
     - vision at the remainder
     - LLM, which is currently routed through `cli.py llm`
   - Remove the import-time `sys.exit` and the `os._exit`.
   - SOLID: SRP and OCP.
7. **`services/*`**
   - Change: invert the dependency so that tools call services, not services
     call tools.
   - Return typed results instead of catching `SystemExit`.
   - Replace argv re-serialisation (`services/compiler_service.py:46-91`)
     with a typed `CompileOptions` dataclass.
   - SOLID: DIP and ISP.
8. **`api/routes/*`, `api/jobs.py`**
   - Change: dispatch jobs through the `JobSpec` and service registry instead
     of argv to scripts.
   - Remove `/api/jobs/generic`, which runs arbitrary commands.
   - Apply `verify_token` to every mutating router.
   - SOLID: DIP and security.
9. **`cli.py`**
   - Change: split it into `cli/` sub-modules, one per Typer sub-app.
   - Resolve the `modernize` name clash (D-09).
   - Move the server-routing helper into a client adapter.
   - SOLID: SRP.
10. **`sources/*.py`**
    - Change: make them adapters behind a `SourceFetcher` Protocol, using
      `sources/base.py` helpers for tokens.
    - Remove the `archive_manager` fallback.
    - Check subprocess exit codes.
    - SOLID: LSP and DIP.
11. **`tools/forex_schema.py`, `tools/mt5_*.py`,
    `tools/convert_forex_manifold.py`, `tools/embed_parquet_descriptions.py`**
    - Change: delete these shims after migrating callers to `forex/*`.
    - SOLID: SSOT.
12. **`.pylintrc`, `POLICY.md`**
    - Change: remove the 48 disabled checks and rewrite the policy to match
      ROBOTS.md (see the next section).
    - Fix the root causes the linter reports.
    - Rule: ROBOTS.md zero-suppression.
13. **`lemgendary_datasets_hub.ps1`**
    - Change: reduce it to a menu that invokes `cli.py` sub-commands only.
    - SOLID: SSOT.

## Debug Report

Severity scale: **Critical** means data loss, security exposure or a core
feature that cannot work. **High** means a feature is broken. **Medium** means
incorrect behaviour in a subset of cases. **Low** means hygiene or latent
issues.

### D-01 Critical: unauthenticated mutating API with wildcard CORS

- **Where:** `api/server.py:82-83,89-97`.
- **Issue:** these routers are public:
  - `/api/gui/custom-compile`
  - `/api/gui/quick-compile`
  - `/api/kaggle/upload|download|sync|update-metadata`
  - `/api/modernize/run|reduce|regenerate-manifolds`
  - `/api/llm/compile|validate|test-suite`
- `/api/llm/validate` opens any `file_path` on disk (`api/routes/llm.py:79`).
- **Root cause:** `verify_token` is applied to only 3 routers. CORS is set to
  `*` with credentials.
- **Fix:**
  - Apply `Depends(verify_token)` to all non-health routers.
  - Restrict origins to the Tauri origins and `http://localhost:1420`.
  - The GUI must send `X-API-Key`; it currently sends no auth at all (see
    Cross-Project Findings).

### D-02 Critical: custom-compile cannot produce a manifold

- **Where:**
  - `api/routes/gui.py:651-652`
  - `api/models.py:269,275`
  - `core/manifold_compile.py:134-136,157-161`
- **Issue (Read):**
  - The default `preset="streaming-webdataset"` does not exist in
    `presets.yaml`. Its keys are `quality-vision`, `restoration-hardlinked`,
    `detection-variable` and `cloud-archival`. The job therefore exits 1.
  - With a valid preset, the default `purge_loose_images=True` appends
    `--cleanup`. `process_dataset` then prints "Cleanup complete" and returns
    before compiling, and the job is reported as COMPLETED.
  - `shard_size` is accepted but never used.
- **Root cause:** `--cleanup` is a placeholder mode that was misused as a
  post-compile purge flag. The preset inventory is duplicated in the GUI.
- **Fix:**
  - Add a real post-compile purge option.
  - Validate the preset name against `core/presets.py` before queuing, and
    return 400.
  - Serve presets only from `presets.yaml`.

### D-03 High: custom-compile corrupts the SSOT registry

- **Where:** `api/routes/gui.py:620-643`.
- **Issue:** the route rewrites `unified_data.yaml` with `yaml.dump`, which:
  - strips all comments and formatting from the 1056-line file
  - silently overwrites an existing key with the same derived `model_key`
  - looks first in a CWD-relative path
  - logs a persist failure and then dispatches the job anyway
- `canonical_format` and `task` are free strings in the request. An invalid
  value makes `load_unified_data` fail, so `core/compiler_core.py:150-158`
  calls `sys.exit(2)` for every later compile (Inferred).
- **Root cause:** there is no registry write API and no validation before the
  write.
- **Fix:**
  - Add a registry service with a ruamel round-trip.
  - Validate with `DatasetEntry` before writing.
  - Return 409 when the key already exists.
  - Abort when the write fails.

### D-04 High: asynchronous LLM compile jobs never start

- **Where:** `api/routes/llm.py:34-53`.
- **Issue:** the route is a sync `def`, so FastAPI runs it in a worker thread.
  There, `asyncio.get_running_loop()` raises `RuntimeError`, which is
  swallowed with `pass`. The job stays PENDING forever.
- `cli.py llm compile` then waits indefinitely on the WebSocket for
  `[PROCESS_TERMINATED]` (`cli.py:969`, `cli.py:76-88`).
- **Verified** with a FastAPI TestClient reproduction, which returned
  `no running event loop`.
- **Fix:**
  - Make the route `async def`, or pass the loop captured at start-up.
  - Remove the `except RuntimeError: pass`, which also violates ROBOTS.md.
  - Use `JobType.LLM`, not `COMPILE`.
  - Use `venv_python()` consistently.

### D-05 High: modernize run ignores the scope and dry-run flags

- **Where:**
  - `api/routes/modernize.py:103-131`
  - GUI caller `../lemgendary-ai-studio-gui/src/components/compiler/CompilerPanel.tsx:340`
- **Issue:**
  - The GUI sends `dataset`, `dry_run` and `force`. The request model has none
    of these, so they are dropped.
  - `purge_loose` defaults to appending `--purge-loose`.
  - `--datasets` is appended once per item, but argparse keeps only the last
    value (`tools/modernization_queue.py:1024`).
- **Root cause:** there is no shared contract (schema) between the GUI and the
  API.
- **Fix:**
  - Use one `ModernizeRunRequest` with `datasets: list[str]`, `dry_run` and
    `force`.
  - Join the list with commas, or better, call the service directly.
  - Generate the GUI types from OpenAPI.

### D-06 High: reduce always fails through the API

- **Where:**
  - `api/routes/modernize.py:141-144`
  - `tools/manifold_reduce.py:478-481`
  - `tools/manifold_reduce.py:65,86,111,116,352,354,386`
- **Issue:**
  - The route passes `--model` and `--scale`, which the script's argparse does
    not define. The script exits with code 2 (Read).
  - The script also depends on interactive `input()` prompts, which cannot
    work in a detached job.
  - `cli.py reduce` and `modernize reduce` (via `MigrationService`) share the
    interactive dependency.
- **Fix:** add a non-interactive `ReduceService(dataset_key, max_gb, suffix)`.
  Keep the prompts only in a CLI wrapper.

### D-07 High: GUI to API contract breaks

These were found by the cross-repo sweep and verified against server models.

- **Quick-compile:** the GUI sends `manifold_key`, but the server requires
  `model` (`api/models.py:223`). Every call returns 422.
- **Presets:** `/api/gui/presets` returns a dict (`api/models.py:218`). The GUI
  expects an array, so it always falls back to its hard-coded presets.
- **Modernize queue:** `/api/modernize/queue` returns a bare list. The GUI
  expects `{queue: [...]}`, so the queue always renders empty.
- **Job IDs:** job routes return `id`, but most GUI handlers read `job_id`.
- **Fix:** generate the TypeScript client from `openapi.json`, and generate
  that file in CI.

### D-08 Medium: API compile drops preset and shard_ext, and the CLI masks presets

- **Where:**
  - `api/routes/jobs.py:64-103`
  - `cli.py:197-200`
- **Issue:**
  - `submit_compile_job` never forwards `req.preset` or `req.shard_ext`.
  - The CLI also hard-codes `webp`, 92, 95 and `webp-lossless` into the
    payload. Even after the preset is forwarded, these values would override
    the preset (presets only fill `None` fields, per
    `core/manifold_compile.py:118-127`).
- **Fix:**
  - Send `None` for unset options.
  - Forward every field. A typed `CompileOptions` shared by the CLI, API and
    service removes this class of bug entirely.

### D-09 Medium: CLI `modernize` command is unreachable

- **Where:** `cli.py:426` (command) and `cli.py:1026` (sub-app of the same
  name).
- **Verified:** a minimal reproduction with Typer 0.27.3 gives
  `No such option: --dry-run`, because the group replaces the command.
- **Fix:** rename the command to `modernize all`, or fold it into
  `modernize_app`.

### D-10 Medium: `llm validate` crashes when validation fails

- **Where:** `cli.py:1017`.
- **Issue:** it reads `iss.invariant` and `iss.message`, but `ValidationIssue`
  defines `field` and `issue` (`tools/llm_helper/quality_validator.py:16-21`).
  The resulting `AttributeError` is caught at `cli.py:1021` and reported as a
  generic "Validation error".
- **Fix:** use `iss.field` and `iss.issue`. Make `ValidationIssue` a frozen
  dataclass so type checkers catch this.

### D-11 Medium: CWD-relative state and data paths

- **Where:**
  - `api/auth.py:16`
  - `api/server.py:33`
  - `cli.py:765`
  - `core/compiler_core.py:114,253-254`
  - `api/routes/modernize.py:55`
  - `api/routes/gui.py:271,620`
  - `api/routes/datasets.py:16`
- **Issue:**
  - The token and PID file are relative to the CWD, but `jobs.db` is anchored
    to the project root (`api/jobs.py:24-27`).
  - A CLI started from another directory creates a different token. It then
    gets 403 and silently falls back to in-process execution, and
    `server stop` cannot find the PID.
  - `raw-sets/`, `config.json` and the output root resolve against the CWD
    when compiling in-process.
- **Fix:** use `ProjectPaths` (Refactoring Plan item 1).

### D-12 Medium: task resolution ignores the registry `task` field

- **Where:** `core/compiler_core.py:336-365`.
- **Issue:**
  - Only `task_override` is consulted. Substring heuristics then classify:
    - `retinaface_mobilenet` as `pose`
    - `nima_authenticity` as `quality`, while `core/docs/metadata.py:66`
      says `authenticity`
    - `forex_universe` and `ai_helper_lemgend` as `detection`
  - The `task` written by custom-compile is ignored.
  - Whether `pose` is intended for RetinaFace landmarks is **uncertain**.
    Either way, the four task maps disagree.
- **Fix:** registry-driven `TaskResolver`, with the heuristic only as an
  explicit, logged fallback.

### D-13 Medium: dead fallback import of a deleted module

- **Where:** `sources/hf.py:125-126,137-138`.
- **Issue:** `except ImportError: from archive_manager import smart_extract`.
  `archive_manager.py` no longer exists, so the real `ImportError` from
  `utils.archive` is masked by a second `ModuleNotFoundError`. The PS1 has the
  same legacy fallback (`lemgendary_datasets_hub.ps1:119-121`).
- **Fix:** delete the fallback.

### D-14 Medium: GitHub clone failures are ignored

- **Where:** `sources/gh.py:22`.
- **Issue:** the `subprocess.run` return code is discarded, so the script
  always exits 0. ROBOTS.md forbids ignoring process exit codes.
- **Fix:** use `check=True` with handling for `CalledProcessError`, and a
  non-zero exit on failure.

### D-15 Medium: the Kaggle credential probe always succeeds

- **Where:** `sources/base.py:38-53`.
- **Issue:** the function sets `KAGGLE_USERNAME` to the default
  `"lemtreursi"` and then counts the username as a credential, so it always
  returns `True`.
- The personal handle is hard-coded in 9 places:
  - `api/routes/kaggle.py:205,273,318`
  - `core/common_sync.py:93,252,290,336,508,687`
- **Fix:** require a key or token. Read the owner from
  `_registry_metadata` (new field).

### D-16 Medium: CLI reports success when job status is unknown

- **Where:** `cli.py:100-102`.
- **Issue:** `_stream_job_logs` returns 0 when it cannot fetch the final job
  state.
- **Fix:** return a non-zero exit code and log at error level.

### D-17 Low: job manager connection leaks and cancel race

- **Where:** `api/jobs.py:49,98,144,262` and `api/jobs.py:153`.
- **Issue:**
  - `with sqlite3.connect()` commits but never closes the connection.
  - A job cancelled while PENDING is set back to RUNNING and executed anyway.
  - `log_file` stores an absolute path that is exposed in `JobResponse`.
- **Fix:**
  - Use `contextlib.closing`.
  - Use a conditional `UPDATE ... WHERE state = 'pending'`.
  - Store a relative log path.

### D-18 Low: stale dataset statistics cache

- **Where:** `api/routes/gui.py:47-56,180`.
- **Issue:** the cache key is only the top-level directory's mtime, which does
  not change when files inside the shard sub-directories change.
- **Fix:** key the cache on the mtimes of the scanned container directories,
  or add a TTL.

### D-19 Low: import-time side effects pollute the repository

- **Where:** `api/jobs.py:428` and `api/__init__.py:16`.
- **Verified:** running `tests/test_migration.py` and importing `cli.py` both
  created `.lgd_server/jobs.db` in the working tree. The reviewer deleted the
  directory afterwards.
- **Issue:** tests share the real job database with a running server.
- **Fix:** create the `JobManager` lazily in the app lifespan, with an
  injectable path. Use `tmp_path` in tests.

### D-20 Low: other defects

These are listed by location:

- `api/routes/kaggle.py:135`: `ValueError` returns 500 instead of 400.
- `core/compiler_core.py:156`: the remedy text names `python config_schema.py`.
  The file is at `core/config_schema.py`.
- `cli.py:947`: mojibake in a section comment (`# ???`).
- `utils/archive.py:148`: hard-coded 7-Zip install paths under Program Files. Use
  `shutil.which` plus a configured path.
- `core/manifold_compile.py:326`: unused `index` (ruff F841).
- `core/manifold_compile.py:761`: redefined `pbar` (ruff F811).
- `tools/stream_zip_to_container.py:243`: the closure does not bind loop
  variable `target_map` (ruff B023). This is a real late-binding hazard.

### ROBOTS.md Compliance Violations

- **Configuration-based suppression:** `.pylintrc` disables 48 checks,
  including:
  - `bare-except`
  - `broad-exception-caught`
  - `used-before-assignment`
  - `duplicate-except`
  - `unused-import`
  - `subprocess-run-check`
- **Policy conflict:** `POLICY.md:7,15,46` explicitly permits
  `# type: ignore[code]` and documented `except: pass`, which ROBOTS.md
  forbids.
- **Swallowed exceptions:**
  - 152 `except Exception` handlers.
  - An AST scan found 76 handlers with no log, no raise and no reporting call.
  - Explicit `pass` bodies at:
    - `api/routes/llm.py:47`
    - `tools/modernization_queue.py:112,129,203`
    - `utils/archive.py:181,186,193`
- **Absolute paths:**
  - `utils/archive.py:148`
  - `tests/test_api_and_services.py:214,257,299,340` (drive-letter literals)
  - `../lemgendary-docs/MD-Papers/PAPER_DATASET_COMPILER.md:289` and its HTML
    counterpart at line 726
- **Library code calls `sys.exit` or `os._exit`:**
  - `core/compiler_core.py:153,158`
  - `core/manifold_compile.py:136,154,841`
  - `forex/injector.py:115-128`
  - `sources/kaggle.py:110-136`
- **Subprocess to sibling Python:**
  - The API job routes (see Weaknesses).
  - `cli.py:272-299` and `api/routes/env.py:30,60` shell out to `lem-env`,
    even though the Environment Manager exposes a server
    (`../lemgendary-env-manager/env_manager/server.py`).

## Documentation Findings

### Whitepaper Structure

The required structure for a dataset-compiler paper is:

1. Abstract
2. High-Velocity Optimizations
3. Hybrid Cloud & Registry Integration
4. Multi-Modal & Format Resilience
5. Comparative Analysis / Benchmarks
6. Synthesis Flow & Topology
7. Unified Models Registry
8. Conclusion

`PAPER_DATASET_COMPILER.md` does **not comply**:

- Section 5 is "2026 Modernization Architecture & Innovations (v16.8.0)"
  (L166).
- Section 9 "Generic LLM Dataset Compiler" (L586) is an extra section.
- The Conclusion is section 10 (L670).
- There is an extra H2 at L3.
- "5.7" is used twice (L236, L243).
- **Fix:** fold the modernization content into sections 2 to 4 and the LLM
  compiler into sections 4 and 6, then renumber to exactly 8 sections.

LaTeX requirements are met in the MD:

- complexity bounds at L24-30
- space recovery at L38-41 and L79
- thresholds at L53-55 and L213

### HTML Counterpart

`../lemgendary-docs/papers/dataset-compiler.html` is **not synchronized**:

- It contains no LaTeX.
- The abstract differs from the MD.
- The version is "16.9.13" in the metadata block but v16.9.19 in the abstract.
- Section numbering is broken: there are two 5.14 headings and no 5.15.
- The table of contents omits section 9 and labels the Conclusion "9.".
- **Template is non-compliant:** there is no `<header>` and no `<footer>`, and
  the nav is `nav#toc.toc-fixed-bar` instead of `nav.toc-sidebar`.

### Paper Claims Against Code

Claims that are false or stale:

- `compiler/llm_compiler.py` does not exist.
- `KnowledgeIngestionEngine` does not exist; the class is `KnowledgeIngestor`.
- `content_sha256` does not appear in the code.
- `_read_stream` does not exist.
- `archive_manager.py` and `kaggle_manager.py` were removed.
- `compile_all` does not exist.
- Manifold counts disagree: the paper says 22, the registry has 21.
- §5.10 says two manifolds use `mds`. The registry now uses `webdataset`.
- Manifold names are inconsistent: `RetinaFaceMobileNet` versus `RetinaFace`,
  and three spellings of the multitask manifold.

Code features missing from the paper:

- the `rain`, `lowlight`, `downsample` and `film` degradation filters
- the `/api/modernize/*`, `/api/llm/*` and `/api/jobs/generic` endpoints
- the `modernize`, `llm`, `forex`, `test` and `sync metadata` CLI groups

### README

- **Version drift:**
  - The changelog's latest entry is v16.9.22 (L23).
  - The status table says v16.9.21-STABLE (L13).
  - v16.9.9 is missing from the changelog.
- **Corrupted text:** BEL control characters at L25 and L28 (`\x07pi.routes`).
- **Broken references:**
  - the link at L5
  - `modernization_roadmap.md` (L520)
  - `presets/`
  - a `test-all` command (L38)
  - direct-script paths at L631-655, which point to the root instead of
    `core/` or `tools/`; `migrate_registry.py` does not exist
- **Stale content:**
  - The architecture tree lists the removed `tools/ai_helper/` and omits
    `tools/llm_helper/`, `core/docs/` and the `llm` and `modernize` routes.
  - The CLI section omits the `forex`, `llm`, `modernize`, `test` and
    `sync metadata` commands.
- **Contradictions with the code:**
  - L300 claims a "zero-drift" `openapi.json`.
  - L73 and L320 claim zero suppressions, which contradicts `.pylintrc`.

### API Specification

`openapi.json` reports version 16.5.0 and is a generated file that has been
frozen.

- 16 operations served by the server are missing from it.
- The `CompileJobRequest` schema lacks `shard_ext`.
- `DatasetDetailStats` is missing 8 fields.

### Version Matrix

Seven different version strings are in use for this one project:

- `core/cli_args.py:29`: 16.9.5. This value is served by the API and by
  `cli.py version`.
- `openapi.json`: 16.5.0
- README status table: v16.9.21-STABLE
- README changelog: v16.9.22
- Paper: v16.9.19
- `documents.json`: v16.9.19-STABLE
- `canonical_capabilities`: v16.9.21-STABLE

### Manifests

- `canonical_capabilities.json` and `.yaml` are identical to each other.
- They list almost none of the compiler's capabilities.
- They record the CLI engine as "argparse / click"; it is actually Typer.
- `supported_formats.manifold` is wrong for `forex_predictor` (parquet) and
  `yolov8n` (directory).
- `documents.json:79`:
  - The title is "Dataset Compiler Whitepaper", which differs from the
    paper's own title.
  - The category is "01.1 COMPILER", while the paper says "01.2".

### markdownlint

markdownlint is not installed, so it was not run. A manual check found that
README and paper lines run up to 599 and 792 characters. MD013 fails unless a
documented configuration exists, and none was found.

## Cross-Project Findings

- **GUI (`../lemgendary-ai-studio-gui`):**
  - The GUI sends no auth headers, so it could not use the protected routers
    even if they were relevant.
  - It hard-codes the sidecar URL in `src/components/.../TestModal.tsx:79`,
    bypassing `VITE_DATASET_COMPILER_URL`.
  - Its fallback presets (`fallbackDatasets.ts:5-30`) have **zero overlap**
    with `presets.yaml`.
  - Its fallback dataset list duplicates the 21 keys of `unified_data.yaml`.
    It diverges in one `kaggle_ref` (`ai_helper_lemgend`).
  - Contract breaks are listed in D-05 and D-07.
  - `fetchDatasets` and `fetchCompilerPresets` swallow errors and return `[]`,
    so a server error looks like "offline".
- **Port 8100** is defined 9 times:
  - `api/server.py:123`
  - `cli.py:43`
  - GUI: `src/api/config.ts:39,54`, `src-tauri/src/commands.rs:32`,
    `vite.config.ts:56` and `TestModal.tsx:79`
  - env-manager: `env_manager/service_manager.py:32-33` and
    `env_manager/server.py:915,931`
  - training-suite: `training/utils/dataset_delegate.py:18`
  - Recommendation: the Environment Manager service registry should be the
    SSOT, and the other repos should read from it through environment
    variables.
- **Training suite:**
  - `core/manifold_compile.py:66-75` imports `training.notebooks` from
    `../lemgendary-training-suite` through `sys.path`. This gives one SSOT for
    notebooks, but couples the two repos at import level. A versioned package
    or an HTTP call to the 8200 sidecar would be cleaner.
  - `training/server/routes/gui.py` keeps its own per-model dataset map, which
    diverges from `unified_data.yaml`. It has no `classification_master_manifold`,
    `forex_universe` or `ai_helper_lemgend`, and it uses `retinaface` instead of
    `retinaface_mobilenet`.
- **Documentation Hub:**
  - The project name drifts: "Datasets Compilation Suite CLI" in
    `MANUAL_CLI.md:379` and `cli-manual.html:590`, and "Data Compiler" in
    `ECOSYSTEM_ARCHITECTURE.md:110`.
  - `LEMGENDARY_CURRENT_STATE.md:54` and `VERSIONING_POLICY.md:75` state
    v16.9.6.
  - `index.html` does not link `dataset-compiler.html` directly.
  - `MANUAL_API.md` omits `/api/jobs/generic`.
- **Environment Manager:** this repo shells out to the `lem-env` executable
  (`cli.py:272-299`, `api/routes/env.py`) instead of calling its typed
  server API.

## Proposed Changelog Entries

These entries are proposed for when the fixes land. None has been applied.

```markdown
### v16.9.23 - Security, Contract & Registry Integrity Hardening

- **`api/server.py`** - All mutating routers now require `verify_token`;
  CORS restricted to Tauri and dev origins.
- **`api/routes/llm.py`** - Async LLM compile jobs now start (route made
  `async`); removed silent `except RuntimeError: pass`.
- **`api/routes/gui.py`** - custom-compile validates presets and registry
  entries before persisting, preserves `unified_data.yaml` comments, rejects
  key collisions, and no longer maps `purge_loose_images` to `--cleanup`.
- **`api/routes/modernize.py`** - `datasets`, `dry_run`, `force` honoured;
  reduce routed to a non-interactive `ReduceService`.
- **`api/routes/jobs.py`** - compile jobs forward `preset` and `shard_ext`.
- **`cli.py`** - `modernize` command/group clash resolved; `llm validate`
  uses `ValidationIssue.field` / `.issue`; unknown job status exits non-zero.
- **`sources/`** - removed dead `archive_manager` fallback; `gh.py` checks
  `git clone` exit codes; Kaggle credential probe requires a key.
- **`paths.py`** - single project/state/datasets-root resolver; removed all
  CWD-relative state paths.
- **`.pylintrc` / `POLICY.md`** - aligned with ROBOTS.md zero-suppression.
- **`openapi.json`** - regenerated from `create_app()`; version 16.9.23.
```

## Remaining Recommendations

Ranked by priority:

1. Authenticate every mutating endpoint and restrict CORS (D-01). Then update
   the GUI client to send the token.
2. Fix the custom-compile path (D-02, D-03), the async LLM job start (D-04),
   and modernize run and reduce (D-05, D-06). These are user-facing features
   that currently cannot work, or that risk data loss.
3. Establish one GUI to API contract: generate `openapi.json` in CI and the GUI
   TypeScript types from it (D-07, D-08).
4. Introduce `paths.py` and a registry service. Remove the 10 or more
   datasets-root resolutions, the 4 task maps, the 5 default copies, and the
   CWD-relative state (D-11, D-12).
5. Unify execution with one typed `JobSpec` mapped to the service registry,
   shared by the API, CLI and hub. Remove `/api/jobs/generic` and the
   argv-to-script spawning.
6. Remove the configuration and policy suppressions (`.pylintrc`, `POLICY.md`)
   and fix the root causes. Triage the 76 silent exception handlers.
7. Apply OCP registries to formats, degradation filters, annotation detectors
   and dataset-type strategies. Decompose `core/manifold_compile.py`,
   `core/compiler_core.py`, `core/common_sync.py` and `cli.py`.
8. Restructure `PAPER_DATASET_COMPILER.md` to the mandated 8 sections. Rebuild
   the HTML from it, with LaTeX, `header`, `footer` and `nav.toc-sidebar`.
   Update the README, manifests and Hub documents to one version string.
9. Move the port, the Kaggle owner and the project inventory to the
   Environment Manager registry, and consume them through environment
   variables or its API.
10. Make the tests hermetic: use temporary state directories and no `C:/`
    literals. Add CI with ruff (141 current findings) and pyright.

## Tools Run

- **Syntax check** (`compile()` over all 158 `.py` files, in a scratch
  venv): 0 syntax errors.
- **ruff 0.17.3**, `--select F,E9,B,PLE`: 141 findings and no undefined names.
  The breakdown is:
  - 67 F401
  - 18 F541
  - 17 B007
  - 8 F841
  - 7 B009
  - 6 B904
  - 6 B010
  - 5 B905
  - 4 B008
  - 2 F811
  - 1 B023
- **unittest, per module, light dependencies only (no torch).**
  - Passed:
    - `test_docs`: 6 tests
    - `test_llm_compiler_pipeline`: 4 tests
    - `test_llm_helper`: 3 tests
    - `test_archive_stream`: 8 tests
    - `test_migration`: 1 test
  - `test_formats`: 15 tests ran, with 1 error because `pyarrow` was not
    installed and 1 skipped.
  - Not importable without heavy dependencies (`kagglehub`, `safetensors`),
    so not run:
    - `test_api_and_services`
    - `test_converters`
    - `test_kaggle_*`
    - `test_modernization_queue`
- **Reproductions:**
  - A Typer 0.27.3 command and group name clash (D-09).
  - FastAPI running a sync route without an event loop (D-04).
- **AST scan** for exception handlers that do not log or raise: 76 found.
- **markdownlint:** not available; the check was manual.
- **pyright:** could not be installed because the scratch path exceeded the
  Windows long-path limit.

## Remediation Status

Status of the Medium and Low findings on branch `fix/review-remediation`
as of v16.9.56 (step L10). Each finding was re-checked against the code
after the module splits of v16.9.36 to v16.9.51. The Critical and High
findings (D-01 to D-07) were fixed in v16.9.23 to v16.9.27.

| Finding | Status | Where |
| --- | --- | --- |
| D-08 preset/shard_ext dropped | Fixed | `3c5b577` (v16.9.55) |
| D-09 `modernize` unreachable | Fixed | `af686ae` (v16.9.26) |
| D-10 `llm validate` crash | Fixed | `1f51f09` (v16.9.25) |
| D-11 CWD-relative paths | Fixed | `3c5b577` (v16.9.55) |
| D-12 registry `task` ignored | Fixed | `3c5b577` (v16.9.55), v16.9.61 |
| D-13 `archive_manager` fallback | Fixed | `7054808` (v16.9.33), `3c5b577` |
| D-14 `git clone` exit code | Fixed | `4843416` (v16.9.29) |
| D-15 Kaggle probe always true | Fixed | `3c5b577` (v16.9.55) |
| D-16 unknown job status exits 0 | Fixed | `3c5b577` (v16.9.55) |
| D-17 job manager leaks and race | Fixed | `3c5b577` (v16.9.55) |
| D-18 stale stats cache | Fixed | `3c5b577` (v16.9.55) |
| D-19 import side effects | Fixed | `3c5b577` (v16.9.55) |
| D-20 other defects | Fixed | see below |

Notes:

- **D-08**: the API builds the compile argv from the same option table as
  the CLI. Unset encoding options stay unset, so presets apply.
- **D-11**: `core/paths.py` anchors the registry, raw sets, `config.json`,
  output root and server state on the project root.
- **D-12**: `detect_task` uses `task_override`, then `task`, then the name
  heuristic (logged). In v16.9.61 every `unified_data.yaml` entry sets `task`,
  and `task_taxonomy.json` is the one source of task ids, layout folders,
  label/mask strategies, heuristic patterns and documentation tasks (it
  replaces `_TASK_PATTERNS`, `_TASK_DIRS`, the strategy dicts and
  `MANIFOLD_TASK_MAP`). The compiler writes `task_type` into each
  `dataset_info.yaml`. The entries keep the tasks the heuristic gave them;
  `retinaface_mobilenet` (`pose`), `nima_authenticity` (`quality`),
  `forex_universe` and `ai_helper_lemgend` (`detection`) carry
  `task_confirmed: false` until the owner confirms them. The training suite's
  map moves to `task_type` in its own step.
- **D-13**: the Python fallback went in v16.9.33; the hub's
  `archive_manager.py` fallback and script-path call went in v16.9.55.
- **D-15**: the owner handle is one constant in `core/kaggle_identity.py`.
  The Markdown templates in `core/docs/templates.py` still spell it in
  their Kaggle URLs (document text, not configuration).
- **D-20**: `ValueError` returning 500 is fixed in `3c5b577`; the remedy
  text in `71000b4` (v16.9.31); the mojibake comment in `3c5b577`; the
  hard-coded 7-Zip paths in `1c78df7` (v16.9.45, environment variables plus
  `shutil.which`); the unused `index`, redefined `pbar` and B023 closure no
  longer exist (ruff reports 0 since `da9871c`).

Also closed in step L10:

- Seven compiler bugs found during the module splits, with regression tests
  (`b420c96`, v16.9.52).
- Ruff 259 to 0 (`da9871c`, v16.9.53) and pylint 0 findings with no
  disabled checks (`135aedd`, v16.9.54).

Closed in step L9 (v16.9.57):

- **CLI and API share the service layer** (Weaknesses, first item). API
  jobs no longer build argv for repository scripts. Each long-running job
  runs a service module as its own process (`python -m
  services.compiler_service`, `degrade_service`, `llm_service`,
  `migration_service queue|reduce`, `sync_service kaggle|manifold`). The
  arguments come from the option table that the module's `main` parses
  (`services/invocation.py`), and its `main` calls the same function the CLI
  calls in-process. `api/jobs.py` picks the interpreter and is the only
  place in the API that spawns a process. `POST /jobs/generic`,
  which ran the command it was given, was removed in v16.9.58. Short
  operations (the synchronous LLM compile, validation, registry and
  catalog reads) run in-process. `cli.py llm compile` is no longer started
  as a script.
- The hub (`lemgendary_datasets_hub.ps1`) is still a separate orchestration
  path, and the `lem-env` shell-out is left for step L7.

Left for step L7 (cross-repository duplication):

- Port 8100, defined in this repo (`api/server.py`, `commands/common.py`),
  the GUI, the Environment Manager and the training suite.
- Project inventories and the shell-out to `lem-env`
  (`commands/env.py`, `api/routes/env.py`) instead of the Environment
  Manager API.
- Model and dataset lists: the GUI fallback presets and datasets, and the
  training suite's per-model dataset map (`training/server/routes/gui.py`).
- Pipeline steps and task maps shared with the training suite, including
  the `training.notebooks` import in `core/compile_notebooks.py`.
- The Kaggle owner handle, also used by the other repositories.
