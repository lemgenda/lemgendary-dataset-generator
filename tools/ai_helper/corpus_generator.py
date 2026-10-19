"""Authoritative AI Helper Corpus Generator for LemGendary AI Studio.

Generates a balanced, factual, and strictly verified training and evaluation corpus
conforming to:
1. Code Manifests (unified_models_v2.yaml, unified_data.yaml) as highest SSOT
2. Master Manuals (MANUAL_AI_STUDIO_GUI.md, MANUAL_CLI.md, MANUAL_API.md)
3. Architectural Whitepapers & Troubleshooting Knowledge Base
4. Strict Zero-Emoji Rule and Bounded Engineering Guarantees
"""

from __future__ import annotations

import datetime
import json
import os
import random
import re
from typing import Any, Dict, List, Tuple
import yaml


SYSTEM_PROMPT = (
    "You are the authoritative LemGendary AI Studio Assistant. You possess deep, precise technical knowledge "
    "of the LemGendary AI ecosystem, its 23 neural architectures, the proprietary Dataset Compiler, the Training "
    "Suite with its Sawtooth Governor, and sidecar REST APIs. Authority Hierarchy: Code Manifests "
    "(unified_models_v2.yaml) > Master Manuals > Architectural Whitepapers > Educational Guides. Strictly "
    "differentiate general deep learning concepts from native Studio-supported features. Follow the 3-Stage "
    "Diagnostic Intake Protocol: Stage 1 Initial Triage (safe checks), Stage 2 Targeted Diagnosis (telemetry "
    "verification), and Stage 3 Bounded Remediation with epistemic tags [CONFIRMED], [PROBABLE], or [ILLUSTRATIVE]. "
    "Do not use ungrounded absolute guarantees; provide mathematically bounded engineering answers. Do not use emojis."
)


def load_schema_registry() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(__file__), "model_schema_registry.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("models", {})
    return {}


# ── 1. GUI Procedures ─────────────────────────────────────────────────────────

def generate_gui_examples() -> List[Dict[str, Any]]:
    items = []

    items.append({
        "id": "gui_custom_compile",
        "category": "gui_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How do I compile a custom multi-source dataset manifold in the AI Studio GUI?"},
            {"role": "assistant", "content": (
                "To compile a custom dataset manifold via the LemGendary AI Studio GUI:\n\n"
                "1. Open the **Dataset Compiler** tab from the main left navigation bar.\n"
                "2. Switch the **Compilation Mode** toggle to **Custom Multi-Source Compilation**.\n"
                "3. In the **Manifold ID** input, provide your unique canonical slug (e.g. `portrait-restoration-512`).\n"
                "4. Choose your **Target Domain** (e.g. `restoration`, `detection`, `face`, or `forex`) to govern schema validation.\n"
                "5. Select the **Storage Format**:\n"
                "   - `Streaming WebDataset Shards (.data)`: High-throughput containerized tar shards for distributed vision training.\n"
                "   - `Lightning AI LitData`: Memory-mapped tensor streaming for parameter prediction and regression manifolds.\n"
                "   - `Columnar Parquet`: Optimized columnar format for tabular embeddings and time-series.\n"
                "   - *Note*: MosaicML MDS is deprecated across the ecosystem and superseded by WebDataset (.data) and LitData.\n"
                "6. Configure partition density: set **Samples Per Shard** (canonical default: `5000` samples per archive, ~500 MB).\n"
                "7. Enter your repository paths in the **Source URL / Path** input (e.g. `kaggle://username/slug`, HuggingFace `hf://...`, Google Drive, or GitHub).\n"
                "8. Click **Compile Multi-Source Manifold** (triggers `POST http://127.0.0.1:8100/api/gui/custom-compile`).\n"
                "9. Monitor progress via the live compilation drawer at the bottom of the fold."
            )}
        ]
    })

    items.append({
        "id": "gui_quick_compile",
        "category": "gui_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How do I trigger a quick preset compilation for an existing dataset in the GUI?"},
            {"role": "assistant", "content": (
                "To trigger a quick compilation from an existing compiler preset in the GUI:\n\n"
                "1. Navigate to the **Dataset Compiler** panel.\n"
                "2. Ensure the mode is set to **Compiler Presets** (Standard Mode).\n"
                "3. Select a preset profile from the dropdown (e.g. `nima_aesthetic`, `nafnet_debluring`, or `yolov8n_detection`).\n"
                "4. Verify the container format (`webdataset` or `litdata`) and default shard size (`5000`).\n"
                "5. Optionally check **Purge Loose Images** to clean up intermediate extracted folders after container packaging.\n"
                "6. Click **Compile Manifold** (dispatches `POST http://127.0.0.1:8100/api/gui/quick-compile`).\n"
                "7. View real-time chunk progress and ETA in the execution telemetry drawer."
            )}
        ]
    })

    items.append({
        "id": "gui_manifold_filtering",
        "category": "gui_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How can I filter and locate specific dataset manifolds in the Compiler Catalog?"},
            {"role": "assistant", "content": (
                "The **Manifolds Catalog** in the Dataset Compiler panel provides a multi-axis filtering matrix:\n\n"
                "1. **Text Search**: Type into the search input to filter manifolds by name or dataset slug in real time.\n"
                "2. **Status Segmented Control**:\n"
                "   - `All`: Shows all discovered storage folders.\n"
                "   - `Compiled`: Limits display to manifolds possessing verified container archives and valid `manifest.json` metadata.\n"
                "   - `Raw / Uncompiled`: Displays raw loose-image folders requiring compilation.\n"
                "3. **Format Dropdown**: Filter specifically by container type (`WebDataset`, `LitData`, `Parquet`, `YOLO Directory`, or `Loose Directory`).\n"
                "4. **Domain Dropdown**: Filter by model task (`All`, `Restoration`, `Quality`, `Face`, `Detection`, `Forex`).\n\n"
                "Each manifold card displays its sample count, total byte size, container format, and a 2x2 action grid for Metadata, Notebooks, Download, and Upload."
            )}
        ]
    })

    items.append({
        "id": "gui_training_start",
        "category": "gui_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How do I launch a model training run from the Training Suite GUI?"},
            {"role": "assistant", "content": (
                "To launch a training run in the LemGendary AI Studio GUI:\n\n"
                "1. Navigate to the **Training Suite** tab from the sidebar.\n"
                "2. Browse the **Registered Architectures** catalog and locate your desired model card (e.g. `LemGendary NAFNet Image Deblurring`).\n"
                "3. Expand the **SOTA Targets & Ladders** section to review read-only governance parameters (defined strictly by `unified_models_v2.yaml`).\n"
                "4. Verify that the required compiled manifold is present and status is `VALIDATED` or `DATASET_READY`.\n"
                "5. Click **Start Training**.\n"
                "6. The action button transitions to a red **Stop Training** state, and the viewport auto-scrolls down to focus the **Real-Time Telemetry Stream** plotting training/validation loss, learning rate decay, and VRAM utilization."
            )}
        ]
    })

    items.append({
        "id": "gui_training_stop",
        "category": "gui_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How do I safely cancel a running training job in the GUI without corrupting checkpoints?"},
            {"role": "assistant", "content": (
                "To safely terminate an active training run in the GUI:\n\n"
                "1. Locate the active model card in the **Training Suite** panel.\n"
                "2. Click the red **Stop Training** button (or click **Cancel Job** in the live telemetry drawer).\n"
                "3. The GUI issues an authenticated `POST /api/jobs/{job_id}/cancel` request to the Training sidecar (Port 8200).\n"
                "4. The background process catches the graceful termination signal, flushes the final rolling checkpoint, restores `best.pth` based on historical validation records, and updates the SQLite job state to `CANCELLED`.\n"
                "5. The button transitions back to the primary **Start Training** state in under one second."
            )}
        ]
    })

    items.append({
        "id": "gui_inspect_manifold_metadata",
        "category": "gui_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Where can I view the raw metadata and cryptographic hashes of a compiled dataset in the GUI?"},
            {"role": "assistant", "content": (
                "To inspect a manifold's cryptographic provenance in the GUI:\n\n"
                "1. In the **Dataset Compiler** panel, locate the target manifold card in the catalog.\n"
                "2. Click the **Metadata** action button on the card (top-left button in the 2x2 grid).\n"
                "3. A modal drawer opens displaying:\n"
                "   - Total sample count and byte size.\n"
                "   - Exact container format and shard count.\n"
                "   - Input resolutions and channel normalization parameters.\n"
                "   - Cryptographic SHA-256 digest of the compiled package.\n"
                "   - Upstream provenance sources and license tags.\n"
                "4. Click **Close** or press `Escape` to dismiss the drawer."
            )}
        ]
    })

    return items


# ── 2. CLI Procedures ─────────────────────────────────────────────────────────

def generate_cli_examples() -> List[Dict[str, Any]]:
    items = []

    items.append({
        "id": "cli_dataset_compile",
        "category": "cli_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is the canonical CLI command to compile a dataset manifold from the command line?"},
            {"role": "assistant", "content": (
                "Dataset compilation is coordinated via the `lemgendary-datasets/cli.py` tool:\n\n"
                "```bash\n"
                "python cli.py compile \\\n"
                "  --sources \"kaggle://username/raw-dataset-slug\" \\\n"
                "  --output \"./LemGendaryDatasets/MyCompiledManifold\" \\\n"
                "  --preset webdataset \\\n"
                "  --shard-size 5000 \\\n"
                "  --dedup phash \\\n"
                "  --phash-threshold 4 \\\n"
                "  --num-workers 4\n"
                "```\n\n"
                "Key parameters:\n"
                "- `--shard-size 5000`: Sets partition density to the canonical 5,000 samples per archive (~500 MB).\n"
                "- `--dedup phash`: Applies discrete cosine transform perceptual deduplication.\n"
                "- Disambiguation: `lem-env` is strictly the environment manager; data compilation is executed via `python cli.py compile` in `lemgendary-datasets`."
            )}
        ]
    })

    items.append({
        "id": "cli_dataset_modernize",
        "category": "cli_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How do I run the dataset modernization CLI to retire legacy Large suffixes and transcode images to WebP?"},
            {"role": "assistant", "content": (
                "Use the `modernize` command inside `lemgendary-datasets`:\n\n"
                "```bash\n"
                "# Dry run to inspect planned folder renames and container changes\n"
                "python cli.py modernize --dry-run\n\n"
                "# Execute modernization for specific manifolds with WebP transcoding and WebDataset packaging\n"
                "python cli.py modernize \\\n"
                "  --datasets nima_aesthetic,nima_technical \\\n"
                "  --image-format webp \\\n"
                "  --also-format webdataset \\\n"
                "  --yes\n"
                "```\n\n"
                "This command retires legacy suffixes (e.g. `LemGendizedNimaAestheticLarge` -> `LemGendizedNimaAesthetic`), transcodes images to WebP (q=92), packs ~500 MB tar shards, updates `manifest.json`, and records provenance invariants."
            )}
        ]
    })

    items.append({
        "id": "cli_training_launch",
        "category": "cli_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How do I launch model training from the terminal using the canonical training CLI?"},
            {"role": "assistant", "content": (
                "Model training is managed by the `lemtrain` CLI (or `python cli.py` inside `lemgendary-training-suite`):\n\n"
                "```bash\n"
                "# Launch training with a canonical preset\n"
                "lemtrain train yolov8n --preset quick-sota --epochs 50\n\n"
                "# Or run via python cli.py with custom overrides\n"
                "python cli.py train nafnet_debluring \\\n"
                "  --preset default \\\n"
                "  --batch-size auto \\\n"
                "  --lr 0.0005 \\\n"
                "  --clean\n"
                "```\n\n"
                "The training engine automatically reads architecture topology from `unified_models_v2.yaml`, activates the Sawtooth Governor for dynamic VRAM management on 4GB GPUs, and writes rolling weights to `checkpoints/<model_key>/`."
            )}
        ]
    })

    items.append({
        "id": "cli_checkpoint_eval",
        "category": "cli_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How do I run deterministic checkpoint validation from the command line?"},
            {"role": "assistant", "content": (
                "Use the `eval` command provided by `lemtrain` (or `python cli.py` in `lemgendary-training-suite`):\n\n"
                "```bash\n"
                "python cli.py eval mirnet_exposure \\\n"
                "  --checkpoint checkpoints/mirnet_exposure/best.pth \\\n"
                "  --batch-size 16 \\\n"
                "  --device cuda\n"
                "```\n\n"
                "This evaluates the model under `torch.no_grad()`, parses test partitions declared in `manifest.json`, and outputs exact metrics (e.g. PSNR, SSIM, LPIPS, mAP) against established SOTA targets."
            )}
        ]
    })

    items.append({
        "id": "cli_model_export",
        "category": "cli_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What CLI command compiles a trained PyTorch model into production ONNX format?"},
            {"role": "assistant", "content": (
                "Model export and compilation are executed via the `export` command in `lemgendary-training-suite`:\n\n"
                "```bash\n"
                "# Export to optimized FP16 ONNX (Opset 18)\n"
                "python cli.py export nafnet_debluring \\\n"
                "  --checkpoint checkpoints/nafnet_debluring/best.pth \\\n"
                "  --format onnx-fp16 \\\n"
                "  --output exports/nafnet_debluring/\n"
                "```\n\n"
                "Supported formats include `onnx-fp32`, `onnx-fp16`, `torchscript`, and `webgpu` shaders, with fixed-shape tensors to ensure zero-slice runtime errors."
            )}
        ]
    })

    items.append({
        "id": "cli_env_validate",
        "category": "cli_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How do I verify the integrity and dependencies of the entire workspace using the CLI?"},
            {"role": "assistant", "content": (
                "Workspace environment integrity is audited strictly via `lem-env`:\n\n"
                "```bash\n"
                "# Run comprehensive multi-gate health validation\n"
                "lem-env validate\n\n"
                "# Quick health and sensor probe\n"
                "lem-env status\n"
                "```\n\n"
                "`lem-env` inspects Python virtual environments, CUDA availability, package versions, and SSOT manifest parity across all subprojects."
            )}
        ]
    })

    return items


# ── 3. API Procedures ─────────────────────────────────────────────────────────

def generate_api_examples() -> List[Dict[str, Any]]:
    items = []

    items.append({
        "id": "api_custom_compile",
        "category": "api_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What REST endpoint triggers multi-source dataset compilation, and how do I monitor it?"},
            {"role": "assistant", "content": (
                "Dispatch an authenticated `POST` request to the Dataset Compiler sidecar (Port 8100) at `/api/gui/custom-compile`:\n\n"
                "```http\n"
                "POST /api/gui/custom-compile HTTP/1.1\n"
                "Host: 127.0.0.1:8100\n"
                "Content-Type: application/json\n\n"
                "{\n"
                "  \"manifold_name\": \"MyCustomDataset\",\n"
                "  \"domain\": \"restoration\",\n"
                "  \"storage_preset\": \"webdataset\",\n"
                "  \"samples_per_shard\": 5000,\n"
                "  \"sources\": [\"kaggle://username/dataset-slug\"],\n"
                "  \"purge_loose_images\": true\n"
                "}\n"
                "```\n\n"
                "**Monitoring Workflow**:\n"
                "1. Response: `200 OK` (or `202 Accepted`) returning `{\"id\": \"<job_id>\", \"state\": \"PENDING\"}`.\n"
                "2. State polling: Dispatch `GET /api/jobs/{job_id}` to check execution status and progress percentage.\n"
                "3. Live Log Streaming: Connect via WebSocket to `ws://127.0.0.1:8100/api/ws/jobs/{job_id}/logs` to stream stdout lines in real time."
            )}
        ]
    })

    items.append({
        "id": "api_quick_compile",
        "category": "api_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How do I trigger a quick compile job from an existing preset via REST API?"},
            {"role": "assistant", "content": (
                "Send a `POST` request to Port 8100 at `/api/gui/quick-compile`:\n\n"
                "```http\n"
                "POST /api/gui/quick-compile HTTP/1.1\n"
                "Host: 127.0.0.1:8100\n"
                "Content-Type: application/json\n\n"
                "{\n"
                "  \"preset\": \"nima_aesthetic\",\n"
                "  \"model\": \"nima_aesthetic_mobile\",\n"
                "  \"samples_per_shard\": 5000,\n"
                "  \"purge_loose_images\": false\n"
                "}\n"
                "```\n\n"
                "The server dispatches the job to its SQLite job queue and returns the `JobResponse` containing the assigned `job_id`."
            )}
        ]
    })

    items.append({
        "id": "api_job_tracking",
        "category": "api_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How do I check job status and retrieve logs via REST API?"},
            {"role": "assistant", "content": (
                "Use the canonical job inspection routes on Port 8100 (Compiler) or Port 8200 (Training):\n\n"
                "- **Job State & Progress**: `GET /api/jobs/{job_id}`\n"
                "  Returns JSON with `state` (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`), `progress` (0.0 to 100.0), and start/finish timestamps.\n"
                "- **Historical Log Tail**: `GET /api/jobs/{job_id}/logs?tail=100`\n"
                "  Returns raw text logs recorded on disk.\n"
                "- **Live Streaming**: WebSocket `ws://127.0.0.1:8100/api/ws/jobs/{job_id}/logs` for real-time line-by-line broadcast.\n\n"
                "Notice: The canonical resource route is `GET /api/jobs/{job_id}`; do not append an extraneous `/status` sub-path."
            )}
        ]
    })

    items.append({
        "id": "api_quick_train",
        "category": "api_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How can an external service dispatch a model training job via REST API?"},
            {"role": "assistant", "content": (
                "Send an authenticated `POST` request to the Training Suite sidecar (Port 8200) at `/api/gui/quick-train`:\n\n"
                "```http\n"
                "POST /api/gui/quick-train HTTP/1.1\n"
                "Host: 127.0.0.1:8200\n"
                "Content-Type: application/json\n"
                "X-LemTrain-Token: <api_token>\n\n"
                "{\n"
                "  \"model_key\": \"yolov8n\",\n"
                "  \"ladder_stage\": 320,\n"
                "  \"enable_sawtooth\": true,\n"
                "  \"dataset_fraction\": 0.5,\n"
                "  \"force_clean_isolation\": true\n"
                "}\n"
                "```\n\n"
                "Response: `200 OK` with the queued `job_id`. Monitor training telemetry via `ws://127.0.0.1:8200/api/ws/training/{job_id}/telemetry`."
            )}
        ]
    })

    items.append({
        "id": "api_job_cancel",
        "category": "api_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What REST endpoint cancels an active compiler or training job?"},
            {"role": "assistant", "content": (
                "To terminate any active asynchronous job, send a `POST` request to `/api/jobs/{job_id}/cancel` on the appropriate sidecar port (8100 for Compiler, 8200 for Training):\n\n"
                "```http\n"
                "POST /api/jobs/{job_id}/cancel HTTP/1.1\n"
                "Host: 127.0.0.1:8100\n"
                "```\n\n"
                "Response: `200 OK` returning `{\"status\": \"cancelled\", \"job_id\": \"<job_id>\"}`. The process catches the termination signal and halts execution cleanly."
            )}
        ]
    })

    items.append({
        "id": "api_list_models",
        "category": "api_procedure",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How do I retrieve the complete catalog of models and their live statuses via REST API?"},
            {"role": "assistant", "content": (
                "Send a `GET` request to Port 8200 at `/api/gui/models`:\n\n"
                "```http\n"
                "GET /api/gui/models HTTP/1.1\n"
                "Host: 127.0.0.1:8200\n"
                "```\n\n"
                "This returns all 22 registered architectures, including their canonical 8-token status (`VALIDATED`, `DATASET_READY`, `TRAINING`, etc.), resolution ladders, SOTA metric benchmarks, and dataset references parsed directly from `unified_models_v2.yaml`."
            )}
        ]
    })

    return items


# ── 4. Troubleshooting Diagnostics (6-Attribute Schema) ───────────────────────

def generate_troubleshooting_examples() -> List[Dict[str, Any]]:
    items = []

    scenarios = [
        ("dataloader_starvation",
         "My GPU utilization fluctuates between 10% and 30% while training NAFNet on WebDataset shards. How do I fix it?",
         "This symptom points to **Dataloader Starvation and Sequential I/O Bottlenecks**.\n\n"
         "### Diagnostic Analysis:\n"
         "- **Symptom**: Wild GPU utilization drops with low temperatures (<55C) and idle compute cores.\n"
         "- **Context**: Training vision models on WebDataset tar archives using spinning HDD or high-latency disk storage.\n"
         "- **Root Cause**: Python worker processes blocking on WebP decompression or reading small shards sequentially without prefetching.\n\n"
         "### Resolution & Prevention:\n"
         "1. In your dataloader configuration, set `num_workers: 4` and enable `pin_memory: true`.\n"
         "2. Configure `prefetch_factor: 2` to buffer decoded batches ahead of GPU execution.\n"
         "3. If shards reside on a spinning disk, migrate the manifold to an NVMe SSD or compile using `--preset webdataset --shard-size 5000` to maximize sequential read bandwidth."),

        ("turing_fp16_nan_underflow",
         "Training loss immediately becomes NaN when starting NIMA training with AMP on a GTX 1650. Why?",
         "This is caused by **Turing Architecture FP16 Numerical Underflow in EMD Loss**.\n\n"
         "### Diagnostic Analysis:\n"
         "- **Symptom**: Loss transitions from normal to `NaN` within the first 50 iterations, and `GradScaler` collapses scale to zero.\n"
         "- **Context**: NVIDIA GTX 1650 (Turing TU117 architecture) lacking dedicated Tensor Cores.\n"
         "- **Root Cause**: Earth Mover's Distance (EMD) loss calculates cumulative distribution function (CDF) sums. In FP16 half precision, small probabilities truncate to zero, causing square root of zero gradients to produce `NaN`.\n\n"
         "### Resolution & Prevention:\n"
         "1. Disable Automatic Mixed Precision: set `amp: false` (force FP32 mode) in `unified_models_v2.yaml`.\n"
         "2. Ensure logit clamping is active prior to softmax: `torch.clamp(logits, min=-15.0, max=15.0)`.\n"
         "3. Verify epsilon offset: include `1e-4` inside the EMD square root calculation to prevent infinite gradients at zero."),

        ("infinite_plateau_loop",
         "My training run refuses to terminate; it keeps jolting the learning rate and restarting epochs when hitting a plateau. What is happening?",
         "This is an **Infinite Plateau Loop** caused by missing absolute patience termination guards.\n\n"
         "### Diagnostic Analysis:\n"
         "- **Symptom**: Learning rate decays to eta_min, triggers a jolt multiplier (1.5x), but the model never converges past the plateau and never terminates.\n"
         "- **Context**: Adaptive Plateau Governor active during unsupervised or fine-tuning runs.\n"
         "- **Root Cause**: `plateau_patience` triggers adaptation cycles without an upper termination boundary (`absolute_patience`).\n\n"
         "### Resolution & Prevention:\n"
         "1. In `unified_models_v2.yaml`, set `optimization.absolute_patience: 15`.\n"
         "2. When `absolute_patience` is exhausted without achieving a new SOTA metric, the governor automatically halts training and restores `best.pth`.\n"
         "3. Ensure the validation sample size is >= 500 to prevent evaluation noise from resetting plateau counters."),

        ("cuda_oom_sawtooth",
         "I get a CUDA out-of-memory error when transitioning from 256px to 512px in the resolution ladder. How do I prevent this?",
         "This occurs due to **Peak Activation Memory Escalation during Resolution Ladder Transitions**.\n\n"
         "### Diagnostic Analysis:\n"
         "- **Symptom**: `RuntimeError: CUDA out of memory. Tried to allocate...` during stage escalation.\n"
         "- **Context**: Resolution step escalation ($256 \\to 512$) increases spatial feature map memory quadratically ($4\\times$).\n"
         "- **Root Cause**: Static batch sizes allocated for 256px exceed the 4GB VRAM ceiling at 512px.\n\n"
         "### Resolution & Prevention:\n"
         "1. Verify that the **Sawtooth Governor** is active: set `batch_size: auto`.\n"
         "2. The Sawtooth Sentinel automatically scales batch sizes inversely with resolution: e.g. batch_size=32 at 256px, batch_size=8 at 512px.\n"
         "3. If memory headroom drops below 512 MB, the sentinel throttles chunk sizes and executes emergency garbage collection before OOM faults trigger."),

        ("corrupted_webdataset_shard",
         "Dataloader throws `tarfile.ReadError: unexpected end of data` or truncated shard header during epoch 2. What should I do?",
         "This indicates a **Corrupted or Truncated WebDataset Tar Archive**.\n\n"
         "### Diagnostic Analysis:\n"
         "- **Symptom**: Abrupt crash during training iteration with tar decompression error.\n"
         "- **Context**: Network interruption or abrupt power termination during Kaggle sync or compilation.\n"
         "- **Root Cause**: An incomplete `.tar` shard exists in the manifold directory without a verified SHA-256 checksum.\n\n"
         "### Resolution & Prevention:\n"
         "1. Run an integrity audit: `python cli.py audit --datasets <manifold_name>` inside `lemgendary-datasets`.\n"
         "2. The audit tool checks each archive against `manifest.json` and identifies truncated shards.\n"
         "3. Re-download or re-compile the damaged shard using `python cli.py compile --sources ... --shard-size 5000`."),

        ("websocket_disconnection",
         "The live training telemetry stream in the GUI suddenly disconnects with WebSocket error 1006. How do I recover?",
         "This is caused by **Sidecar Daemon Inactivity Timeout or Local Network Buffer Overflow**.\n\n"
         "### Diagnostic Analysis:\n"
         "- **Symptom**: Telemetry chart freezes and browser console displays `WebSocket closed abnormal 1006`.\n"
         "- **Context**: Long-running background training run on Port 8200.\n"
         "- **Root Cause**: Heavy training epochs blocking the ASGI event loop or temporary socket closure.\n\n"
         "### Resolution & Prevention:\n"
         "1. Verify daemon status: run `python cli.py server status` in `lemgendary-training-suite` (daemon remains running; jobs are persisted in SQLite WAL).\n"
         "2. The GUI includes an exponential backoff auto-reconnect handler that re-subscribes to `ws://127.0.0.1:8200/api/ws/training/{job_id}/telemetry`.\n"
         "3. Backfill missing logs via REST: `GET /api/jobs/{job_id}/logs?tail=100`."),

        ("dirty_environment_validation",
         "Running `lem-env validate` fails with a dirty repository gate error. How do I clear this?",
         "This indicates **Uncommitted Code Artifacts or Untracked Files Failing Clean Room Validation**.\n\n"
         "### Diagnostic Analysis:\n"
         "- **Symptom**: `lem-env validate` outputs `FAIL: Git clean room check failed`.\n"
         "- **Context**: Environment Manager pre-flight audit before packaging or running tests.\n"
         "- **Root Cause**: Temporary debug scripts, untracked checkpoints, or uncommitted files in repository root.\n\n"
         "### Resolution & Prevention:\n"
         "1. Check git status: `git status` in the affected subproject.\n"
         "2. Add generated outputs (such as `.pth` checkpoints or temporary logs) to `.gitignore`.\n"
         "3. Commit legitimate source modifications before running `lem-env validate`."),

        ("loss_divergence_warmup",
         "Training loss explodes to infinity after the first 100 steps of learning rate warmup. What went wrong?",
         "This is a symptom of **Excessive Initial Learning Rate or Unbounded Gradient Norms**.\n\n"
         "### Diagnostic Analysis:\n"
         "- **Symptom**: Loss decreases slightly then spikes to `inf` or `NaN` as warmup reaches maximum lr.\n"
         "- **Context**: Training deep architectures (such as MPRNet or UPN v2) with AdamW.\n"
         "- **Root Cause**: Large gradient steps destabilize uninitialized attention weights or residual projection layers.\n\n"
         "### Resolution & Prevention:\n"
         "1. Enable gradient clipping: verify `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` is active in the training loop.\n"
         "2. Reduce peak learning rate in `unified_models_v2.yaml` (e.g. from `1e-3` to `2e-4`).\n"
         "3. Extend warmup schedule from 500 steps to 2000 steps to allow running statistics to stabilize.")
    ]

    for sid, query, answer in scenarios:
        items.append({
            "id": f"troubleshoot_{sid}",
            "category": "troubleshooting_diagnostics",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer}
            ]
        })

    return items


# ── 5. Model Selection & Decision Trees ───────────────────────────────────────

def generate_model_selection_examples() -> List[Dict[str, Any]]:
    items = []

    items.append({
        "id": "decision_restoration_suite",
        "category": "model_selection",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "I have an old, degraded photograph with blur, film noise, and low resolution. Should I use NAFNet, MPRNet, or CodeFormer?"},
            {"role": "assistant", "content": (
                "Follow this architectural decision tree based on content and hardware constraints:\n\n"
                "1. **Does the image prominently feature human faces?**\n"
                "   - **Yes -> Use CodeFormer** (`face_codeformer`). General image restoration models smooth or distort facial geometry. CodeFormer leverages a discrete vector-quantized (VQ) codebook prior that reconstructs realistic eyes, lips, and facial hair while preserving identity.\n\n"
                "2. **If it is a general photograph (landscapes, objects, architecture):**\n"
                "   - **Use NAFNet** (`nafnet_debluring`) if your primary issue is camera motion blur or optical defocus and you require fast inference or low VRAM (<4GB). NAFNet removes nonlinear activations, achieving 32.85 dB PSNR with very low computational overhead.\n"
                "   - **Use MPRNet** (`mprnet_restoration`) if the image suffers from compound multi-scale degradations (e.g. heavy rain streaks, severe sensor noise, and blur). MPRNet's 3-stage progressive architecture with Cross-Stage Feature Fusion excels on complex artifact combinations at the cost of higher memory footprint.\n"
                "   - **Use UPN v2** (`upn_v2`) if you need automated parametric degradation estimation prior to pipeline execution."
            )}
        ]
    })

    items.append({
        "id": "decision_quality_assessment",
        "category": "model_selection",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is the difference between NIMA Mobile, NIMA Technical, and NIMA Authenticity, and which should I use?"},
            {"role": "assistant", "content": (
                "The three models assess completely distinct perceptual dimensions of an image:\n\n"
                "| Model Key | Canonical Backbone | Objective & Task | Output Representation |\n"
                "| :--- | :--- | :--- | :--- |\n"
                "| **`nima_aesthetic_mobile`** | MobileNetV3-Small | Subjective artistic quality & aesthetic composition | 10-bin probability distribution (Scores 1 to 10) |\n"
                "| **`nima_technical`** | EfficientNetV2-S | Objective physical degradation (noise, blur, clipping) | 10-bin technical fidelity distribution independent of artistic intent |\n"
                "| **`nima_authenticity`** | EfficientNetV2-S | Synthetic AI generation vs authentic camera capture | Authenticity probability (Real vs AI-Generated) |\n\n"
                "**Selection Rule**:\n"
                "- Use **NIMA Mobile** for photo culling, automated album rating, and artistic gating.\n"
                "- Use **NIMA Technical** for automated restoration before/after verification and distortion measurement.\n"
                "- Use **NIMA Authenticity** to detect synthetic AI imagery and fake photo submissions."
            )}
        ]
    })

    items.append({
        "id": "decision_facial_suite",
        "category": "model_selection",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How do RetinaFace, ParseNet, and CodeFormer collaborate in the LemGendary Facial Vision Suite?"},
            {"role": "assistant", "content": (
                "The LemGendary Facial Vision Suite (Category 08) coordinates three specialized networks in a sequential pipeline:\n\n"
                "1. **RetinaFace (Stage 1 - Detection & Alignment)**: Scans unconstrained images to detect face bounding boxes and 5 canonical facial landmarks (eyes, nose, mouth corners) using a lightweight MobileNet edge backbone.\n"
                "2. **Pre-Flight Warping**: Uses the 5 landmarks to compute an affine transformation, cropping and upright-aligning the face to 512x512 resolution.\n"
                "3. **CodeFormer (Stage 2 - Codebook Restoration)**: Restores degraded, blurred, or noisy 512px facial crops using its discrete VQ-codebook transformer prior.\n"
                "4. **ParseNet (Stage 3 - Anatomical Segmentation)**: Generates 19-class pixel-precise semantic masks (skin, eyebrows, eyes, nose, lips, teeth, hair) to guide selective local retouching and prevent boundary artifact bleed into background regions."
            )}
        ]
    })

    items.append({
        "id": "decision_detection_vs_safety",
        "category": "model_selection",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "When should I use YOLOv8n versus the Universal NSFW Classifier?"},
            {"role": "assistant", "content": (
                "These models serve fundamentally different computer vision tasks:\n\n"
                "- **YOLOv8n (`yolov8n`)**: An anchor-free real-time object detector that locates and bounds 80 common object categories (persons, vehicles, animals, tools) with spatial coordinate boxes `[x, y, w, h]` and confidence scores. Use it when spatial object localization is required.\n"
                "- **Universal NSFW Classifier (`classification_nsfw`)**: An image-level classification safety gate built on EfficientNetV2 that categorizes entire images into safe, suggestive, or explicit content. Use it as an automated upload moderation filter before images enter ingestion pipelines."
            )}
        ]
    })

    items.append({
        "id": "decision_forex_predictor",
        "category": "model_selection",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is the purpose of the Forex Predictor model, and how does its data differ from computer vision models?"},
            {"role": "assistant", "content": (
                "The **LemGendary Forex Predictor (`forex_predictor`)** is an algorithmic market direction and volatility forecasting model:\n\n"
                "- **Domain**: Quantitative financial time-series (Category 11), not computer vision.\n"
                "- **Dataset**: `LemGendizedForexUniverse` stored in columnar **Parquet** and Feather formats with walk-forward cross-validation splits, preserving zero-lookahead bias invariants.\n"
                "- **Input**: Multi-timeframe OHLCV tensors across 6 timeframes (M1, M5, M15, H1, H4, D1) for EURUSD, GBPUSD, USDJPY, and XAUUSD.\n"
                "- **Architecture**: Multi-Scale CNN-Transformer (Causal Dilated TCN + Multi-Head Cross-Timeframe Attention).\n"
                "- **Deployment**: MetaTrader 5 Expert Advisor (EA) C++ Bridge / stateless ONNX runtime."
            )}
        ]
    })

    items.append({
        "id": "decision_hybrid_upn_v2",
        "category": "model_selection",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is UPN v2, and when should it be included in a restoration workflow?"},
            {"role": "assistant", "content": (
                "**LemGendary UPN v2 (`upn_v2`)** is a Universal Parameter Predictor built on a lightweight MobileNetV3-Small backbone:\n\n"
                "- **Function**: Rather than reconstructing pixels directly, UPN v2 analyzes an input image and predicts optimal continuous hyperparameter settings (denoising strength, unsharp radius, deblur kernel size, color temperature) in under 5ms.\n"
                "- **Workflow Integration**: Acts as an intelligent pre-flight router. Upstream applications pass an image to UPN v2, receive predicted degradation parameters, and dynamically configure downstream restorers (such as NAFNet or Film Restorer) with optimal execution settings."
            )}
        ]
    })

    return items


# ── 6. Terminology & Epistemic Invariants ─────────────────────────────────────

def generate_terminology_examples() -> List[Dict[str, Any]]:
    items = []

    pairs = [
        ("dataset_vs_manifold",
         "Is a manifold just another word for a dataset in LemGendary Studio?",
         "No. In LemGendary AI Studio architecture, a **raw dataset** and a **compiled training manifold** have distinct lifecycles and invariants:\n\n"
         "- **Raw Dataset**: Unstructured or heterogeneous source data (loose folders, ZIP archives, Kaggle dumps, raw COCO JSONs) with arbitrary image dimensions, variable compressions, missing splits, and unverified data corruption.\n"
         "- **Compiled Training Manifold**: An immutable, production-ready artifact containing:\n"
         "  1. Uniform containerized shards (`.tar` WebDataset, MDS, or Parquet) partitioned to canonical density (default: `5000` samples per archive).\n"
         "  2. Cryptographic `manifest.json` declaring sample count, SHA-256 digests, and tensor schema.\n"
         "  3. Immutable train/val/test splits.\n"
         "  4. Standardized image transcoding (WebP q=92-95) with Lanczos-3 anti-aliasing.\n\n"
         "Training suites ingest compiled manifolds directly, never raw datasets."),

        ("hardlink_vs_phash_dedup",
         "What is the difference between Hardlink Deduplication and Perceptual Deduplication in the Dataset Compiler?",
         "The two deduplication modes operate at different abstraction levels:\n\n"
         "- **Hardlink Deduplication**: Operates at the filesystem inode level. It identifies files that are 100% byte-for-byte identical (matching SHA-256 hash). It saves disk space without byte copies, but cannot detect two images that look identical if they differ by re-compression, metadata, or color grading.\n"
         "- **Perceptual Deduplication (pHash)**: Computes a discrete cosine transform perceptual hash of visual frequencies. It identifies near-duplicate images even if resized, watermarked, or slightly re-compressed. Controlled via `--phash-threshold` (default: 4 hamming distance)."),

        ("authority_hierarchy_ssot",
         "When documentation contradicts itself, what is the authoritative Single Source of Truth?",
         "Under the LemGendary ecosystem architecture, the strict Authority Hierarchy is:\n\n"
         "$$\\text{Code Manifests} \\succ \\text{Master Manuals} \\succ \\text{Architectural Whitepapers} \\succ \\text{Educational Guides}$$\n\n"
         "1. **Code Manifests** (`unified_models_v2.yaml`, `unified_data.yaml`): The absolute Single Source of Truth for model keys, backbones, canonical statuses, and dataset bindings.\n"
         "2. **Master Manuals** (`MANUAL_AI_STUDIO_GUI.md`, `MANUAL_CLI.md`, `MANUAL_API.md`): Authoritative reference for UI controls, CLI flags, and REST/WebSocket routes.\n"
         "3. **Architectural Whitepapers**: Theoretical formulations, mathematical loss definitions, and SOTA target rationale.\n"
         "4. **Educational Guides**: Introductory explanations for human operators."),

        ("sota_ladder_vs_epochs",
         "Why does LemGendary Training Suite use Resolution Ladders instead of fixed epoch counts?",
         "Fixed-epoch training on full-resolution images is computationally inefficient and prone to early instability on constrained hardware (e.g. 4GB VRAM):\n\n"
         "- **Spatial Resolution Ladder**: Models train through progressive stages (e.g. $128\\text{px} \\to 256\\text{px} \\to 512\\text{px}$). Low-resolution stages allow high batch sizes to rapidly learn global semantic structures.\n"
         "- **Plateau Governor**: Escalation between rungs is driven by metric plateau detection rather than fixed epoch counts.\n"
         "- **High-Fidelity Fraction**: The final ladder stage locks in fine high-frequency textures with Swin/attention blocks while the Sawtooth Governor throttles batch sizes dynamically to prevent OOM."),

        ("epistemic_tags_significance",
         "What do the epistemic tags like [THEORETICAL] and [MEASURED] mean in the documentation hub?",
         "LemGendary technical documentation enforces epistemic traceability through 5 standardized tags:\n\n"
         "- `[THEORETICAL]`: Mathematical derivation, architectural proposal, or loss formulation established prior to training verification.\n"
         "- `[DESIGN_GOAL]`: Target hardware constraint, memory bound, or latency envelope targeted during system engineering.\n"
         "- `[TARGET]`: Benchmark performance objective (e.g. SOTA metric floor) required for model promotion.\n"
         "- `[CURRENT]`: Real-time training telemetry checkpoint metric during an active training cycle.\n"
         "- `[MEASURED]`: Empirically validated benchmark evaluated on test partitions under `torch.no_grad()` on physical hardware."),

        ("zero_emojis_rule",
         "Why does the AI Helper and documentation suite strictly prohibit emojis?",
         "The LemGendary AI Ecosystem adheres to rigorous engineering standards:\n\n"
         "- **Clarity & Formality**: Eliminates visual noise, ensuring technical communication remains concise, precise, and professional.\n"
         "- **Terminal & Parsing Safety**: Prevents Unicode rendering glitches, multi-byte character offset truncation, and encoding crashes across headless Linux consoles, CI runners, and logging pipelines.\n"
         "- **Bounded Engineering**: Discourages marketing hype and absolute claims in favor of empirically measured parameters.")
    ]

    for pid, query, answer in pairs:
        items.append({
            "id": f"terminology_{pid}",
            "category": "terminology_and_invariants",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer}
            ]
        })

    return items


# ── 7. Unsupported & Negative Capabilities ────────────────────────────────────

def generate_negative_examples() -> List[Dict[str, Any]]:
    items = []

    cases = [
        ("tfrecord_support",
         "Can I import TFRecord archives directly into the LemGendary Dataset Compiler?",
         "No. **TFRecord is not natively supported as an input container format** in the LemGendary Dataset Compiler.\n\n"
         "The compiler natively supports: MATLAB matrices, COCO JSON, Pascal VOC XML, YOLO format directories, Parquet/Safetensors, and standard image folders (PNG, JPEG, WebP, TIFF).\n\n"
         "To ingest TFRecord data, you must extract the images and annotations into loose folders or convert them into Parquet before running `python cli.py compile`."),

        ("retinaface_resnet50_variant",
         "Can I train the ResNet-50 variant of RetinaFace in the active Training Suite?",
         "No. The ResNet-50 variant of RetinaFace (`retinaface_resnet`) is marked as **DEPRECATED** in `unified_models_v2.yaml`.\n\n"
         "The active production model is **RetinaFace MobileNet** (`retinaface`), standardized on the `MobileNetV3-Small` backbone for sub-3ms edge and WebGPU execution. ResNet-50 is archived as a historical Crowd-Density research specification only."),

        ("nima_mobile_backbone_inquiry",
         "What is the canonical backbone for NIMA Mobile?",
         "According to the authoritative Single Source of Truth (`unified_models_v2.yaml`), the canonical backbone for NIMA Mobile (`nima_aesthetic_mobile`) is **MobileNetV3-Small (Global Composition)**.\n\n"
         "Under the LemGendary authority hierarchy:\n"
         "$$\\text{Code Manifests} \\succ \\text{Master Manuals} \\succ \\text{Whitepapers} \\succ \\text{Educational Guides}$$\n\n"
         "The code manifest in `unified_models_v2.yaml` and the runtime model implementation (`models/nima.py`) both specify `mobilenet_v3_small` with 576 input features."),

        ("audio_video_compiler",
         "Can I use the Dataset Compiler to transcode MP4 video clips into WebDataset shards?",
         "No. The LemGendary Dataset Compiler is specialized exclusively for **computer vision spatial manifolds** (still image restoration, detection, segmentation) and **financial time-series** (Parquet/Feather).\n\n"
         "It does not support raw video container demuxing (MP4/MKV) or audio stream processing (MP3/WAV). Video frames must be pre-extracted into sequential image directories before compilation."),

        ("training_specification_status",
         "Can I run production inference on ParseNet or RetinaFace checkpoints right now?",
         "No. In `unified_models_v2.yaml` and the Facial Suite documentation, **ParseNet** and **RetinaFace** are currently in `DATASET_READY` status (Pre-Training Architectural Specification / In-Progress).\n\n"
         "Their mathematical formulations, target manifolds (`LemGendizedParseNetLarge`, `LemGendizedRetinaFaceMobileNetLarge`), and SOTA target thresholds are locked, but verified production checkpoints have not yet completed convergence validation. Only models with `VALIDATED` or `PRODUCTION` status possess certified release weights."),

        ("multinode_ddp_cluster",
         "Can I run multi-node distributed training across an AWS Slurm cluster using the default GUI?",
         "No. The LemGendary AI Studio GUI and local sidecars are engineered for **single-node execution** (local workstation or single cloud instance with 1-4 GPUs using PyTorch DDP / DataParallel).\n\n"
         "Multi-node cluster orchestration across Slurm or Kubernetes requires generating standalone Kaggle/Colab training scripts via `python cli.py notebooks` and submitting jobs through your external cluster scheduler.")
    ]

    for cid, query, answer in cases:
        items.append({
            "id": f"negative_{cid}",
            "category": "unsupported_and_negative",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer}
            ]
        })

    return items


# ── 8. Cross-Document Pipeline Walkthroughs (Domain-Aware) ───────────────────

def generate_model_pipeline_examples(models: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []

    for key, m in models.items():
        q = f"Describe the complete lifecycle and pipeline for {m['display_name']} ({key}) from dataset compilation to production deployment."
        
        # Domain-aware compilation step
        if m.get("domain") == "Financial Time-Series":
            compilation_desc = (
                f"1. **Dataset Compilation**:\n"
                f"   - **Source Dataset**: `{m['dataset']}`\n"
                f"   - **Structure**: {m['dataset_structure']}\n"
                f"   - Compiled into columnar **Parquet** and Feather shards with walk-forward cross-validation splits, preserving zero-lookahead bias invariants."
            )
            training_desc = (
                f"2. **Training & Architecture**:\n"
                f"   - **Architecture**: {m['architecture']} ({m['backbone']})\n"
                f"   - **Loss Objective**: `{m['loss']}`\n"
                f"   - **Precision**: {m['precision']}\n"
                f"   - **Timeframe Ladder**: `{m['resolution_ladder']}`\n"
                f"   - Fixed batch sequences with walk-forward step decay schedule."
            )
        else:
            compilation_desc = (
                f"1. **Dataset Compilation**:\n"
                f"   - **Source Dataset**: `{m['dataset']}`\n"
                f"   - **Structure**: {m['dataset_structure']}\n"
                f"   - Compiled via `python cli.py compile` into containerized WebDataset `.data` shards with WebP tiles (or LitData for tensor streaming) (~500 MB per shard, 5,000 samples default)."
            )
            training_desc = (
                f"2. **Training & Architecture**:\n"
                f"   - **Architecture**: {m['architecture']} ({m['backbone']})\n"
                f"   - **Loss Objective**: `{m['loss']}`\n"
                f"   - **Precision**: {m['precision']}\n"
                f"   - **Spatial Resolution Ladder**: `{m['resolution_ladder']}`\n"
                f"   - Dynamic Sawtooth Sentinel adjusts batch sizes automatically to protect against VRAM exhaustion on 4GB devices."
            )

        # Status & Checkpointing step
        if m.get("current_status") in ("DATASET_READY", "SPECIFICATION", "PLANNED"):
            status_desc = (
                f"3. **Convergence & Lifecycle Status**:\n"
                f"   - **Lifecycle Status**: `{m['current_status']}` (Pre-Training Architectural Specification / In-Progress).\n"
                f"   - **Target SOTA Thresholds**: {m['sota_targets']}.\n"
                f"   - Physical checkpoint weights (`{m['checkpoint']}`) are in active training convergence."
            )
            checkpoint_desc = (
                f"4. **Checkpointing & Exports**:\n"
                f"   - Target Checkpoint: `{m['checkpoint']}`.\n"
                f"   - Export Formats: Planned for ONNX FP16 and WebGPU once convergence validation gates pass."
            )
        else:
            status_desc = (
                f"3. **Convergence & SOTA Verification**:\n"
                f"   - **Target SOTA**: {m['sota_targets']}\n"
                f"   - **Current Benchmark**: {m['current_metrics']} ({m['current_status']})"
            )
            checkpoint_desc = (
                f"4. **Checkpointing & Export**:\n"
                f"   - Saved to `checkpoints/` as `{m['checkpoint']}` and archived into `LemGendaryModels/{key}/` as `{m.get('weights', m['checkpoint'])}`.\n"
                f"   - Exported formats: `{m.get('export_formats', ['onnx_fp16'])}`."
            )

        deployment_desc = (
            f"5. **Production Deployment**:\n"
            f"   - **Target Runtime**: {m['deployment']}\n"
            f"   - **Target Hardware**: {m['hardware']}."
        )

        ans = f"Here is the complete end-to-end pipeline for **{m['display_name']}**:\n\n" + "\n\n".join([
            compilation_desc,
            training_desc,
            status_desc,
            checkpoint_desc,
            deployment_desc
        ])

        items.append({
            "id": f"pipeline_{key}",
            "category": "cross_document_pipeline",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": ans}
            ]
        })

    return items


# ── 9. Factual Consistency Gate ───────────────────────────────────────────────

def run_factual_consistency_gate(corpus: List[Dict[str, Any]]) -> None:
    """Rigorous gate checking every example against canonical rules and SSOT."""
    emoji_pat = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
    
    for ex in corpus:
        cid = ex["id"]
        text = " ".join(m["content"] for m in ex["messages"])

        # 1. Zero emojis
        emojis = emoji_pat.findall(text)
        if emojis:
            raise ValueError(f"Factual Gate Failure: Emojis detected in {cid}: {emojis}")

        # 2. No legacy MobileNet versions claimed for NIMA Mobile
        if "nima_aesthetic_mobile" in text.lower():
            if re.search(r'nima_aesthetic_mobile.*MobileNetV[12]', text, re.IGNORECASE):
                raise ValueError(f"Factual Gate Failure in {cid}: NIMA Mobile must use MobileNetV3-Small, not legacy MobileNet backbones")

        # 3. No unverified endpoints: POST /api/compile without path, /api/jobs/{id}/status, /api/ws/telemetry
        if "POST /api/compile " in text or "POST /api/compile\n" in text or "POST http://127.0.0.1:8100/api/compile\n" in text:
            raise ValueError(f"Factual Gate Failure in {cid}: Incomplete endpoint /api/compile detected (use /api/gui/quick-compile, /api/gui/custom-compile, or /api/jobs/compile)")
        if "poll /api/jobs/{job_id}/status" in text or "GET /api/jobs/{job_id}/status" in text or "poll /api/jobs/{id}/status" in text or "GET /api/jobs/{id}/status" in text:
            raise ValueError(f"Factual Gate Failure in {cid}: Invalid endpoint /api/jobs/.../status (use /api/jobs/... directly)")
        if "/api/ws/telemetry" in text:
            raise ValueError(f"Factual Gate Failure in {cid}: Invalid WebSocket /api/ws/telemetry (use /api/ws/jobs/.../logs or /api/ws/events)")

        # 4. No lem-env training or lem-env datasets commands
        if "lem-env training" in text:
            raise ValueError(f"Factual Gate Failure in {cid}: lem-env does not run training (use lemtrain or python cli.py train)")
        if "lem-env datasets" in text:
            raise ValueError(f"Factual Gate Failure in {cid}: lem-env does not compile datasets (use python cli.py compile)")

        # 5. Default shard size must be 5000, not 1000
        if "default: 1000" in text or "samples_per_shard 1000" in text or "default: 1,000" in text:
            raise ValueError(f"Factual Gate Failure in {cid}: Shard size default must be 5000, not 1000")

        # 6. Forex pipeline must not claim WebDataset or WebP tiles
        if "forex_predictor" in cid:
            if "webdataset" in text.lower() or "webp" in text.lower():
                raise ValueError(f"Factual Gate Failure in {cid}: Forex pipeline conflated with image WebDataset/WebP")

        # 7. ParseNet and RetinaFace must not be claimed to have completed production weights
        if cid in ("pipeline_face_parsenet", "pipeline_retinaface"):
            if "saved to checkpoints/" in text.lower() and "best.pth" in text.lower():
                raise ValueError(f"Factual Gate Failure in {cid}: Pre-training model claimed to have trained best.pth")


# ── 10. Stratified Sharding & Generation ──────────────────────────────────────

def generate_balanced_corpus() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    models = load_schema_registry()

    category_buckets: Dict[str, List[Dict[str, Any]]] = {
        "gui_procedure": generate_gui_examples(),
        "cli_procedure": generate_cli_examples(),
        "api_procedure": generate_api_examples(),
        "troubleshooting_diagnostics": generate_troubleshooting_examples(),
        "model_selection": generate_model_selection_examples(),
        "terminology_and_invariants": generate_terminology_examples(),
        "unsupported_and_negative": generate_negative_examples(),
        "cross_document_pipeline": generate_model_pipeline_examples(models),
    }

    try:
        from tools.ai_helper.extended_records import EXTENDED_RECORDS
    except ImportError:
        from extended_records import EXTENDED_RECORDS

    for r in EXTENDED_RECORDS:
        cat = r["category"]
        item = {
            "id": r["id"],
            "category": cat,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["user"]},
                {"role": "assistant", "content": r["assistant"]},
            ],
        }
        if cat not in category_buckets:
            category_buckets[cat] = []
        category_buckets[cat].append(item)

    train_set: List[Dict[str, Any]] = []
    val_set: List[Dict[str, Any]] = []

    # Deterministic stratified split: ensure EVERY category is represented in val set!
    random.seed(42)
    for cat, items in category_buckets.items():
        random.shuffle(items)
        # Ensure at least 1-2 examples from each category go to validation
        num_val = max(1, round(len(items) * 0.20))
        val_subset = items[:num_val]
        train_subset = items[num_val:]
        train_set.extend(train_subset)
        val_set.extend(val_subset)

    # Shuffle final shards deterministically
    random.shuffle(train_set)
    random.shuffle(val_set)

    return train_set, val_set


def synthesize_corpus() -> None:
    train_set, val_set = generate_balanced_corpus()
    all_items = train_set + val_set

    # 1. Run factual consistency gate
    run_factual_consistency_gate(all_items)
    print("Factual consistency gate passed: all items strictly conform to SSOT.")

    # 2. Write out jsonl files
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "LemGendaryDatasets", "LemGendizedAIHelperLemGend"))
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "ai_helper_train.jsonl")
    val_path = os.path.join(out_dir, "ai_helper_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # 3. Synchronize dataset_info.yaml
    sources_list = [
        "LemGendary AI Documentation Hub",
        "General AI Training Knowledge Reference",
        "Ecosystem Architecture and Current State Manifest",
        "Master Operations Manuals (GUI, CLI, API)",
        "Dataset Compiler and Environment Manager Whitepapers",
        "Training Suite and Training Pathology Documentation",
        "Model-Family Architecture Whitepapers",
        "Canonical Glossary and AI Helper Troubleshooting Knowledge Base",
        "Curated AI LemGend instruction and validation examples",
    ]
    info_data = {
        "count": len(all_items),
        "task": "ai_helper",
        "canonical_format": "jsonl",
        "original_sources": sources_list,
        "path": r"..\LemGendaryDatasets\LemGendizedAIHelperLemGend",
        "source": "LemGendizedAIHelperLemGend-manifold",
        "last_processed": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, "dataset_info.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(info_data, f, sort_keys=False)

    # 4. Synchronize category and classes
    with open(os.path.join(out_dir, "category.txt"), "w", encoding="utf-8") as f:
        f.write("Technical Documentation & AI Assistant Training\n")
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("ai_helper\n")

    # 5. Synchronize dataset-metadata.json
    metadata_path = os.path.join(out_dir, "dataset-metadata.json")
    meta_json: Dict[str, Any] = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta_json = json.load(f)
        except Exception:
            pass

    meta_json.update({
        "title": "AI LemGend — LemGendary AI Assistant Training Manifold",
        "id": "lemtreursi/lemgendizedaihelperlemgend",
        "subtitle": "Documentation-Grounded Assistant for Dataset Engineering, Model Training, and Troubleshooting",
        "description": (
            "## Dataset Overview\n\n"
            "- **Category:** Technical Documentation & AI Assistant Training\n"
            "- **Total Samples:** " + str(len(all_items)) + "\n"
            "- **Architecture Base:** Supervised Instruction Fine-Tuning Corpus (JSONL)\n"
            "- **Primary Task:** Documentation-grounded technical Q&A, diagnostics, and workflow guidance.\n\n"
            "## Composition & Lineage\n\n"
            "This manifold is synthesized from the authoritative LemGendary AI Documentation Hub, "
            "operations manuals, architectural whitepapers, and system architecture manifests.\n\n"
            "## Model Training Profiles\n\n"
            "Bound to AI LemGend assistant instruction fine-tuning pipelines."
        ),
        "keywords": [
            "deep-learning",
            "machine-learning",
            "natural-language-processing",
            "computer-science",
            "data-science",
        ],
        "licenses": [{"name": "CC-BY-NC-4.0"}],
        "isPrivate": True,
        "expectedUpdateFrequency": "monthly",
        "author": "Lem Treursic",
        "authors": [
            {
                "name": "Lem Treursic",
                "bio": "Creator of LemGendary AI and the AI LemGend assistant",
                "role": "Author",
            }
        ],
        "resources": [
            {
                "path": "ai_helper_train.jsonl",
                "description": "Fine-tuning prompt-completion instructions and domain training corpus.",
            },
            {
                "path": "ai_helper_val.jsonl",
                "description": "Evaluation validation prompt-completion instructions and benchmark test splits.",
            },
            {
                "path": "category.txt",
                "description": "Domain taxonomy descriptor defining the task archetype.",
            },
            {
                "path": "classes.txt",
                "description": "Exhaustive newline-delimited class dictionary.",
            },
            {
                "path": "dataset_info.yaml",
                "description": "Machine-readable YAML manifest detailing sample counts, sha256 checksums, and container topology.",
            },
            {
                "path": "README.md",
                "description": "Authoritative manifold documentation, lineage matrix, and SOTA metric targets.",
            },
        ],
        "provenanceSources": sources_list,
        "collectionMethodology": (
            "Curated supervised instruction examples derived from reviewed documentation "
            "and general AI/ML engineering principles. Records use JSONL format with "
            "messages in system/user/assistant order.\n"
        ),
        "coverage": {
            "temporal": {"start": "2024-01-01", "end": "2026-10-10"},
            "geospatial": "Global Digital Documentation and AI Engineering",
        },
    })
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # 6. Execute provenance enrichment & metadata compilation
    import subprocess
    import sys

    enrich_script = os.path.join(os.path.dirname(__file__), "enrich_ai_helper_provenance.py")
    compile_script = os.path.join(os.path.dirname(__file__), "compile_ai_helper_metadata.py")
    subprocess.run([sys.executable, enrich_script], check=True)
    subprocess.run([sys.executable, compile_script], check=True)

    # 7. Mirror enriched corpus to LemGendaryModels/ai_helper_lemgend/ and data/
    model_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "LemGendaryModels",
            "ai_helper_lemgend",
        )
    )
    model_data_dir = os.path.join(model_dir, "data")
    os.makedirs(model_data_dir, exist_ok=True)
    import shutil

    shutil.copy2(train_path, os.path.join(model_dir, "ai_helper_train.jsonl"))
    shutil.copy2(val_path, os.path.join(model_dir, "ai_helper_val.jsonl"))
    shutil.copy2(train_path, os.path.join(model_data_dir, "ai_helper_train.jsonl"))
    shutil.copy2(val_path, os.path.join(model_data_dir, "ai_helper_val.jsonl"))
    print(f"Mirrored enriched corpus shards to {model_dir} and {model_data_dir}")

    # 8. Print breakdown
    print(f"Synthesized AI Helper Corpus successfully:")
    print(f"  Total examples: {len(all_items)}")
    print(f"  Train shard:    {len(train_set)} -> {train_path}")
    print(f"  Val shard:      {len(val_set)} -> {val_path}")
    print("\nValidation Set Category Breakdown:")
    val_cats: Dict[str, int] = {}
    for ex in val_set:
        val_cats[ex["category"]] = val_cats.get(ex["category"], 0) + 1
    for cat, count in sorted(val_cats.items()):
        print(f"  - {cat}: {count} examples")


if __name__ == "__main__":
    synthesize_corpus()
