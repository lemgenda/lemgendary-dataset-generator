"""Diverse boundary and negative refusal synthesizer for AI Studio Copilot.

Generates distinct, model- and dataset-specific ``boundary_and_negative``
records across three refusal archetypes:
1. ``active_training: null`` or offline training sidecar
2. Unsupported container requests (TFRecord, HDF5, LMDB, RecordIO)
3. Missing SOTA targets in live state and retrieved manifest context

Each scenario uses distinct prompt/reply framing and real registry attributes
to survive semantic deduplication (~10% of the compiled corpus).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .response_formatter import ReplyOptions, format_structured_reply, strip_emojis
from .taxonomy import HARDWARE_PROFILES, sidecar_ports
from .token_budget import (
    ASSISTANT_HARD_MAX_TOKENS,
    SYNTHESIS_MAX_TOTAL_TOKENS,
    count_record_tokens,
    truncate_to_token_limit,
)

UNSUPPORTED_CONTAINERS: tuple[str, ...] = ("TFRecord", "HDF5", "LMDB", "RecordIO")


@dataclass(frozen=True)
class _ModelSummary:
    """Distilled registry fields used to ground model boundary prompts uniquely."""

    key: str
    name: str
    arch: str
    loss_fn: str
    checkpoint: str
    meta: dict[str, str]

    @property
    def category(self) -> str:
        """Model category label."""
        return self.meta["category"]

    @property
    def primary_metric(self) -> str:
        """Primary evaluation metric key."""
        return self.meta["primary_metric"]

    @property
    def metrics_list(self) -> str:
        """Comma-separated evaluation metric names."""
        return self.meta["metrics_list"]

    @property
    def accelerator(self) -> str:
        """Default Kaggle accelerator profile."""
        return self.meta["accelerator"]

    @property
    def description(self) -> str:
        """Compact single-line model description."""
        return self.meta["description"]


@dataclass(frozen=True)
class _DatasetSummary:
    """Distilled registry fields used to ground dataset boundary prompts uniquely."""

    key: str
    title: str
    task: str
    folder: str
    canonical_fmt: str
    meta: dict[str, Any]

    @property
    def subtitle(self) -> str:
        """Compact manifold subtitle."""
        return str(self.meta["subtitle"])

    @property
    def sources(self) -> str:
        """Comma-separated provenance source summary."""
        return str(self.meta["sources"])

    @property
    def val_split(self) -> Any:
        """Declared validation split fraction."""
        return self.meta["val_split"]


def _summarize_model(mkey: str, info: dict[str, Any]) -> _ModelSummary:
    vsuite = info.get("validation_suite") or {}
    metrics = vsuite.get("metrics") or ["eval_loss"]
    raw_desc = strip_emojis(str(info.get("description") or "")).strip()
    desc = truncate_to_token_limit(" ".join(raw_desc.split()), 28) or f"{mkey} architecture"
    return _ModelSummary(
        key=mkey,
        name=str(info.get("name", mkey)),
        arch=str(info.get("architecture_type", "Neural Architecture")),
        loss_fn=str(info.get("loss_fn", "cross_entropy")),
        checkpoint=str(info.get("checkpoint", f"{mkey}.pth")),
        meta={
            "category": str(info.get("category", "general")),
            "primary_metric": str(vsuite.get("primary_metric") or metrics[0]),
            "metrics_list": ", ".join(str(m) for m in metrics[:4]),
            "accelerator": str(info.get("kaggle_accelerator", "gpu-t4x2")),
            "description": desc,
        },
    )


def _summarize_dataset(dkey: str, info: dict[str, Any]) -> _DatasetSummary:
    sources = info.get("provenance_sources") or [dkey]
    raw_sub = strip_emojis(str(info.get("subtitle") or "")).strip()
    return _DatasetSummary(
        key=dkey,
        title=str(info.get("title") or info.get("name") or dkey),
        task=str(info.get("task", "vision")),
        folder=str(info.get("modernized_folder", dkey)),
        canonical_fmt=str(info.get("canonical_format", "webdataset")),
        meta={
            "subtitle": truncate_to_token_limit(" ".join(raw_sub.split()), 20) or dkey,
            "sources": ", ".join(str(s) for s in sources[:3]),
            "val_split": info.get("val_split", 0.12),
        },
    )


def _boundary_id(prefix: str, key: str) -> str:
    return f"{prefix}_{hashlib.sha256(key.encode()).hexdigest()[:10]}"


def _format_sys_turn(
    base_prompt: str,
    gui_state: dict[str, Any],
    retrieved_docs: str,
) -> str:
    state_json = json.dumps(gui_state, separators=(",", ":"), ensure_ascii=True)
    return (
        f"{strip_emojis(base_prompt).strip()}\n\n"
        f"[GUI_STATE]{state_json}[/GUI_STATE]\n\n"
        f"[RETRIEVED_DOCS]{strip_emojis(retrieved_docs).strip()}[/RETRIEVED_DOCS]"
    )


def _state_snapshot(
    *,
    active_tab: str,
    hw: dict[str, Any],
    training_online: bool,
    active_training: dict[str, Any] | None,
) -> dict[str, Any]:
    env_p, comp_p, train_p = sidecar_ports()
    return {
        "active_tab": active_tab,
        "hardware": hw,
        "sidecars": {
            f"env_{env_p}": "ONLINE",
            f"compiler_{comp_p}": "ONLINE",
            f"training_{train_p}": "ONLINE" if training_online else "OFFLINE",
        },
        "active_training": active_training,
        "active_compile": None,
        "recent_errors": [],
    }


def _offline_or_idle_specs(
    ms: _ModelSummary, hw: dict[str, Any], variant: int, train_p: int
) -> tuple[bool, str, str, list[str], str]:
    """Return ``(training_online, user, opening, bullets, next_step)`` across 8 angles."""
    gpu_desc = f"{hw['gpu']} ({hw['vram_mb']} MB, backend={hw['backend']})"
    specs: tuple[tuple[bool, str, str, list[str], str], ...] = (
        (
            False,
            f"Port {train_p} shows OFFLINE while inspecting `{ms.name}`. What epoch is running?",
            f"[CONFIRMED] No active run can be observed for `{ms.key}` because sidecar "
            f"port {train_p} is OFFLINE (`active_training: null`).",
            [
                f"Sidecar Link: `training_{train_p}` reports OFFLINE for `{ms.name}` ({ms.arch}).",
                f"Model Profile: `{ms.key}` trains `{ms.category}` weights (`{ms.checkpoint}`, "
                f"loss `{ms.loss_fn}`) covering {ms.description}.",
                "Dashboard Action: click `Start Training Suite` on `tab-dashboard` to boot.",
                f"CLI Probe: execute `lem-env status` then `lemtrain train {ms.key}`.",
            ],
            f"Next step: restore sidecar port {train_p} and click `train-btn-refresh-models`.",
        ),
        (
            True,
            f"Why is `training-telemetry-tabs-tab-local` blank for `{ms.key}` ({ms.checkpoint})?",
            f"[CONFIRMED] Telemetry for `{ms.key}` is blank because `active_training` is "
            f"`null` on port {train_p} despite the sidecar being ONLINE.",
            [
                f"Idle State: `{ms.name}` ({ms.loss_fn}) has no active process on {gpu_desc}.",
                f"Architecture Scope: `{ms.key}` ({ms.arch}) tracks `{ms.metrics_list}` "
                f"and targets {ms.description}.",
                f"Start Run: pick batch size in `train-select-batch-{ms.key}` and click "
                f"`train-btn-start-{ms.key}` (Local Training).",
                f"CLI Alternative: run `lemtrain train {ms.key}` to stream live epochs.",
            ],
            f"Next step: launch `{ms.key}` via Local Training to populate local telemetry.",
        ),
        (
            False,
            f"Can I monitor `Blend Soup` progress for `{ms.key}` while port {train_p} is down?",
            f"[CONFIRMED] Live training and Blend Soup status for `{ms.key}` are unavailable "
            f"because port {train_p} is OFFLINE (`active_training: null`).",
            [
                f"Offline Guard: `training_{train_p}: OFFLINE` blocks `POST /api/models/"
                f"{ms.key}/soup` and live metric polling.",
                f"Model Context: `{ms.name}` evaluates `{ms.metrics_list}` before blending.",
                f"Checkpoint Lineage: `{ms.checkpoint}` ({ms.arch}) covers {ms.description}.",
                "Recovery: start Training Suite on `tab-dashboard` or run "
                f"`lemtrain soup {ms.key}`.",
            ],
            f"Next step: bring port {train_p} ONLINE before opening `train-btn-soup-{ms.key}`.",
        ),
        (
            True,
            f"What Sawtooth resolution rung is `{ms.key}` currently holding on `{hw['gpu']}`?",
            f"[CONFIRMED] No Sawtooth rung is active for `{ms.key}` (`active_training: null`) "
            f"on port {train_p}, so no live rung or fraction exists.",
            [
                f"No Active Rung: `{ms.name}` ({ms.description}) is idle on {gpu_desc}.",
                f"Domain Metadata: `{ms.category}` model uses `{ms.loss_fn}` and "
                f"`{ms.accelerator}` acceleration.",
                f"Monotonic Ratchet: rung tracking starts only after `train-btn-start-{ms.key}`.",
                f"Inspect Checkpoint: run `lemtrain evaluate {ms.key}` to test `{ms.checkpoint}`.",
            ],
            f"Next step: trigger `train-btn-start-{ms.key}` or `lemtrain train {ms.key}`.",
        ),
        (
            False,
            f"`Local Training` for `{ms.key}` refuses to start because port {train_p} is OFFLINE.",
            f"[CONFIRMED] Training dispatch for `{ms.key}` cannot succeed while "
            f"`training_{train_p}` is OFFLINE and `active_training` is `null`.",
            [
                f"Route Blocked: `POST /api/gui/quick-train` requires port {train_p} ONLINE.",
                f"Target Spec: `{ms.name}` (`{ms.key}`) executes {ms.arch} with "
                f"`{ms.primary_metric}` primary metric ({ms.description}).",
                f"GUI Start: click `Start All Services` on `tab-dashboard` to wake `{ms.key}`.",
                f"CLI Fallback: run `lem-env status` and `lemtrain train {ms.key}` locally.",
            ],
            "Next step: wait for the `tab-training` sidebar indicator dot to turn green.",
        ),
        (
            True,
            f"Read live VRAM consumption for `{ms.key}` on `{hw['gpu']}` "
            f"(`vram_mb={hw['vram_mb']}`).",
            f"[CONFIRMED] Live training VRAM telemetry is absent for `{ms.key}` because "
            f"`active_training` is `null` on port {train_p}.",
            [
                f"Idle GPU: `{hw['gpu']}` reports `vram_mb: {hw['vram_mb']}` with no active job.",
                f"Architecture Note: `{ms.name}` uses `{ms.arch}` and `{ms.accelerator}` profile.",
                f"Training Scope: `{ms.key}` optimizes `{ms.loss_fn}` for {ms.description}.",
                f"Start Monitoring: click `train-btn-start-{ms.key}` or "
                f"`train-btn-cloud-{ms.key}`.",
            ],
            f"Next step: start `{ms.key}` and inspect `training-telemetry-tabs-tab-local`.",
        ),
        (
            True,
            f"Should I toggle `train-input-override-frozen-{ms.key}` before checking active loss?",
            f"[CONFIRMED] Neither active loss nor frozen-run drift exists yet for `{ms.key}` "
            f"because `active_training` is `null` on port {train_p}.",
            [
                f"Idle Guard: `{ms.name}` has no running session (`status` unavailable).",
                f"Weights & Loss: `{ms.checkpoint}` trains under `{ms.loss_fn}` covering "
                f"{ms.description}.",
                "Override Rule: leave `Override frozen settings` unchecked unless resuming after "
                "a `train-frozen-refusal` alert.",
                f"Launch Action: click `train-btn-start-{ms.key}` or run "
                f"`lemtrain train {ms.key}`.",
            ],
            f"Next step: launch `{ms.key}` without override unless `FrozenSettingsChanged` fires.",
        ),
        (
            False,
            f"Click `train-btn-stop-{ms.key}` now — how many epochs did `{ms.name}` finish?",
            f"[CONFIRMED] Stop Training cannot report completed epochs for `{ms.key}` because "
            f"sidecar `training_{train_p}` is OFFLINE and `active_training` is `null`.",
            [
                f"No Running Job: `POST /api/jobs/{{job_id}}/cancel` requires an active "
                f"port {train_p} job.",
                f"Offline Check: `{ms.name}` (`{ms.checkpoint}`) has no visible epoch count.",
                f"Model Summary: `{ms.key}` ({ms.arch}) logs `{ms.metrics_list}` during runs.",
                "Recovery: click `Start Training Suite` on `tab-dashboard` or `lem-env status`.",
            ],
            f"Next step: bring port {train_p} ONLINE and click `train-btn-refresh-models`.",
        ),
    )
    return specs[variant % len(specs)]


def _offline_record(
    idx: int, ms: _ModelSummary, variant: int, system_prompt: str
) -> dict[str, Any]:
    train_p = sidecar_ports()[2]
    hw = dict(HARDWARE_PROFILES[idx % len(HARDWARE_PROFILES)])
    online, user_turn, opening, bullets, next_step = _offline_or_idle_specs(
        ms, hw, variant, train_p
    )
    gui_state = _state_snapshot(
        active_tab="tab-dashboard" if variant in (0, 4, 7) else "tab-training",
        hw=hw,
        training_online=online,
        active_training=None,
    )
    retrieved = (
        f"{ms.name} ({ms.key}) uses {ms.arch} ({ms.loss_fn}). Training sidecar port {train_p} "
        f"exposes local launch via train-btn-start-{ms.key} and lemtrain train {ms.key}."
    )
    asst_turn = format_structured_reply(
        opening,
        bullets,
        next_step,
        ReplyOptions(
            category="boundary_and_negative",
            epistemic_tag="[CONFIRMED]",
            seed_key=f"boff:{ms.key}:{variant}:{idx}",
        ),
    )
    return {
        "id": _boundary_id("boundary_state", f"{ms.key}:{variant}:{idx}"),
        "category": "boundary_and_negative",
        "source_doc": "GUI_CONTROL_REGISTRY.md",
        "section_title": f"Boundary Offline/Idle ({ms.key}, v{variant})",
        "heading_path": f"boundary_and_negative > sidecar_or_idle > {ms.key}",
        "authority_tier": "tier_1_manifest",
        "messages": [
            {"role": "system", "content": _format_sys_turn(system_prompt, gui_state, retrieved)},
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": asst_turn},
        ],
    }


def _unsupported_format_specs(
    fmt: str, ds: _DatasetSummary, variant: int
) -> tuple[str, str, list[str], str]:
    """Return ``(user, opening, bullets, next_step)`` across 6 format refusal angles."""
    is_tabular = "forex" in ds.key.lower() or (fmt == "HDF5" and variant % 2 == 1)
    native = "Apache Parquet (.parquet)" if is_tabular else "WebDataset (.data)"
    if variant == 0:
        return (
            f"Compile `{ds.title}` (`{ds.key}`) into `{fmt}` via `comp-btn-compile`.",
            f"[CONFIRMED] `{fmt}` is unsupported by `comp-btn-compile` (Compile Manifold) "
            f"for `{ds.title}` (`{ds.key}`), which uses `{ds.canonical_fmt}`.",
            [
                f"Manifold Scope: `{ds.title}` ({ds.subtitle}) writes `{ds.folder}` "
                f"for `{ds.task}`.",
                f"Upstream Provenance: `{ds.key}` draws from {ds.sources} with "
                f"val_split={ds.val_split}.",
                f"Native Targets: convert `{fmt}` inputs to `{native}` or `LitData`.",
                f"GUI Steps: choose `{ds.key}` in `comp-select-manifold`, set "
                "`comp-select-shard-ext`, and click `comp-btn-compile`.",
            ],
            f"Next step: convert `{fmt}` files externally to raw media or `{native}` first.",
        )
    if variant == 1:
        return (
            f"Under `comp-mode-custom`, can `custom-format-select` output `{fmt}` for `{ds.key}`?",
            f"[CONFIRMED] Custom Multi-Source compilation cannot emit `{fmt}` because "
            "`custom-format-select` accepts only native ecosystem containers.",
            [
                f"Allowed Options: `custom-format-select` offers WebDataset (.data), "
                f"Apache Parquet (.parquet), MDS, and LitData instead of `{fmt}`.",
                f"Source Context: `{ds.key}` ({ds.subtitle}) ingests `{ds.sources}`.",
                f"Target Layout: `{ds.folder}` stores `{ds.task}` splits at "
                f"val_split={ds.val_split}.",
                f"Recommended Conversion: stage `{fmt}` records into `{native}` before build.",
            ],
            f"Next step: select `{native}` or `LitData` in `custom-format-select`.",
        )
    if variant == 2:
        return (
            f"Does `python cli.py modernize` transcode `{fmt}` archives into `{ds.folder}`?",
            f"[CONFIRMED] `python cli.py modernize` does not ingest `{fmt}` directly; convert "
            f"`{fmt}` to `{native}` or `LitData` prior to `{ds.folder}` packing.",
            [
                f"Non-Native Input: `{fmt}` lacks a container writer for `{ds.title}` ({ds.task}).",
                f"Catalog Provenance: `{ds.key}` ({ds.subtitle}) pulls from {ds.sources}.",
                f"Conversion Path: extract `{fmt}` payloads to standard images/JSON or Parquet.",
                f"CLI Command: run `python cli.py compile --model {ds.key}` after staging.",
            ],
            f"Next step: unpack `{fmt}` into `{native}` and run `python cli.py verify`.",
        )
    if variant == 3:
        return (
            f"If I upload `{fmt}` shards for `{ds.key}` via `comp-btn-push-{ds.key}`, is it valid?",
            f"[CONFIRMED] Uploading `{fmt}` shards into `{ds.folder}` violates the `{ds.key}` "
            f"canonical `{ds.canonical_fmt}` contract.",
            [
                f"Sync Guard: `comp-btn-push-{ds.key}` and `python cli.py sync push` expect "
                f"`{ds.canonical_fmt}` or `{native}` with `shard_offsets.json`.",
                f"Dataset Identity: `{ds.title}` splits at `val_split={ds.val_split}` "
                f"from {ds.sources}.",
                f"Domain Metadata: `{ds.subtitle}` serves `{ds.task}` workloads cleanly.",
                f"Remediation: re-pack `{fmt}` into `{native}` via `comp-btn-compile`.",
            ],
            f"Next step: recompile `{ds.key}` into `{native}` before clicking Upload.",
        )
    if variant == 4:
        return (
            f"Can `reduce-manifold-select` downsample a `{fmt}` copy of `{ds.key}`?",
            f"[CONFIRMED] Scale reduction via `reduce-manifold-select` rejects `{fmt}` inputs "
            f"and supports only native `{native}`, `WebDataset (.data)`, or `LitData`.",
            [
                f"Supported Formats: scale reduction reads `{ds.canonical_fmt}` shards "
                f"in `{ds.folder}`.",
                f"Manifold Detail: `{ds.title}` ({ds.subtitle}) covers {ds.sources}.",
                f"Target Conversion: export `{fmt}` samples into `{native}` or `Apache Parquet`.",
                "GUI Action: open `compiler-subtabs-tab-reduce` after native compilation.",
            ],
            f"Next step: compile `{ds.key}` as `{native}` before sampling at 10-75%.",
        )
    return (
        f"Will `compiler-btn-validate-schema` accept `{fmt}` as `canonical_format` for `{ds.key}`?",
        f"[CONFIRMED] `compiler-btn-validate-schema` rejects `{fmt}` because `unified_data.yaml` "
        f"permits only WebDataset, Parquet, LitData, MDS, Directory, or JSONL.",
        [
            f"Schema Failure: `{ds.title}` (`{ds.task}`) requires `{ds.canonical_fmt}`, "
            f"not `{fmt}`.",
            f"Source Spec: `{ds.folder}` ({ds.subtitle}) targets {ds.sources}.",
            f"Migration Advice: convert `{fmt}` buffers to `{native}` or `LitData`.",
            "CLI Check: verify manifest compliance with `python cli.py verify`.",
        ],
        f"Next step: keep `{ds.canonical_fmt}` in `unified_data.yaml` and convert "
        f"`{fmt}` upstream.",
    )


def _unsupported_format_record(
    idx: int, ds: _DatasetSummary, variant: int, system_prompt: str
) -> dict[str, Any]:
    comp_p = sidecar_ports()[1]
    fmt = UNSUPPORTED_CONTAINERS[idx % len(UNSUPPORTED_CONTAINERS)]
    gui_state = _state_snapshot(
        active_tab="tab-datasets",
        hw=dict(HARDWARE_PROFILES[idx % len(HARDWARE_PROFILES)]),
        training_online=True,
        active_training=None,
    )
    retrieved = (
        f"{ds.title} ({ds.key}) stores {ds.task} data in {ds.folder} using canonical format "
        f"{ds.canonical_fmt} on Compiler port {comp_p}. Non-native {fmt} is not supported."
    )
    user_turn, opening, bullets, next_step = _unsupported_format_specs(fmt, ds, variant)
    asst_turn = format_structured_reply(
        opening,
        bullets,
        next_step,
        ReplyOptions(
            category="boundary_and_negative",
            epistemic_tag="[CONFIRMED]",
            seed_key=f"bfmt:{fmt}:{ds.key}:{variant}:{idx}",
        ),
    )
    return {
        "id": _boundary_id("boundary_format", f"{fmt}:{ds.key}:{variant}:{idx}"),
        "category": "boundary_and_negative",
        "source_doc": "unified_data.yaml",
        "section_title": f"Unsupported Format ({fmt}, {ds.key}, v{variant})",
        "heading_path": f"boundary_and_negative > unsupported_container > {fmt} > {ds.key}",
        "authority_tier": "tier_1_manifest",
        "messages": [
            {"role": "system", "content": _format_sys_turn(system_prompt, gui_state, retrieved)},
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": asst_turn},
        ],
    }


def _missing_sota_specs(
    ms: _ModelSummary, hw: dict[str, Any], variant: int, epoch: int
) -> tuple[Any, str, str, list[str], str]:
    """Return ``(targets_val, user, opening, bullets, next_step)`` across 6 angles."""
    if variant == 0:
        return (
            None,
            f"`active_training.targets` is `null` for `{ms.key}` at epoch {epoch}. "
            f"What exact `{ms.primary_metric}` SOTA threshold must it beat?",
            f"[CONFIRMED] Exact `{ms.primary_metric}` SOTA targets for `{ms.key}` cannot be "
            "stated because `active_training.targets` is `null` and absent from context.",
            [
                f"Anti-Guess Guard: never fabricate `{ms.primary_metric}` numbers for `{ms.name}`.",
                f"Architecture Detail: `{ms.key}` ({ms.arch}) trains `{ms.checkpoint}` "
                f"on `{ms.loss_fn}` ({ms.description}).",
                f"Card Popover: open `Target Metrics (SOTA)` on the `{ms.key}` card.",
                "Manifest Lookup: open `sidebar-btn-config` (Config & Secrets), click "
                "`config-tabs-tab-manifests`, and select `unified_models_v2.yaml`.",
            ],
            f"Next step: read `sota_targets` in `unified_models_v2.yaml` for `{ms.key}`.",
        )
    if variant == 1:
        return (
            {},
            f"Has `{ms.name}` (`{ms.key}`) at epoch {epoch} cleared SOTA when `targets` is `{{}}`?",
            f"[CONFIRMED] SOTA convergence cannot be verified for `{ms.key}` because "
            "`active_training.targets` is empty (`{}`) and omitted from `[RETRIEVED_DOCS]`.",
            [
                f"Missing Reference: `{ms.key}` tracks `{ms.metrics_list}`, but targets are blank.",
                f"Model Identity: `{ms.name}` ({ms.category}, {ms.arch}) solves {ms.description}.",
                f"GUI Check: inspect `Target Metrics (SOTA)` or run `train-btn-eval-{ms.key}`.",
                "Registry SSOT: verify `unified_models_v2.yaml` via `manifest-select`.",
            ],
            f"Next step: run `lemtrain evaluate {ms.key}` after reading `Target Metrics (SOTA)`.",
        )
    if variant == 2:
        return (
            None,
            f"Should I click `train-btn-export-{ms.key}` now while `targets` is `null` in state?",
            f"[CONFIRMED] Do not assume `{ms.key}` (`{ms.checkpoint}`) is SOTA-ready because "
            "`active_training.targets` is `null` in `[GUI_STATE]`.",
            [
                f"Export Gate: `{ms.name}` ({ms.arch}) needs verified "
                f"`{ms.primary_metric}` targets.",
                f"Loss & Scope: `{ms.key}` uses `{ms.loss_fn}` on {ms.accelerator} "
                f"covering {ms.description}.",
                "Locate Targets: check `Target Metrics (SOTA)` or `unified_models_v2.yaml` "
                "inside `sidebar-btn-config` (Config & Secrets).",
                f"CLI Evaluation: run `lemtrain evaluate {ms.key}` before "
                f"`train-btn-export-{ms.key}`.",
            ],
            "Next step: verify `sota_targets` in `unified_models_v2.yaml` before exporting.",
        )
    if variant == 3:
        return (
            {},
            f"If I blend `{ms.key}` with `train-btn-soup-{ms.key}`, will it pass SOTA benchmarks?",
            f"[CONFIRMED] I decline to predict SOTA pass status for `{ms.key}` because no "
            "`sota_targets` values appear in `[GUI_STATE]` or `[RETRIEVED_DOCS]`.",
            [
                f"Missing Numeric Bounds: `{ms.primary_metric}` target for `{ms.name}` is missing.",
                f"Donor Context: `{ms.key}` ({ms.checkpoint}) logs `{ms.metrics_list}` "
                f"({ms.description}).",
                "Registry Lookup: open `Target Metrics (SOTA)` popover or `manifest-select` "
                "for `unified_models_v2.yaml` under `Config & Secrets`.",
                f"Post-Soup Eval: run `lemtrain soup {ms.key}` then `lemtrain evaluate {ms.key}`.",
            ],
            f"Next step: inspect `Target Metrics (SOTA)` on `{ms.key}` before blending.",
        )
    if variant == 4:
        return (
            None,
            f"How far is `{ms.key}` (`{ms.category}`) from VALIDATED status with `targets: null`?",
            f"[CONFIRMED] Distance to `VALIDATED` status for `{ms.key}` cannot be computed "
            "because `active_training.targets` is `null` in `[GUI_STATE]`.",
            [
                f"Missing Denominator: `{ms.name}` (`{ms.loss_fn}`) has no target value loaded.",
                f"Model Spec: `{ms.key}` runs `{ms.arch}` on `{ms.accelerator}` "
                f"({ms.description}).",
                "Authoritative Source: open `Target Metrics (SOTA)` popover on the card or "
                "`unified_models_v2.yaml` in `Config & Secrets` (`sidebar-btn-config`).",
                f"Telemetry Sync: click `train-btn-refresh-models` and check `{ms.metrics_list}`.",
            ],
            f"Next step: read `sota_targets` from `unified_models_v2.yaml` for `{ms.key}`.",
        )
    return (
        {},
        f"Compare `{ms.key}` on `{hw['gpu']}` against target SOTA `{ms.primary_metric}`.",
        f"[CONFIRMED] I cannot compare `{ms.key}` on `{hw['gpu']}` against SOTA because "
        "`active_training.targets` is empty and no target numbers were retrieved.",
        [
            f"Hardware Snapshot: `{hw['gpu']}` (`vram_mb={hw['vram_mb']}`) runs `{ms.arch}`.",
            f"Domain Scope: `{ms.name}` (`{ms.checkpoint}`) optimizes `{ms.loss_fn}` "
            f"for {ms.description}.",
            "Target Lookup: check `Target Metrics (SOTA)` popover or `unified_models_v2.yaml` "
            "under `config-tabs-tab-manifests` in `Config & Secrets`.",
            f"CLI Benchmark: run `lemtrain evaluate {ms.key}` to log `{ms.metrics_list}`.",
        ],
        f"Next step: verify `{ms.key}` targets in `unified_models_v2.yaml` via `manifest-select`.",
    )


def _missing_sota_record(
    idx: int, ms: _ModelSummary, variant: int, system_prompt: str
) -> dict[str, Any]:
    hw = dict(HARDWARE_PROFILES[idx % len(HARDWARE_PROFILES)])
    epoch = 7 + (idx % 24)
    targets_val, user_turn, opening, bullets, next_step = _missing_sota_specs(
        ms, hw, variant, epoch
    )
    gui_state = _state_snapshot(
        active_tab="tab-training",
        hw=hw,
        training_online=True,
        active_training={
            "model_key": ms.key,
            "status": "TRAINING",
            "epoch": epoch,
            "budget": 80,
            "rung": 256,
            "fraction": 0.4,
            "governor": "NOMINAL_CRUISE",
            "metrics": {ms.primary_metric: 0.512},
            "targets": targets_val,
        },
    )
    retrieved = (
        f"{ms.name} ({ms.key}) monitors {ms.metrics_list}. Authoritative targets live in "
        "unified_models_v2.yaml and the Target Metrics (SOTA) popover on the model card."
    )
    asst_turn = format_structured_reply(
        opening,
        bullets,
        next_step,
        ReplyOptions(
            category="boundary_and_negative",
            epistemic_tag="[CONFIRMED]",
            seed_key=f"bsota:{ms.key}:{variant}:{idx}",
        ),
    )
    return {
        "id": _boundary_id("boundary_sota", f"{ms.key}:{variant}:{idx}"),
        "category": "boundary_and_negative",
        "source_doc": "unified_models_v2.yaml",
        "section_title": f"Missing SOTA Target ({ms.key}, v{variant})",
        "heading_path": f"boundary_and_negative > missing_sota_target > {ms.key}",
        "authority_tier": "tier_1_manifest",
        "messages": [
            {"role": "system", "content": _format_sys_turn(system_prompt, gui_state, retrieved)},
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": asst_turn},
        ],
    }


def _build_boundary_candidate(
    idx: int,
    m_summaries: list[_ModelSummary],
    d_summaries: list[_DatasetSummary],
    system_prompt: str,
) -> dict[str, Any]:
    """Dispatch one deterministic boundary record across the 3 safety categories."""
    step = idx // 3
    bucket = idx % 3
    if bucket == 0:
        item = m_summaries[step % len(m_summaries)]
        return _offline_record(idx, item, (step // len(m_summaries)) % 8, system_prompt)
    if bucket == 1:
        item = d_summaries[step % len(d_summaries)]
        return _unsupported_format_record(
            idx, item, (step // len(d_summaries)) % 6, system_prompt
        )
    item = m_summaries[step % len(m_summaries)]
    return _missing_sota_record(idx, item, (step // len(m_summaries)) % 6, system_prompt)


def synthesize_boundary_and_negative_records(
    models: dict[str, dict[str, Any]],
    datasets: dict[str, dict[str, Any]],
    system_prompt: str,
    target_count: int,
) -> list[dict[str, Any]]:
    """Synthesize ``target_count`` diverse boundary/negative refusal records."""
    if target_count <= 0:
        return []
    m_summaries = [
        _summarize_model(k, models.get(k) or {})
        for k in (sorted(models.keys()) or ["nafnet_debluring", "codeformer", "ai_helper_lemgend"])
    ]
    d_summaries = [
        _summarize_dataset(k, datasets.get(k) or {})
        for k in (sorted(datasets.keys()) or ["nima_aesthetic", "forex_universe"])
    ]

    records: list[dict[str, Any]] = []
    for idx in range(target_count):
        rec = _build_boundary_candidate(idx, m_summaries, d_summaries, system_prompt)
        total_toks, asst_toks = count_record_tokens(rec)
        if asst_toks <= ASSISTANT_HARD_MAX_TOKENS and total_toks <= SYNTHESIS_MAX_TOTAL_TOKENS:
            records.append(rec)
    return records
