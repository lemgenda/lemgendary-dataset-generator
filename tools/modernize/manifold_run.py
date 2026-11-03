"""Run phases of ``tools.modernize_manifold``: select, create, convert, publish."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from core.config_schema import ImageFormatPolicy
from tools.migrate_manifold_image_format import image_policy_from_args
from tools.modernize.manifold_build import _create_modernized_batch
from tools.modernize.manifold_publish import (
    _compute_new_kaggle_ref,
    _kaggle_reupload,
    _regen_docs,
    _report,
    _update_kaggle_ref,
    _write_name_suffix,
    convert_to_container,
)
from tools.modernize.manifold_scan import (
    REGISTRY_YAML,
    _enumerate_eligible,
    _load_registry,
    _save_registry,
)
from tools.modernize.manifold_select import _confirm, select_manifolds


def _target_format(item: dict, override: str | None) -> str:
    """The container format to write for ``item`` (``--format`` override or canonical)."""
    return (override or item.get("canonical_format", "directory")).strip().lower()


def _print_dry_run(
    selected: list[dict], args: argparse.Namespace, policy: ImageFormatPolicy
) -> None:
    print()
    print("[DRY-RUN] Would perform the following:")
    override = args.format or args.also_format
    for item in selected:
        new_ref = _compute_new_kaggle_ref(item)
        canon_fmt = _target_format(item, override)
        print(
            f"  {item['folder_name']}  ->  {item['target_name']} (new folder)  [Canonical:"
            f" {canon_fmt.upper()}]  (kaggle: {new_ref})"
        )
        print(f"    + create new modernized manifold directory: {item['target_name']}")
        print("    + copy non-image structures, labels, and metadata files")
        if not args.skip_transcode and policy.format != "keep":
            print(f"    + transcode images -> {args.image_format} (q={args.image_quality})")
        if canon_fmt != "directory" and not args.skip_container:
            print(f"    + convert to modern container format -> {canon_fmt}")
            if not args.keep_intermediate:
                print(
                    "    + [ZERO-DUP] purge intermediate loose image directories (images/,"
                    " targets/, masks/)"
                )
        print("    + regenerate modern dataset documentation and index manifests")
        if not args.skip_kaggle:
            print(f"    + upload modernized manifold to kaggle: {new_ref}")
    print()
    print("[DRY-RUN] unified_data.yaml would be updated with new kaggle_ref values.")
    print("[DRY-RUN] name_suffix would be set to '' after all succeed.")


def _phase_heading(title: str) -> None:
    print()
    print(title)


def _convert_containers(modernized: list[dict], args: argparse.Namespace) -> None:
    """Phase C: canonical container conversion and zero duplication."""
    _phase_heading("── Phase C: Canonical Container Conversion & Zero Duplication ──")
    override = args.format or args.also_format
    for item in modernized:
        convert_to_container(
            item, _target_format(item, override), purge_loose=not args.keep_intermediate
        )


@dataclass
class _KaggleOutcome:
    """Results of the Kaggle re-upload phase."""

    ok: list[dict] = field(default_factory=list)
    failed: list[tuple[dict, str]] = field(default_factory=list)


def _reupload_all(
    modernized: list[dict], registry: dict, skip_kaggle: bool
) -> _KaggleOutcome:
    """Phase D: re-upload every modern manifold, recording new refs in ``registry``."""
    outcome = _KaggleOutcome()
    if skip_kaggle:
        _phase_heading("── Phase D: Kaggle re-upload SKIPPED (--skip-kaggle) ────────────────")
        return outcome
    _phase_heading("── Phase D: Kaggle re-upload ────────────────────────────────────────")
    for item in modernized:
        ok, msg = _kaggle_reupload(item, dry_run=False)
        if ok:
            print(f"[OK] {item['target_name']}: {msg}")
            outcome.ok.append(item)
            _update_kaggle_ref(registry, item)
        else:
            print(f"[FAIL] {item['target_name']}: {msg}")
            outcome.failed.append((item, msg))
    return outcome


def _modernize(
    selected: list[dict],
    eligible: list[dict],
    registry: dict,
    args: argparse.Namespace,
    policy: ImageFormatPolicy,
) -> int:
    """Phases A-F for the confirmed selection; returns the exit code."""
    _phase_heading("── Phase A: Create Modernized Manifold Folders ──────────────────────")
    create_ok, modernized = _create_modernized_batch(
        selected,
        policy=policy,
        skip_transcode=args.skip_transcode,
    )
    if not create_ok:
        print()
        print("[HALT] Modernization batch aborted. name_suffix NOT written.")
        _report(modernized, [], [], suffix_written=False)
        return 1

    _phase_heading("── Phase B: Metadata regeneration ───────────────────────────────────")
    for item in modernized:
        _regen_docs(item)

    if not args.skip_container:
        _convert_containers(modernized, args)

    kaggle = _reupload_all(modernized, registry, args.skip_kaggle)

    # Phase E: persist registry updates
    if kaggle.ok or args.skip_kaggle:
        _save_registry(registry)
        print(f"[OK] unified_data.yaml updated ({len(kaggle.ok)} kaggle_ref entries).")

    # Phase F: name_suffix write (only if all eligible succeeded)
    all_success = (not kaggle.failed) and len(modernized) == len(selected)
    suffix_written = False
    if all_success and len(modernized) == len(eligible):
        _write_name_suffix(registry, "")
        suffix_written = True

    _report(modernized, kaggle.ok, kaggle.failed, suffix_written=suffix_written)
    return 0 if all_success else 1


def run(args: argparse.Namespace) -> int:
    """Modernize the selected manifolds; returns the exit code."""
    if not REGISTRY_YAML.exists():
        print(f"[ERROR] Registry not found: {REGISTRY_YAML}")
        return 1

    registry = _load_registry()
    eligible = _enumerate_eligible(registry)
    if not eligible:
        print("[INFO] No eligible manifolds found (no unmodernized folders with actual data).")
        return 0

    selected, code = select_manifolds(args, eligible)
    if not selected:
        return code

    policy = image_policy_from_args(args)
    if args.dry_run:
        _print_dry_run(selected, args, policy)
        return 0

    if not args.yes and not _confirm(selected, override_format=args.format or args.also_format):
        print("[ABORTED] User declined. No changes made.")
        return 0

    return _modernize(selected, eligible, registry, args, policy)
