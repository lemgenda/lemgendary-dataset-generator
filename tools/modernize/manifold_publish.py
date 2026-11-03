"""Publish modern manifolds: docs, container conversion, Kaggle re-upload, report."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from core.doc_generator import generate_dataset_docs
from core.kaggle_identity import DEFAULT_KAGGLE_OWNER
from formats.base import parse_also_format
from tools import manifold_sync, migrate_manifold_format
from tools.modernize.common import YAML_UPDATE_ERRORS, merge_into_yaml
from tools.modernize.manifold_scan import _save_registry

logger = logging.getLogger(__name__)


# ─── Docs Regeneration ──────────────────────────────────────────────────────
def _regen_docs(item: dict) -> bool:
    """Regenerate dataset_info.yaml, README.md, category.txt, classes.txt, and index.json."""
    try:
        generate_dataset_docs(
            item["target_path"],
            final_index=None,
            pascal_name=item["base_name"],
        )
        print(f"[OK] Docs regenerated: {item['target_name']}")
        return True
    except (ImportError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as e:
        logger.warning("Docs regeneration failed for %s: %s", item.get("target_name"), e)
        print(f"[WARN] Docs regeneration failed for {item['target_name']}: {e}")
        return False


# ─── Kaggle Re-upload ───────────────────────────────────────────────────────
def _compute_new_kaggle_ref(item: dict) -> str:
    owner = DEFAULT_KAGGLE_OWNER
    old_ref = item.get("old_kaggle_ref") or ""
    if old_ref.startswith("kaggle://"):
        parts = old_ref[len("kaggle://"):].split("/")
        if parts and parts[0]:
            owner = parts[0]
    new_slug = item["target_name"].lower()
    return f"kaggle://{owner}/{new_slug}"


def _kaggle_reupload(item: dict, dry_run: bool = False) -> tuple[bool, str]:
    new_ref = _compute_new_kaggle_ref(item)
    item["new_kaggle_ref"] = new_ref
    clean_ref = new_ref.replace("kaggle://", "")

    if dry_run:
        return True, f"[DRY-RUN] Would upload {item['target_name']} -> {new_ref}"

    try:
        manifold_sync.action_sync(
            manifold_name=item["target_name"], repo_id=clean_ref, no_wait=False
        )
        return True, f"Uploaded to {new_ref}"
    except SystemExit as exc:
        logger.warning("Kaggle re-upload of %s exited with %r", item["target_name"], exc.code)
        return False, f"manifold_sync exited {exc.code}"
    except (ImportError, OSError, ValueError, RuntimeError) as e:
        logger.warning("Kaggle re-upload failed for %s: %s", item["target_name"], e)
        return False, f"sync error: {e}"


# ─── Registry Update ────────────────────────────────────────────────────────
def _update_kaggle_ref(registry: dict, item: dict) -> bool:
    key = item.get("kaggle_key")
    if not key:
        return False
    new_ref = item.get("new_kaggle_ref", "")
    if not new_ref:
        return False
    registry["datasets"][key]["kaggle_ref"] = new_ref
    return True


def _write_name_suffix(registry: dict, value: str) -> None:
    registry.setdefault("_registry_metadata", {})["name_suffix"] = value
    _save_registry(registry)
    print(f"[OK] unified_data.yaml name_suffix set to: '{value}'")


# ─── Reporting ──────────────────────────────────────────────────────────────
def _report(
    modernized: list[dict],
    kaggle_ok: list[dict],
    kaggle_fail: list[tuple[dict, str]],
    suffix_written: bool,
) -> None:
    print()
    print("══════════════════════════════════════════════════════════════════════")
    print(" MODERNIZATION SUMMARY")
    print("══════════════════════════════════════════════════════════════════════")
    print(f" Modernized manifolds created: {len(modernized)}")
    for item in modernized:
        print(f"   - {item['folder_name']} (source) -> {item['target_name']} (new)")
    if kaggle_ok:
        print(f"\n Kaggle re-uploads succeeded: {len(kaggle_ok)}")
        for item in kaggle_ok:
            print(f"   - {item.get('new_kaggle_ref', '?')}")
    if kaggle_fail:
        print(f"\n Kaggle re-uploads FAILED: {len(kaggle_fail)}")
        for item, err in kaggle_fail:
            print(f"   - {item['target_name']}: {err}")
    print(f"\n name_suffix updated to \"\": {'YES' if suffix_written else 'NO'}")
    if not suffix_written:
        print("   (name_suffix write skipped or preserved for remaining legacy sets)")
    print("══════════════════════════════════════════════════════════════════════")



# ─── Canonical Container Conversion ─────────────────────────────────────────
_CONTAINER_ERRORS = (ImportError, OSError, ValueError, KeyError, RuntimeError, yaml.YAMLError)


def _record_format(target_path: Path, target_format: str) -> None:
    """Set ``format`` in the manifold's dataset_info.yaml, when it exists."""
    ds_info_path = target_path / "dataset_info.yaml"
    if not ds_info_path.exists():
        return
    try:
        merge_into_yaml(ds_info_path, {"format": target_format})
    except YAML_UPDATE_ERRORS as exc:
        logger.warning("Failed updating format in %s: %s", ds_info_path, exc)


def _migrate_container(item: dict, target_format: str, purge_loose: bool) -> None:
    """Write ``target_format`` for a modern manifold, then refresh its metadata."""
    target_path = item["target_path"]
    mig_code = migrate_manifold_format.migrate_manifold(
        root=target_path,
        formats=parse_also_format(target_format),
        force_duplicate=True,
        accept_space_loss=True,
        purge_source=purge_loose,
    )
    if mig_code != 0:
        print(f"[WARN] Container migration returned code {mig_code} for {target_path.name}")
        return
    if purge_loose:
        print(f"[ZERO-DUP] Purged intermediate loose image directories from {target_path.name}.")
    _record_format(target_path, target_format)
    # Regenerate the docs so the README structure no longer lists the deleted images
    _regen_docs(item)


def convert_to_container(item: dict, target_format: str, purge_loose: bool) -> None:
    """Convert one modern manifold to its canonical container format (Phase C)."""
    target_path = item["target_path"]
    if target_format == "directory":
        print(
            f"[CONTAINER] {target_path.name} canonical format is DIRECTORY. Retaining"
            " optimized WebP structure."
        )
        return
    print(
        f"[CONTAINER] Modernizing {target_path.name} to canonical format:"
        f" {target_format.upper()}..."
    )
    try:
        _migrate_container(item, target_format, purge_loose)
    except _CONTAINER_ERRORS as exc:
        logger.error("Container conversion failed for %s: %s", target_path, exc)
        print(
            f"[ERROR] Container conversion encountered an issue for {target_path.name}:"
            f" {exc}"
        )
