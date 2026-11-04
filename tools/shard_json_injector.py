"""Add the per-image JSON labels to the NIMA manifolds' WebDataset shards."""

import io
import logging
import os
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_MODELS = (
    "LemGendizedNimaAesthetic",
    "LemGendizedNimaAuthenticity",
    "LemGendizedNimaTechnical",
    "LemGendizedRetinaFace",
)
_SPLITS = ("train", "val")
_PREVIEW_LIMIT = 10  # Cap summary previews at 10 items


@dataclass
class _SplitReport:
    """Images left without a label and labels used, tracked per split."""

    unmatched_images: list[str] = field(default_factory=list)
    matched_json_stems: set[str] = field(default_factory=set)


def _copy_with_labels(
    src_tar: tarfile.TarFile,
    dest_tar: tarfile.TarFile,
    json_dir: Path,
    shard_name: str,
    report: _SplitReport,
) -> None:
    """Copy every member of ``src_tar``, adding ``<stem>.json`` after each labelled image."""
    # 2. List and iterate through members
    for member in src_tar.getmembers():
        extracted_file = src_tar.extractfile(member)

        # Safe check: if it's a directory or link, extractfile returns None
        if extracted_file is None:
            # Copy directories and links as they are, without reading bytes
            dest_tar.addfile(member)
            continue
        dest_tar.addfile(member, io.BytesIO(extracted_file.read()))

        # 3. If it's a webp image, check for a matching label
        if not member.name.lower().endswith(".webp"):
            continue
        stem = Path(member.name).stem
        json_file = json_dir / f"{stem}.json"
        if not json_file.exists():
            report.unmatched_images.append(f"{shard_name} -> {member.name}")
            continue

        # 4. Insert corresponding JSON metadata cleanly side-by-side
        json_bytes = json_file.read_bytes()
        json_member = tarfile.TarInfo(name=f"{stem}.json")
        json_member.size = len(json_bytes)
        dest_tar.addfile(json_member, io.BytesIO(json_bytes))
        report.matched_json_stems.add(stem)


def _rewrite_shard(shard_file: Path, json_dir: Path, report: _SplitReport) -> None:
    """Rebuild one ``.data`` shard with its JSON labels, keeping the original name."""
    # 1. Rename .data to .tar
    tar_path = shard_file.with_suffix(".tar")
    shard_file.rename(tar_path)
    print(f"  -> Opened: {tar_path.name}")

    # Setup paths for a temporary rebuild strategy
    temp_output_tar = tar_path.with_name(tar_path.stem + "_temp.tar")

    try:
        with (
            tarfile.open(tar_path, "r") as src_tar,
            tarfile.open(temp_output_tar, "w") as dest_tar,
        ):
            _copy_with_labels(src_tar, dest_tar, json_dir, shard_file.name, report)
    except (OSError, tarfile.TarError) as e:
        logger.error("Error processing shard archive %s: %s", tar_path, e)
        print(f"    Error processing shard archive {tar_path.name}: {e}")
        # Revert to original naming structure if processing breaks mid-way
        if tar_path.exists() and not shard_file.exists():
            tar_path.rename(shard_file)
        return

    # Overwrite original tar with the updated bundle
    os.remove(tar_path)
    shutil.move(str(temp_output_tar), str(tar_path))

    # 5. Rename archive back from .tar to .data
    tar_path.rename(shard_file)
    print(f"  <- Finalized & Restored: {shard_file.name}")


def _print_preview(items: list[str], line_format: str) -> None:
    for item in items[:_PREVIEW_LIMIT]:
        print(line_format.format(item))
    if len(items) > _PREVIEW_LIMIT:
        print(f"  ... and {len(items) - _PREVIEW_LIMIT} more.")


def _print_split_summary(split: str, json_dir: Path, report: _SplitReport) -> None:
    """Report images without labels and labels no image used."""
    # 6. Calculate mismatched/unmatched labels left over in local folder
    all_json_stems = {f.stem for f in json_dir.glob("*.json")}
    unmatched_labels = all_json_stems - report.matched_json_stems

    # Print comprehensive summary logs for this run
    print(f"\n--- Split Summary [{split}] ---")
    print(f"Total Unmatched Images: {len(report.unmatched_images)}")
    _print_preview(report.unmatched_images, "  Missing label for image: {}")

    print(f"Total Unused/Orphan JSON Labels: {len(unmatched_labels)}")
    _print_preview(list(unmatched_labels), "  Orphan Label: {}.json")


def _process_split(model_path: Path, split: str) -> None:
    """Rewrite every shard of one split and print its summary."""
    shards_dir = model_path / "shards" / split
    json_dir = model_path / "json-labels" / split

    if not shards_dir.exists():
        print(f"Skipping split {split}: Path does not exist.")
        return

    # Find all shard files with the current .data extension
    shard_files = list(shards_dir.glob("*.data"))
    if not shard_files:
        print(f"No .data shards found in {split} folder.")
        return

    print(f"\nProcessing [{split}] split - Found {len(shard_files)} shards.")
    # Track mismatched images and unused labels globally per split
    report = _SplitReport()
    for shard_file in shard_files:
        _rewrite_shard(shard_file, json_dir, report)
    _print_split_summary(split, json_dir, report)


def process_shards():
    """Rewrite each NIMA manifold shard with the matching JSON label added to every image."""
    base_dir = Path(__file__).resolve().parent.parent.parent / "LemGendaryDatasets"

    for model in _MODELS:
        model_path = base_dir / model
        print("\n========================================")
        print(f"Starting Dataset Upgrade: {model}")
        print("========================================")

        for split in _SPLITS:
            _process_split(model_path, split)


if __name__ == "__main__":
    process_shards()
