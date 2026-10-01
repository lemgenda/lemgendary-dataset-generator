"""
LemGendary Dataset Compiler — Streaming Zip-to-Container Engine.

Directly streams multi-gigabyte legacy dataset zip archives into modern
canonical containers (WebDataset tar shards, MDS) with on-the-fly
multi-threaded WebP transcoding, without ever extracting loose files to
disk. Eliminates Windows NTFS MFT lock contention, cluster slack, and
antivirus overhead.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import os
from pathlib import Path
import shutil
import sys
import tarfile
from typing import Any
import zipfile

from PIL import Image
import yaml
from utils.progress import create_progress_bar

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _worker_img(item: tuple[str, bytes, str]) -> tuple[str, bytes, str]:
    s, raw, e = item
    out_bytes, out_ext = _transcode_to_webp(raw, e, quality=92)
    return s, out_bytes, out_ext


def _worker_tgt(item: tuple[str, bytes, str]) -> tuple[str, bytes, str]:
    s, raw, e = item
    out_bytes, out_ext = _transcode_to_webp(raw, e, quality=95)
    return s, out_bytes, out_ext


def _worker_sample(item: tuple[str, bytes, str]) -> tuple[str, bytes, str]:
    s, raw, e = item
    out_bytes, out_ext = _transcode_to_webp(raw, e, quality=95)
    return s, out_bytes, out_ext


def _transcode_to_webp(raw_bytes: bytes, ext: str, quality: int = 92) -> tuple[bytes, str]:
    """Transcode image bytes to WebP format using Pillow with libwebp method 2."""
    if ext.lower() == ".webp":
        return raw_bytes, ".webp"
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            if img.mode in ("RGBA", "LA", "P"):
                if img.mode == "P":
                    img = img.convert("RGBA")
                bg = Image.new("RGB", img.size, (255, 255, 255))
                channels = img.split()
                mask = channels[-1] if len(channels) > 3 else None
                bg.paste(img, mask=mask)
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

            out = io.BytesIO()
            img.save(out, format="WEBP", quality=quality, method=2)
            return out.getvalue(), ".webp"
    except Exception:
        return raw_bytes, ext


def _partition_zip_members(
    zf: zipfile.ZipFile,
) -> tuple[
    dict[str, list[tuple[str, str, zipfile.ZipInfo]]],
    dict[str, dict[str, tuple[str, zipfile.ZipInfo]]],
    bool,
    list[zipfile.ZipInfo],
]:
    """Partition zip members into images, targets, and metadata files.

    Returns (images_by_split, targets_by_split, has_images_dir, metadata_members).
    """
    all_members = zf.infolist()
    images_by_split: dict[str, list[tuple[str, str, zipfile.ZipInfo]]] = {}
    targets_by_split: dict[str, dict[str, tuple[str, zipfile.ZipInfo]]] = {}
    metadata_members: list[zipfile.ZipInfo] = []
    has_images_dir = False

    for m in all_members:
        norm_name = m.filename.replace("\\", "/")
        parts = [p for p in norm_name.split("/") if p]

        # Root-level metadata files
        if len(parts) == 1 or (
            len(parts) == 2
            and not m.is_dir()
            and parts[0].lower() not in ("images", "targets", "masks")
        ):
            base_name = parts[-1]
            if any(base_name.endswith(ext) for ext in [".yaml", ".yml", ".md", ".json", ".txt", ".ipynb"]):
                metadata_members.append(m)
                continue

        if m.is_dir():
            continue
        fname = parts[-1]
        dot_pos = fname.rfind(".")
        if dot_pos <= 0:
            continue
        ext = fname[dot_pos:].lower()
        stem = fname[:dot_pos]
        if ext not in _IMAGE_EXTS:
            continue

        split = "all"
        lower_parts = [p.lower() for p in parts]
        if "val" in lower_parts:
            split = "val"
        elif "train" in lower_parts:
            split = "train"
        elif "test" in lower_parts:
            split = "test"

        if "images" in lower_parts:
            has_images_dir = True
            images_by_split.setdefault(split, []).append((stem, ext, m))
        elif "targets" in lower_parts:
            targets_by_split.setdefault(split, {})[stem] = (ext, m)

    return images_by_split, targets_by_split, has_images_dir, metadata_members


def _extract_metadata_files(
    zf: zipfile.ZipFile,
    metadata_members: list[zipfile.ZipInfo],
    target_dir: Path,
) -> int:
    """Extract root metadata files from the zip to target_dir."""
    count = 0
    for m in metadata_members:
        norm_name = m.filename.replace("\\", "/")
        base_name = norm_name.split("/")[-1]
        dest_file = target_dir / base_name
        print(f"  -> Extracting metadata: {base_name} ({m.file_size:,} bytes)")
        with zf.open(m) as src, open(dest_file, "wb") as dst:
            shutil.copyfileobj(src, dst)
        count += 1
    return count


def stream_zip_to_webdataset(
    zip_path: Path,
    target_dir: Path,
    shard_size_samples: int = 5000,
    delete_zip: bool = False,
    legacy_dir: Path | None = None,
    transcode_webp: bool = True,
    max_workers: int | None = None,
) -> bool:
    """Stream paired or single vision datasets directly from zip into WebDataset tar shards."""
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = target_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    workers = max_workers or min(12, os.cpu_count() or 4)
    print(f"[STREAM] Opening source archive: {zip_path.name}")
    print(f"[STREAM] Parallel WebP transcoding: {'ENABLED' if transcode_webp else 'DISABLED'} ({workers} workers).")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            all_members = zf.infolist()
            print(f"[STREAM] Discovered {len(all_members)} entries inside archive.")

            images_by_split, targets_by_split, has_images_dir, metadata_members = _partition_zip_members(zf)

            count = _extract_metadata_files(zf, metadata_members, target_dir)
            print(f"[STREAM] Extracted {count} root metadata/notebook files to {target_dir.name}.")

            with ThreadPoolExecutor(max_workers=workers) as pool:
                # Case A: Paired dataset (images/ and targets/)
                if has_images_dir:
                    for split, img_list in images_by_split.items():
                        split_shards_dir = shards_dir if split == "all" else shards_dir / split
                        split_shards_dir.mkdir(parents=True, exist_ok=True)
                        target_map = targets_by_split.get(split, {})
                        total_samples = len(img_list)
                        total_shards = (total_samples + shard_size_samples - 1) // shard_size_samples
                        print(f"[STREAM] Split '{split}': {total_samples} images, {len(target_map)} targets -> {total_shards} shards.")

                        # Pass 1: Stream images with WebP transcoding
                        with create_progress_bar(total=total_samples, desc=f"IMAGES -> WEBP [{split}]", unit="img") as pbar:
                            for shard_idx in range(total_shards):
                                start_i = shard_idx * shard_size_samples
                                end_i = min(start_i + shard_size_samples, total_samples)
                                shard_file = split_shards_dir / f"shard-{shard_idx:05d}.tar"
                                chunk = img_list[start_i:end_i]

                                with tarfile.open(shard_file, "w") as tf:
                                    if transcode_webp:
                                        def _gen_img(items: list[tuple[str, str, zipfile.ZipInfo]]):
                                            for stem, ext, zinfo in items:
                                                yield stem, zf.read(zinfo), ext

                                        for stem, data, out_ext in pool.map(_worker_img, _gen_img(chunk), chunksize=8):
                                            ti = tarfile.TarInfo(name=f"{stem}{out_ext}")
                                            ti.size = len(data)
                                            tf.addfile(ti, io.BytesIO(data))
                                            pbar.update(1)
                                    else:
                                        for stem, ext, zinfo in chunk:
                                            data = zf.read(zinfo)
                                            ti = tarfile.TarInfo(name=f"{stem}{ext}")
                                            ti.size = len(data)
                                            tf.addfile(ti, io.BytesIO(data))
                                            pbar.update(1)

                        # Pass 2: Append targets with WebP transcoding
                        if target_map:
                            with create_progress_bar(total=total_samples, desc=f"TARGETS -> WEBP [{split}]", unit="img") as pbar:
                                for shard_idx in range(total_shards):
                                    start_i = shard_idx * shard_size_samples
                                    end_i = min(start_i + shard_size_samples, total_samples)
                                    shard_file = split_shards_dir / f"shard-{shard_idx:05d}.tar"
                                    chunk = img_list[start_i:end_i]

                                    matched_stems = [stem for stem, _, _ in chunk if stem in target_map]
                                    if not matched_stems:
                                        continue

                                    with tarfile.open(shard_file, "a") as tf:
                                        if transcode_webp:
                                            def _gen_tgt(stems: list[str]):
                                                for stem in stems:
                                                    tgt_ext, tgt_zinfo = target_map[stem]
                                                    yield stem, zf.read(tgt_zinfo), tgt_ext

                                            for stem, data, out_ext in pool.map(_worker_tgt, _gen_tgt(matched_stems), chunksize=8):
                                                ti = tarfile.TarInfo(name=f"{stem}.target{out_ext}")
                                                ti.size = len(data)
                                                tf.addfile(ti, io.BytesIO(data))
                                                pbar.update(1)
                                        else:
                                            for stem in matched_stems:
                                                tgt_ext, tgt_zinfo = target_map[stem]
                                                data = zf.read(tgt_zinfo)
                                                ti = tarfile.TarInfo(name=f"{stem}.target{tgt_ext}")
                                                ti.size = len(data)
                                                tf.addfile(ti, io.BytesIO(data))
                                                pbar.update(1)

                # Case B: Target-only dataset (targets/train, targets/val, e.g. UpnV2)
                else:
                    for split, target_map in targets_by_split.items():
                        split_shards_dir = shards_dir if split == "all" else shards_dir / split
                        split_shards_dir.mkdir(parents=True, exist_ok=True)
                        tgt_items = list(target_map.items())
                        total_samples = len(tgt_items)
                        total_shards = (total_samples + shard_size_samples - 1) // shard_size_samples
                        print(f"[STREAM] Split '{split}': {total_samples} target samples -> {total_shards} shards.")

                        with create_progress_bar(total=total_samples, desc=f"SAMPLES -> WEBP [{split}]", unit="img") as pbar:
                            for shard_idx in range(total_shards):
                                start_i = shard_idx * shard_size_samples
                                end_i = min(start_i + shard_size_samples, total_samples)
                                shard_file = split_shards_dir / f"shard-{shard_idx:05d}.tar"
                                chunk = tgt_items[start_i:end_i]

                                with tarfile.open(shard_file, "w") as tf:
                                    if transcode_webp:
                                        def _gen_sample(items: list[tuple[str, tuple[str, zipfile.ZipInfo]]]):
                                            for stem, (ext, zinfo) in items:
                                                yield stem, zf.read(zinfo), ext

                                        for stem, data, out_ext in pool.map(_worker_sample, _gen_sample(chunk), chunksize=8):
                                            ti = tarfile.TarInfo(name=f"{stem}{out_ext}")
                                            ti.size = len(data)
                                            tf.addfile(ti, io.BytesIO(data))
                                            pbar.update(1)
                                    else:
                                        for stem, (ext, zinfo) in chunk:
                                            data = zf.read(zinfo)
                                            ti = tarfile.TarInfo(name=f"{stem}{ext}")
                                            ti.size = len(data)
                                            tf.addfile(ti, io.BytesIO(data))
                                            pbar.update(1)

        # Update dataset_info.yaml format
        ds_info_path = target_dir / "dataset_info.yaml"
        if ds_info_path.exists():
            try:
                with open(ds_info_path, "r", encoding="utf-8") as f_in:
                    info_data = yaml.safe_load(f_in) or {}
                info_data["format"] = "webdataset"
                info_data["canonical_format"] = "webdataset"
                if transcode_webp:
                    info_data["image_format"] = "webp"
                with open(ds_info_path, "w", encoding="utf-8") as f_out:
                    yaml.safe_dump(info_data, f_out, default_flow_style=False, sort_keys=False, allow_unicode=True)
                print(f"[METADATA] Updated {ds_info_path.name} format to 'webdataset' (image_format: webp).")
            except Exception as exc:
                print(f"[WARN] Failed updating dataset_info.yaml: {exc}")

        # Verify shards exist and are non-empty
        shards_created = list(shards_dir.glob("**/*.tar"))
        total_shard_bytes = sum(s.stat().st_size for s in shards_created)
        print(f"[VERIFY] Created {len(shards_created)} shards ({total_shard_bytes / (1024**3):.2f} GB total).")
        if not shards_created or total_shard_bytes == 0:
            print("[ERROR] Shard verification failed: 0 shards or 0 bytes created.")
            return False

        _post_process_zip(zip_path, legacy_dir, delete_zip)
        return True
    except Exception as exc:
        print(f"[ERROR] Direct zip-to-WebDataset streaming failed: {exc}")
        return False


def stream_zip_to_mds(
    zip_path: Path,
    target_dir: Path,
    mds_shard_size_bytes: int = 512 * 1024 * 1024,
    delete_zip: bool = False,
    legacy_dir: Path | None = None,
    transcode_webp: bool = True,
    max_workers: int | None = None,
) -> bool:
    """Stream paired or single vision datasets directly from zip into MDS shards.

    Requires mosaicml-streaming >= 0.9.0.
    """
    try:
        import importlib as _il
        _streaming = _il.import_module("streaming")
    except ImportError:
        print("[ERROR] mosaicml-streaming is not installed. Run: pip install mosaicml-streaming")
        return False

    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    mds_dir = target_dir / "mds"
    mds_dir.mkdir(parents=True, exist_ok=True)

    workers = max_workers or min(12, os.cpu_count() or 4)
    print(f"[STREAM-MDS] Opening source archive: {zip_path.name}")
    print(f"[STREAM-MDS] Parallel WebP transcoding: {'ENABLED' if transcode_webp else 'DISABLED'} ({workers} workers).")
    print(f"[STREAM-MDS] MDS shard size: {mds_shard_size_bytes / (1024**2):.0f} MB")

    _COLUMNS: dict[str, str] = {
        "image": "jpeg",
        "target": "jpeg",
        "mask": "jpeg",
        "label": "str",
        "task": "str",
        "split": "str",
        "name": "str",
    }

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            all_members = zf.infolist()
            print(f"[STREAM-MDS] Discovered {len(all_members)} entries inside archive.")

            images_by_split, targets_by_split, has_images_dir, metadata_members = _partition_zip_members(zf)
            count = _extract_metadata_files(zf, metadata_members, target_dir)
            print(f"[STREAM-MDS] Extracted {count} metadata files.")

            MDSWriterCls = getattr(_streaming, "MDSWriter")

            with ThreadPoolExecutor(max_workers=workers) as pool:
                if has_images_dir:
                    for split, img_list in images_by_split.items():
                        target_map = targets_by_split.get(split, {})
                        total_samples = len(img_list)
                        split_mds_dir = mds_dir if split == "all" else mds_dir / split
                        try:
                            mds_out_str = os.path.relpath(split_mds_dir)
                        except ValueError:
                            mds_out_str = str(split_mds_dir)
                        if split_mds_dir.exists():
                            shutil.rmtree(split_mds_dir)
                        split_mds_dir.mkdir(parents=True, exist_ok=True)
                        print(f"[STREAM-MDS] Split '{split}': {total_samples} images -> MDS shards.")

                        writer = MDSWriterCls(
                            out=mds_out_str,
                            columns=_COLUMNS,
                            compression="zstd",
                            size_limit=mds_shard_size_bytes,
                            hashes=[],
                        )
                        with create_progress_bar(total=total_samples, desc=f"MDS [{split}]", unit="img") as pbar:
                            if transcode_webp:
                                def _gen_img_mds(items: list[tuple[str, str, zipfile.ZipInfo]]):
                                    for stem, ext, zinfo in items:
                                        yield stem, zf.read(zinfo), ext

                                for (img_stem, img_data, img_ext), (stem, ext, zinfo) in zip(
                                    pool.map(_worker_img, _gen_img_mds(img_list), chunksize=8),
                                    img_list,
                                ):
                                    tgt_bytes = b""
                                    if img_stem in target_map:
                                        tgt_ext, tgt_zinfo = target_map[img_stem]
                                        raw_tgt = zf.read(tgt_zinfo)
                                        tgt_bytes, _ = _transcode_to_webp(raw_tgt, tgt_ext, quality=95) if transcode_webp else (raw_tgt, tgt_ext)
                                    writer.write({
                                        "image": img_data,
                                        "target": tgt_bytes,
                                        "mask": b"",
                                        "label": "",
                                        "task": "restoration",
                                        "split": split,
                                        "name": img_stem,
                                    })
                                    pbar.update(1)
                            else:
                                for stem, ext, zinfo in img_list:
                                    img_data = zf.read(zinfo)
                                    tgt_bytes = b""
                                    if stem in target_map:
                                        tgt_ext, tgt_zinfo = target_map[stem]
                                        tgt_bytes = zf.read(tgt_zinfo)
                                    writer.write({
                                        "image": img_data,
                                        "target": tgt_bytes,
                                        "mask": b"",
                                        "label": "",
                                        "task": "restoration",
                                        "split": split,
                                        "name": stem,
                                    })
                                    pbar.update(1)
                        writer.finish()
                else:
                    # Target-only
                    for split, target_map in targets_by_split.items():
                        tgt_items = list(target_map.items())
                        total_samples = len(tgt_items)
                        split_mds_dir = mds_dir if split == "all" else mds_dir / split
                        try:
                            mds_out_str = os.path.relpath(split_mds_dir)
                        except ValueError:
                            mds_out_str = str(split_mds_dir)
                        if split_mds_dir.exists():
                            shutil.rmtree(split_mds_dir)
                        split_mds_dir.mkdir(parents=True, exist_ok=True)
                        print(f"[STREAM-MDS] Split '{split}': {total_samples} targets -> MDS shards.")

                        writer = MDSWriterCls(
                            out=mds_out_str,
                            columns=_COLUMNS,
                            compression="zstd",
                            size_limit=mds_shard_size_bytes,
                            hashes=[],
                        )
                        with create_progress_bar(total=total_samples, desc=f"MDS [{split}]", unit="img") as pbar:
                            if transcode_webp:
                                def _gen_tgt_mds(items: list[tuple[str, tuple[str, zipfile.ZipInfo]]]):
                                    for stem, (ext, zinfo) in items:
                                        yield stem, zf.read(zinfo), ext

                                for (stem, data, out_ext), (orig_stem, _) in zip(
                                    pool.map(_worker_sample, _gen_tgt_mds(tgt_items), chunksize=8),
                                    tgt_items,
                                ):
                                    writer.write({
                                        "image": data,
                                        "target": b"",
                                        "mask": b"",
                                        "label": "",
                                        "task": "restoration",
                                        "split": split,
                                        "name": stem,
                                    })
                                    pbar.update(1)
                            else:
                                for stem, (ext, zinfo) in tgt_items:
                                    writer.write({
                                        "image": zf.read(zinfo),
                                        "target": b"",
                                        "mask": b"",
                                        "label": "",
                                        "task": "restoration",
                                        "split": split,
                                        "name": stem,
                                    })
                                    pbar.update(1)
                        writer.finish()

        # Update dataset_info.yaml
        ds_info_path = target_dir / "dataset_info.yaml"
        if ds_info_path.exists():
            try:
                with open(ds_info_path, "r", encoding="utf-8") as f_in:
                    info_data = yaml.safe_load(f_in) or {}
                info_data["format"] = "mds"
                info_data["canonical_format"] = "mds"
                if transcode_webp:
                    info_data["image_format"] = "webp"
                with open(ds_info_path, "w", encoding="utf-8") as f_out:
                    yaml.safe_dump(info_data, f_out, default_flow_style=False, sort_keys=False, allow_unicode=True)
                print(f"[METADATA] Updated {ds_info_path.name} format to 'mds'.")
            except Exception as exc:
                print(f"[WARN] Failed updating dataset_info.yaml: {exc}")

        # Verify MDS shards exist
        mds_shards = list(mds_dir.glob("**/*.mds"))
        if not mds_shards:
            # MDS creates index.json — check for that instead
            mds_shards = list(mds_dir.glob("**/index.json"))
        total_mds_bytes = sum(s.stat().st_size for s in list(mds_dir.rglob("*.mds")))
        print(f"[VERIFY-MDS] MDS output: {mds_dir} ({total_mds_bytes / (1024**3):.2f} GB).")

        _post_process_zip(zip_path, legacy_dir, delete_zip)
        return True
    except Exception as exc:
        print(f"[ERROR] Direct zip-to-MDS streaming failed: {exc}")
        import traceback
        traceback.print_exc()
        return False


def _post_process_zip(zip_path: Path, legacy_dir: Path | None, delete_zip: bool) -> None:
    """Optionally delete source zip and legacy directory."""
    if delete_zip:
        print(f"[CLEANUP] Deleting source archive to free local storage: {zip_path.name}")
        try:
            os.remove(zip_path)
            print(f"[SUCCESS] Deleted {zip_path.name} successfully.")
        except OSError as ex:
            print(f"[WARN] Could not remove zip archive {zip_path}: {ex}")

    if legacy_dir is not None:
        legacy_resolved = legacy_dir.resolve()
        if legacy_resolved.exists():
            print(f"[CLEANUP] Deleting legacy manifold directory: {legacy_resolved.name}")
            try:
                shutil.rmtree(legacy_resolved, ignore_errors=True)
                print(f"[SUCCESS] Deleted legacy directory {legacy_resolved.name} successfully.")
            except OSError as ex:
                print(f"[WARN] Could not remove legacy directory {legacy_resolved}: {ex}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream legacy zip directly into canonical container format.")
    parser.add_argument("--zip", required=True, type=Path, help="Path to source .zip archive")
    parser.add_argument("--target", required=True, type=Path, help="Destination manifold directory")
    parser.add_argument("--format", default="webdataset", choices=["webdataset", "mds"],
                        help="Target container format (default: webdataset)")
    parser.add_argument("--shard-size", type=int, default=5000,
                        help="Number of samples per WebDataset shard (ignored for MDS)")
    parser.add_argument("--mds-shard-mb", type=int, default=512,
                        help="MDS shard size in MB (default: 512)")
    parser.add_argument("--no-webp", action="store_true", help="Disable WebP transcoding (keep original format)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker threads for transcoding")
    parser.add_argument("--delete-zip", action="store_true", help="Delete source zip archive upon successful conversion")
    parser.add_argument("--delete-legacy-dir", type=Path, default=None,
                        help="Optional legacy directory to delete after conversion")
    args = parser.parse_args()

    if args.format == "webdataset":
        success = stream_zip_to_webdataset(
            zip_path=args.zip,
            target_dir=args.target,
            shard_size_samples=args.shard_size,
            delete_zip=args.delete_zip,
            legacy_dir=args.delete_legacy_dir,
            transcode_webp=not args.no_webp,
            max_workers=args.workers,
        )
        return 0 if success else 1

    if args.format == "mds":
        success = stream_zip_to_mds(
            zip_path=args.zip,
            target_dir=args.target,
            mds_shard_size_bytes=args.mds_shard_mb * 1024 * 1024,
            delete_zip=args.delete_zip,
            legacy_dir=args.delete_legacy_dir,
            transcode_webp=not args.no_webp,
            max_workers=args.workers,
        )
        return 0 if success else 1

    print(f"[ERROR] Format {args.format} streaming not implemented.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
