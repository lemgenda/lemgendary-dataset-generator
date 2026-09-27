"""
LemGendary Dataset Compiler — Streaming Zip-to-Container Engine.

Directly streams multi-gigabyte legacy dataset zip archives into modern
canonical containers (WebDataset tar shards) with on-the-fly multi-threaded
WebP transcoding, without ever extracting loose files to disk.
Eliminates Windows NTFS MFT lock contention, cluster slack, and antivirus overhead.
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
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


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
            total_members = len(all_members)
            print(f"[STREAM] Discovered {total_members} entries inside archive.")

            # 1. Extract root metadata files directly (README, yaml, txt, json, ipynb)
            metadata_extracted = 0
            for m in all_members:
                norm_name = m.filename.replace("\\", "/")
                parts = [p for p in norm_name.split("/") if p]
                if len(parts) == 1 or (len(parts) == 2 and not m.is_dir() and not parts[0].lower() in ("images", "targets", "masks")):
                    base_name = parts[-1]
                    if any(base_name.endswith(ext) for ext in [".yaml", ".yml", ".md", ".json", ".txt", ".ipynb"]):
                        dest_file = target_dir / base_name
                        print(f"  -> Extracting metadata: {base_name} ({m.file_size:,} bytes)")
                        with zf.open(m) as src, open(dest_file, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        metadata_extracted += 1
            print(f"[STREAM] Extracted {metadata_extracted} root metadata/notebook files to {target_dir.name}.")

            # 2. Partition image and target members
            images_by_split: dict[str, list[tuple[str, str, zipfile.ZipInfo]]] = {}
            targets_by_split: dict[str, dict[str, tuple[str, zipfile.ZipInfo]]] = {}
            has_images_dir = False

            for m in all_members:
                if m.is_dir():
                    continue
                norm_name = m.filename.replace("\\", "/")
                parts = norm_name.split("/")
                if len(parts) < 2:
                    continue
                fname = parts[-1]
                dot_pos = fname.rfind(".")
                if dot_pos <= 0:
                    continue
                ext = fname[dot_pos:].lower()
                stem = fname[:dot_pos]
                if ext not in _IMAGE_EXTS:
                    continue

                # Determine split
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
                        with tqdm(total=total_samples, desc=f"IMAGES -> WEBP [{split}]", unit="img", dynamic_ncols=True, mininterval=0.5) as pbar:
                            for shard_idx in range(total_shards):
                                start_i = shard_idx * shard_size_samples
                                end_i = min(start_i + shard_size_samples, total_samples)
                                shard_file = split_shards_dir / f"shard-{shard_idx:05d}.tar"

                                chunk = img_list[start_i:end_i]
                                raw_items = [(stem, zf.read(zinfo), ext) for stem, ext, zinfo in chunk]

                                if transcode_webp:
                                    def _worker_img(item: tuple[str, bytes, str]) -> tuple[str, bytes, str]:
                                        s, raw, e = item
                                        out_bytes, out_ext = _transcode_to_webp(raw, e, quality=92)
                                        return s, out_bytes, out_ext

                                    encoded_items = list(pool.map(_worker_img, raw_items))
                                else:
                                    encoded_items = [(s, raw, e) for s, raw, e in raw_items]

                                with tarfile.open(shard_file, "w") as tf:
                                    for stem, data, out_ext in encoded_items:
                                        ti = tarfile.TarInfo(name=f"{stem}{out_ext}")
                                        ti.size = len(data)
                                        tf.addfile(ti, io.BytesIO(data))
                                pbar.update(len(chunk))

                        # Pass 2: Append targets with WebP transcoding
                        if target_map:
                            with tqdm(total=total_samples, desc=f"TARGETS -> WEBP [{split}]", unit="img", dynamic_ncols=True, mininterval=0.5) as pbar:
                                for shard_idx in range(total_shards):
                                    start_i = shard_idx * shard_size_samples
                                    end_i = min(start_i + shard_size_samples, total_samples)
                                    shard_file = split_shards_dir / f"shard-{shard_idx:05d}.tar"

                                    chunk = img_list[start_i:end_i]
                                    tgt_chunk: list[tuple[str, bytes, str]] = []
                                    for stem, _, _ in chunk:
                                        if stem in target_map:
                                            tgt_ext, tgt_zinfo = target_map[stem]
                                            tgt_chunk.append((stem, zf.read(tgt_zinfo), tgt_ext))

                                    if tgt_chunk:
                                        if transcode_webp:
                                            def _worker_tgt(item: tuple[str, bytes, str]) -> tuple[str, bytes, str]:
                                                s, raw, e = item
                                                out_bytes, out_ext = _transcode_to_webp(raw, e, quality=95)
                                                return s, out_bytes, out_ext

                                            encoded_tgts = list(pool.map(_worker_tgt, tgt_chunk))
                                        else:
                                            encoded_tgts = [(s, raw, e) for s, raw, e in tgt_chunk]

                                        with tarfile.open(shard_file, "a") as tf:
                                            for stem, data, out_ext in encoded_tgts:
                                                ti = tarfile.TarInfo(name=f"{stem}.target{out_ext}")
                                                ti.size = len(data)
                                                tf.addfile(ti, io.BytesIO(data))
                                    pbar.update(len(chunk))

                # Case B: Target-only dataset (targets/train, targets/val, e.g. UpnV2)
                else:
                    for split, target_map in targets_by_split.items():
                        split_shards_dir = shards_dir if split == "all" else shards_dir / split
                        split_shards_dir.mkdir(parents=True, exist_ok=True)
                        tgt_items = list(target_map.items())
                        total_samples = len(tgt_items)
                        total_shards = (total_samples + shard_size_samples - 1) // shard_size_samples
                        print(f"[STREAM] Split '{split}': {total_samples} target samples -> {total_shards} shards.")

                        with tqdm(total=total_samples, desc=f"SAMPLES -> WEBP [{split}]", unit="img", dynamic_ncols=True, mininterval=0.5) as pbar:
                            for shard_idx in range(total_shards):
                                start_i = shard_idx * shard_size_samples
                                end_i = min(start_i + shard_size_samples, total_samples)
                                shard_file = split_shards_dir / f"shard-{shard_idx:05d}.tar"

                                chunk = tgt_items[start_i:end_i]
                                raw_items = [(stem, zf.read(zinfo), ext) for stem, (ext, zinfo) in chunk]

                                if transcode_webp:
                                    def _worker_sample(item: tuple[str, bytes, str]) -> tuple[str, bytes, str]:
                                        s, raw, e = item
                                        out_bytes, out_ext = _transcode_to_webp(raw, e, quality=95)
                                        return s, out_bytes, out_ext

                                    encoded_samples = list(pool.map(_worker_sample, raw_items))
                                else:
                                    encoded_samples = [(s, raw, e) for s, raw, e in raw_items]

                                with tarfile.open(shard_file, "w") as tf:
                                    for stem, data, out_ext in encoded_samples:
                                        ti = tarfile.TarInfo(name=f"{stem}{out_ext}")
                                        ti.size = len(data)
                                        tf.addfile(ti, io.BytesIO(data))
                                pbar.update(len(chunk))

        # 3. Update dataset_info.yaml format
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

        # 4. Verify shards exist and are non-empty
        shards_created = list(shards_dir.glob("**/*.tar"))
        total_shard_bytes = sum(s.stat().st_size for s in shards_created)
        print(f"[VERIFY] Created {len(shards_created)} shards ({total_shard_bytes / (1024**3):.2f} GB total).")
        if not shards_created or total_shard_bytes == 0:
            print("[ERROR] Shard verification failed: 0 shards or 0 bytes created.")
            return False

        # 5. Delete source zip archive if requested
        if delete_zip:
            print(f"[CLEANUP] Deleting source archive to free local storage: {zip_path.name}")
            try:
                os.remove(zip_path)
                print(f"[SUCCESS] Deleted {zip_path.name} successfully.")
            except OSError as ex:
                print(f"[WARN] Could not remove zip archive {zip_path}: {ex}")

        # 6. Delete legacy directory if specified
        if legacy_dir is not None and legacy_dir.resolve() != target_dir.resolve():
            legacy_resolved = legacy_dir.resolve()
            if legacy_resolved.exists():
                print(f"[CLEANUP] Deleting legacy manifold directory: {legacy_resolved.name}")
                try:
                    shutil.rmtree(legacy_resolved, ignore_errors=True)
                    print(f"[SUCCESS] Deleted legacy directory {legacy_resolved.name} successfully.")
                except OSError as ex:
                    print(f"[WARN] Could not remove legacy directory {legacy_resolved}: {ex}")

        return True
    except Exception as exc:
        print(f"[ERROR] Direct zip-to-WebDataset streaming failed: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream legacy zip directly into canonical container format.")
    parser.add_argument("--zip", required=True, type=Path, help="Path to source .zip archive")
    parser.add_argument("--target", required=True, type=Path, help="Destination manifold directory")
    parser.add_argument("--format", default="webdataset", choices=["webdataset"], help="Target container format")
    parser.add_argument("--shard-size", type=int, default=5000, help="Number of samples per shard")
    parser.add_argument("--no-webp", action="store_true", help="Disable WebP transcoding (keep original format)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker threads for transcoding")
    parser.add_argument("--delete-zip", action="store_true", help="Delete source zip archive upon successful conversion")
    parser.add_argument("--delete-legacy-dir", type=Path, default=None, help="Optional legacy directory to delete after conversion")
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

    print(f"[ERROR] Format {args.format} streaming not yet implemented.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
