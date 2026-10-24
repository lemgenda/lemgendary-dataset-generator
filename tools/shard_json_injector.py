import io
import os
import shutil
import tarfile
from pathlib import Path


def process_shards():
    base_dir = Path(r"C:\Development\python\model-training\LemGendaryDatasets")

    models = [
        "LemGendizedNimaAesthetic",
        "LemGendizedNimaAuthenticity",
        "LemGendizedNimaTechnical",
        "LemGendizedRetinaFace",
    ]

    splits = ["train", "val"]

    for model in models:
        model_path = base_dir / model
        print("\n========================================")
        print(f"Starting Dataset Upgrade: {model}")
        print("========================================")

        for split in splits:
            shards_dir = model_path / "shards" / split
            json_dir = model_path / "json-labels" / split

            if not shards_dir.exists():
                print(f"Skipping split {split}: Path does not exist.")
                continue

            # Track mismatched images and unused labels globally per split
            unmatched_images = []
            matched_json_stems = set()

            # Find all shard files with the current .data extension
            shard_files = list(shards_dir.glob("*.data"))
            if not shard_files:
                print(f"No .data shards found in {split} folder.")
                continue

            print(f"\nProcessing [{split}] split - Found {len(shard_files)} shards.")

            for shard_file in shard_files:
                # 1. Rename .data to .tar
                tar_path = shard_file.with_suffix(".tar")
                shard_file.rename(tar_path)
                print(f"  -> Opened: {tar_path.name}")

                # Setup paths for a temporary rebuild strategy
                temp_output_tar = tar_path.with_name(tar_path.stem + "_temp.tar")

                try:
                    with tarfile.open(tar_path, "r") as src_tar, tarfile.open(temp_output_tar, "w") as dest_tar:
                        # 2. List and iterate through members
                        for member in src_tar.getmembers():
                            extracted_file = src_tar.extractfile(member)

                            # Safe check: if it's a directory or link, extractfile returns None
                            if extracted_file is not None:
                                file_bytes = extracted_file.read()
                                dest_tar.addfile(member, io.BytesIO(file_bytes))
                            else:
                                # Safe fallback to copy directories/links directly without reading bytes
                                dest_tar.addfile(member)
                                continue

                            # 3. If it's a webp image, check for a matching label
                            if member.name.lower().endswith(".webp"):
                                stem = Path(member.name).stem
                                json_file = json_dir / f"{stem}.json"

                                if json_file.exists():
                                    # 4. Insert corresponding JSON metadata cleanly side-by-side
                                    json_bytes = json_file.read_bytes()

                                    json_member = tarfile.TarInfo(name=f"{stem}.json")
                                    json_member.size = len(json_bytes)

                                    dest_tar.addfile(json_member, io.BytesIO(json_bytes))
                                    matched_json_stems.add(stem)
                                else:
                                    unmatched_images.append(f"{shard_file.name} -> {member.name}")

                except Exception as e:
                    print(f"    Error processing shard archive {tar_path.name}: {e}")
                    # Revert to original naming structure if processing breaks mid-way
                    if tar_path.exists() and not shard_file.exists():
                        tar_path.rename(shard_file)
                    continue

                # Overwrite original tar with the updated bundle
                src_tar.close()
                os.remove(tar_path)
                shutil.move(str(temp_output_tar), str(tar_path))

                # 5. Rename archive back from .tar to .data
                tar_path.rename(shard_file)
                print(f"  <- Finalized & Restored: {shard_file.name}")

            # 6. Calculate mismatched/unmatched labels left over in local folder
            all_json_stems = {f.stem for f in json_dir.glob("*.json")}
            unmatched_labels = all_json_stems - matched_json_stems

            # Print comprehensive summary logs for this run
            print(f"\n--- Split Summary [{split}] ---")
            print(f"Total Unmatched Images: {len(unmatched_images)}")
            if unmatched_images:
                for img in unmatched_images[:10]:  # Cap preview at 10 items
                    print(f"  Missing label for image: {img}")
                if len(unmatched_images) > 10:
                    print(f"  ... and {len(unmatched_images) - 10} more.")

            print(f"Total Unused/Orphan JSON Labels: {len(unmatched_labels)}")
            if unmatched_labels:
                for lbl in list(unmatched_labels)[:10]:
                    print(f"  Orphan Label: {lbl}.json")
                if len(unmatched_labels) > 10:
                    print(f"  ... and {len(unmatched_labels) - 10} more.")


if __name__ == "__main__":
    process_shards()
