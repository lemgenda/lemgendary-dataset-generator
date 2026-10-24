import json
from pathlib import Path


def parse_nima_line(line: str):
    """Parses 10 float values representing NIMA score distribution."""
    parts = line.strip().split()
    if len(parts) != 10:
        raise ValueError(f"Expected 10 values for NIMA, got {len(parts)}")

    distribution = [float(x) for x in parts]

    # Calculate a weighted mean score (assuming rating scale 1 to 10)
    mean_score = sum((i + 1) * prob for i, prob in enumerate(distribution))

    return {"distribution": distribution, "mean_score": round(mean_score, 4)}


def parse_retinaface_lines(lines: list):
    """Parses standard YOLO/RetinaFace bounding boxes (can have multiple faces)."""
    faces = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:5]]

        faces.append({"category_id": class_id, "bbox_cxcywh_normalized": coords})
    return {"face_count": len(faces), "objects": faces}


def convert_labels():
    # Base configuration based on your paths
    base_dir = Path(r"C:\Development\python\model-training\LemGendaryDatasets")

    models = {
        "LemGendizedNimaAesthetic": "nima",
        "LemGendizedNimaAuthenticity": "nima",
        "LemGendizedNimaTechnical": "nima",
        "LemGendizedRetinaFace": "retinaface",
    }

    splits = ["train", "val"]

    for model_name, model_type in models.items():
        model_path = base_dir / model_name
        print(f"\nProcessing model: {model_name} ({model_type})")

        for split in splits:
            src_label_dir = model_path / "labels" / split
            dest_json_dir = model_path / "json-labels" / split

            if not src_label_dir.exists():
                print(f"  Warning: Directory not found, skipping split: {src_label_dir}")
                continue

            dest_json_dir.mkdir(parents=True, exist_ok=True)
            txt_files = list(src_label_dir.glob("*.txt"))

            if not txt_files:
                print(f"  Info: No .txt files in {split} split.")
                continue

            print(f"  Found {len(txt_files)} files in '{split}'. Converting...")
            converted_count = 0

            for file_path in txt_files:
                try:
                    with open(file_path, encoding="utf-8") as f:
                        lines = f.readlines()

                    if not lines:
                        continue

                    # Structure JSON payload based on the model's objective
                    if model_type == "nima":
                        parsed_data = parse_nima_line(lines[0])
                        json_data = {"filename": f"{file_path.stem}.jpg", "nima_scores": parsed_data}
                    else:  # retinaface
                        parsed_data = parse_retinaface_lines(lines)
                        json_data = {"filename": f"{file_path.stem}.jpg", **parsed_data}

                    # Write output JSON
                    output_file = dest_json_dir / f"{file_path.stem}.json"
                    with open(output_file, "w", encoding="utf-8") as jf:
                        json.dump(json_data, jf, indent=2)

                    converted_count += 1

                except Exception as e:
                    print(f"  Error processing {file_path.name}: {e}")

            print(f"  Finished '{split}': Converted {converted_count} files.")


if __name__ == "__main__":
    convert_labels()
