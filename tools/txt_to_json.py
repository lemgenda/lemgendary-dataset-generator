"""Convert the NIMA and RetinaFace .txt labels of the legacy manifolds to JSON files."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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


def _label_payload(file_path: Path, model_type: str, lines: list[str]) -> dict:
    """Structure JSON payload based on the model's objective."""
    if model_type == "nima":
        return {"filename": f"{file_path.stem}.jpg", "nima_scores": parse_nima_line(lines[0])}
    # retinaface
    return {"filename": f"{file_path.stem}.jpg", **parse_retinaface_lines(lines)}


def _convert_file(file_path: Path, model_type: str, dest_json_dir: Path) -> bool:
    """Write the JSON label for one .txt label; False when it is empty or unreadable."""
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return False

        json_data = _label_payload(file_path, model_type, lines)

        # Write output JSON
        output_file = dest_json_dir / f"{file_path.stem}.json"
        with open(output_file, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, indent=2)
        return True

    except (OSError, ValueError, IndexError) as e:
        logger.warning("Error converting label file %s: %s", file_path, e)
        print(f"  Error processing {file_path.name}: {e}")
        return False


def _convert_split(model_path: Path, model_type: str, split: str) -> None:
    src_label_dir = model_path / "labels" / split
    dest_json_dir = model_path / "json-labels" / split

    if not src_label_dir.exists():
        print(f"  Warning: Directory not found, skipping split: {src_label_dir}")
        return

    dest_json_dir.mkdir(parents=True, exist_ok=True)
    txt_files = list(src_label_dir.glob("*.txt"))

    if not txt_files:
        print(f"  Info: No .txt files in {split} split.")
        return

    print(f"  Found {len(txt_files)} files in '{split}'. Converting...")
    converted_count = sum(
        1 for file_path in txt_files if _convert_file(file_path, model_type, dest_json_dir)
    )
    print(f"  Finished '{split}': Converted {converted_count} files.")


def convert_labels():
    # Base configuration based on your paths
    """Write a JSON label file next to every .txt label of the configured manifolds."""
    base_dir = Path(__file__).resolve().parent.parent.parent / "LemGendaryDatasets"

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
            _convert_split(model_path, model_type, split)


if __name__ == "__main__":
    convert_labels()
