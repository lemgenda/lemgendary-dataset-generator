import os
import json
from pathlib import Path

BASE = Path("../raw-sets")
OUT = Path("../processed/index.json")

TASK_MAP = {
    "nima_aesthetic": "quality_aesthetic",
    "nima_technical": "quality_technical",
    "codeformer": "face_restoration",
    "parsenet": "face_parsing",
    "retinaface": "face_detection",
    "yolov8n": "object_detection",
    "ultrazoom": "super_resolution",
    "ffanet": "dehaze",
    "mprnet": "derain",
    "mirnet": "lowlight_exposure",
    "nafnet": "denoise_deblur"
}

def scan_images(folder):
    return list(folder.rglob("*.jpg")) + list(folder.rglob("*.png"))

def build():
    index = []

    for model_dir in BASE.iterdir():
        if not model_dir.is_dir():
            continue

        task = TASK_MAP.get(model_dir.name, "unknown")

        for dataset in model_dir.iterdir():
            if not dataset.is_dir():
                continue

            images = scan_images(dataset)

            for img in images:
                entry = {
                    "path": str(img),
                    "task": task,
                    "dataset": dataset.name,
                    "degradation": None,  # filled later dynamically
                }
                index.append(entry)

    OUT.parent.mkdir(exist_ok=True, parents=True)
    with open(OUT, "w") as f:
        json.dump(index, f, indent=2)

    print(f"✅ Indexed {len(index)} images")

if __name__ == "__main__":
    build()
