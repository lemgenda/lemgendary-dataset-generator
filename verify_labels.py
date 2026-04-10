import os
import json
import cv2
import argparse
import numpy as np
from pathlib import Path
import random

# ---------- CONFIG ----------
COMPILED_DIR = Path("./compiled-datasets")
VERIFY_DIR = Path("./verification")
NUM_SAMPLES = 8

def draw_yolo(img, labels):
    h, w = img.shape[:2]
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    
    for line in labels:
        parts = list(map(float, line.split()))
        cls = int(parts[0])
        color = colors[cls % len(colors)]
        data = parts[1:]
        
        if len(data) == 4: # Bbox: x_center y_center w h
            xc, yc, bw, bh = data
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, str(cls), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        elif len(data) > 4: # Segmentation or Pose
            # If it's a long list and looks like polygon: x y x y...
            # Note: YOLO Pose is box + points. Simple check: if len parts is Odd...
            # For this visualizer, let's treat long list > 4 as polygon or points
            points = np.array(data).reshape(-1, 2)
            cv2.polylines(img, [npx.astype(np.int32) for npx in [(points * [w, h])]], True, color, 1)
            for p in points:
                cv2.circle(img, (int(p[0]*w), int(p[1]*h)), 3, color, -1)

def verify():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Dataset folder name")
    args = parser.parse_args()

    # Dataset Selection
    subsets = [d for d in COMPILED_DIR.iterdir() if d.is_dir()]
    if args.name:
        dataset_path = COMPILED_DIR / args.name
    elif len(subsets) == 1:
        dataset_path = subsets[0]
    elif len(subsets) > 1:
        print("\n📂 Multiple datasets found:")
        for i, s in enumerate(subsets):
            print(f"  {i+1}. {s.name}")
        choice = input(f"\nSelect dataset (1-{len(subsets)}): ")
        try:
            dataset_path = subsets[int(choice)-1]
        except:
            print("❌ Invalid selection.")
            return
    else:
        print("❌ No compiled datasets found.")
        return

    index_path = dataset_path / "index.json"
    if not index_path.exists():
        print(f"❌ No index.json found in {dataset_path.name}")
        return

    VERIFY_DIR.mkdir(exist_ok=True)
    with open(index_path) as f:
        data = json.load(f)

    if not data:
        print("❌ Dataset is empty.")
        return

    samples = random.sample(data, min(len(data), NUM_SAMPLES))

    print(f"🔍 [VERIFY] Generating {len(samples)} samples for [{dataset_path.name}]...")
    for item in samples:
        img_path = Path(item["path"])
        label_path = Path(item["label_path"])
        
        if not img_path.exists() or not label_path.exists():
            print(f"  ⚠️ Missing: {img_path.name}")
            continue
        
        img = cv2.imread(str(img_path))
        with open(label_path) as f:
            labels = f.readlines()
        
        draw_yolo(img, labels)
        
        out_path = VERIFY_DIR / f"verify_{dataset_path.name}_{item['name']}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"  ✅ Saved: {out_path.name}")

if __name__ == "__main__":
    verify()
