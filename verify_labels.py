import os
import json
import cv2
import numpy as np
from pathlib import Path
import random

# ---------- CONFIG ----------
COMPILED_DIR = Path("./compiled-datasets")
INDEX_PATH = COMPILED_DIR / "index.json"
VERIFY_DIR = Path("./verification")
NUM_SAMPLES = 5

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
    if not INDEX_PATH.exists():
        print("❌ No index.json found. Run compiler-pipeline.py first.")
        return

    VERIFY_DIR.mkdir(exist_ok=True)
    with open(INDEX_PATH) as f:
        data = json.load(f)

    if not data: return
    samples = random.sample(data, min(len(data), NUM_SAMPLES))

    print(f"🔍 [VERIFY] Generating {len(samples)} visual verification samples...")
    for item in samples:
        img_path = Path(item["path"])
        label_path = Path(item["label_path"])
        
        if not img_path.exists() or not label_path.exists(): continue
        
        img = cv2.imread(str(img_path))
        with open(label_path) as f:
            labels = f.readlines()
        
        draw_yolo(img, labels)
        
        out_path = VERIFY_DIR / f"verify_{item['name']}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"  ✅ Saved: {out_path.name}")

if __name__ == "__main__":
    verify()
