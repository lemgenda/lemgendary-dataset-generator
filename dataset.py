import json
import cv2
import numpy as np
import torch
from pathlib import Path
from transforms import degrade

class MultiTaskDataset(torch.utils.data.Dataset):
    """
    Generic Multi-Task Dataset for restoration and detection training.
    Loads metadata from dataset_index.json and applies synthetic degradations.
    """
    def __init__(self, index_path):
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found at {index_path}")
            
        with open(self.index_path) as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = Path(item["path"])

        # Load image via OpenCV
        img = cv2.imread(str(img_path))
        if img is None:
            return self.__getitem__((idx + 1) % len(self))
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = img.copy()

        # Apply synthetic degradation
        degraded, meta = degrade(img, task=item["task"])

        # Load Multitask Labels if present
        labels = []
        label_path = item.get("label_path")
        if label_path and Path(label_path).exists():
            with open(label_path, "r") as f:
                for line in f:
                    if line.strip():
                        # YOLO format: [cls, x1, y1, ...]
                        parts = list(map(float, line.split()))
                        labels.append(parts)

        # Convert to Tensor
        input_tensor = torch.from_numpy(degraded).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(target).permute(2, 0, 1).float() / 255.0

        return {
            "input": input_tensor,
            "target": target_tensor,
            "task": item["task"],
            "labels": torch.tensor(labels) if labels else None,
            "meta": meta,
            "path": str(img_path),
            "nima_score": item.get("nima_score", 0),
            "is_autolabeled": item.get("is_autolabeled", False)
        }