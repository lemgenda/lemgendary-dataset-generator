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
            # Fallback/Safety: Skip or provide a dummy
            return self.__getitem__((idx + 1) % len(self))
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = img.copy()

        # Apply synthetic degradation dynamically based on task
        degraded, meta = degrade(img, task=item["task"])

        # Convert to Tensor (CHW format, normalized to [0,1])
        input_tensor = torch.from_numpy(degraded).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(target).permute(2, 0, 1).float() / 255.0

        return {
            "input": input_tensor,
            "target": target_tensor,
            "task": item["task"],
            "meta": meta,
            "path": str(img_path)
        }