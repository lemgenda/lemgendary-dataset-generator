"""Regression tests for lemgendary-datasets data pipeline under Python 3.14 & NumPy 2.x."""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.detection import AutoLabeler


class SyntheticDataset(Dataset):
    """Synthetic PyTorch dataset for DataLoader multiprocessing verification."""

    def __init__(self, size: int = 16):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.randn(3, 64, 64),
            "label": torch.tensor(index % 10, dtype=torch.long),
        }


def test_dataloader_multiprocessing_roundtrip() -> None:
    """Test DataLoader with num_workers=2 on Python 3.14."""
    ds = SyntheticDataset(size=16)
    loader = DataLoader(ds, batch_size=4, num_workers=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 4
    assert batches[0]["image"].shape == (4, 3, 64, 64)


def test_mediapipe_numpy2_stride_safety() -> None:
    """Verify non-contiguous sliced RGB array handling with np.ascontiguousarray."""
    annotator = AutoLabeler()
    # Create non-contiguous array via slicing/striding
    base_arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    non_contiguous = base_arr[:, ::2, ::-1]
    assert not non_contiguous.flags["C_CONTIGUOUS"]

    # Wrap in PIL and run through face_landmarks
    pil_img = Image.fromarray(non_contiguous)
    # This should execute cleanly without ctypes stride error
    annotations = annotator.face_landmarks(pil_img)
    assert isinstance(annotations, list)


def test_transformers_and_hf_hub_loading() -> None:
    """Test tokenizer and config imports from transformers and huggingface_hub."""
    from huggingface_hub import constants
    from transformers import AutoConfig
    assert constants is not None
    assert AutoConfig is not None
