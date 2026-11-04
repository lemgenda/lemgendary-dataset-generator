"""NIMA-based image quality scoring."""

import torch
from torchvision import transforms

from .nima import NimaModel


class QualitySentry:
    """
    Automated image quality vetting using a pre-trained NIMA model.
    """

    def __init__(
        self,
        model_path,
        model_name="nima_technical",
        device=None,
    ):
        # None picks CUDA when it is available (resolved per instance, not at import).
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Technical NIMA uses EfficientNetV2-S, Aesthetic uses MobileNetV3-Small
        backbone = "mobilenet_v2" if "aesthetic" in model_name else "efficientnet_v2_s"
        self.model = NimaModel(backbone=backbone).to(self.device)

        # Load SOTA Weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # Reverted to 224 for standard NIMA compatibility
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def rate(self, img_pil) -> tuple[float, list[float]]:
        """An image's mean NIMA score (1..10) and its 10-bin score distribution."""
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        mean_score, probs = NimaModel.expected_score(self.model(img_tensor))
        return mean_score.item(), probs[0].cpu().numpy().tolist()

    def score(self, img_pil) -> float:
        """An image's mean NIMA score (1..10)."""
        return self.rate(img_pil)[0]
