import torch
import torch.nn as nn
from torchvision import transforms
from models.nima import NIMA_Model

class QualitySentry:
    """
    Automated image quality vetting using a pre-trained NIMA model.
    """
    def __init__(self, model_path, model_name="nima_technical", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # Technical NIMA uses EfficientNetV2-S, Aesthetic uses MobileNetV2
        backbone = "mobilenet_v2" if "aesthetic" in model_name else "efficientnet_v2_s"
        self.model = NIMA_Model(backbone=backbone).to(self.device)
        
        # Load SOTA Weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Diagnostic: Checksum of classifier weights to detect 'Zombie Models'
        w_sum = self.model.classifier[1].weight.sum().item()
        print(f"📊 [JUDGE] NIMA Model Loaded. Weight Checksum: {w_sum:.6f}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # Reverted to 224 for standard NIMA compatibility
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def score(self, img_pil, return_probs=False):
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(img_tensor)
            
            # DIAGNOSTIC: Print raw logit spread for the first few images
            if getattr(self, "_diag_count", 0) < 5:
                self._diag_count = getattr(self, "_diag_count", 0) + 1
                l_min, l_max = logits.min().item(), logits.max().item()
                l_var = logits.var().item()
                print(f"🧠 [NEURAL] Logit Spread: {l_min:.2f} to {l_max:.2f} | Var: {l_var:.4f}")

            probs = nn.functional.softmax(logits, dim=1)
        
        # Calculate mean score from distribution [1, 10]
        weights = torch.arange(1, 11).to(self.device).float()
        mean_score = torch.sum(probs * weights, dim=1).item()
        
        if return_probs:
            return mean_score, probs[0].cpu().numpy().tolist()
        return mean_score
