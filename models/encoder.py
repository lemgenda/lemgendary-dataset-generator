import torch
from transformers import CLIPProcessor, CLIPModel

class CLIPManifold:
    """
    Advanced Multimodal Manifold for Style Clustering and Aesthetic Scoring.
    Utilizes CLIP features for zero-shot style tagging.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_id = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        
        # Standard Style Manifold for Zero-Shot Tagging
        self.styles = ["photo", "anime", "cg-art", "sketch", "oil-painting", "vector-art", "minimalist"]

    @torch.no_grad()
    def extract_features(self, img_pil):
        """Extracts latent vector for style clustering"""
        inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        # Normalize for cosine similarity / clustering stability
        return image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    @torch.no_grad()
    def tag_style(self, img_pil):
        """Zero-shot style classification"""
        inputs = self.processor(
            text=self.styles, images=img_pil, return_tensors="pt", padding=True
        ).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image 
        probs = logits_per_image.softmax(dim=1)
        best_idx = probs.argmax().item()
        return self.styles[best_idx]
