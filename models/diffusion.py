import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class CaptionSentry:
    """
    Automated image caption generation using BLIP.
    Essential for diffusion-task datasets.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # Using the base model for optimal balance of speed and descriptive fidelity
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, img_pil):
        try:
            inputs = self.processor(img_pil, return_tensors="pt").to(self.device)
            # 2026 Resilience: We use greedy search for speed
            out = self.model.generate(**inputs, max_new_tokens=50)
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception:
            return "a high quality image" # Fallback safety descriptor
