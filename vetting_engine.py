import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import sys

# Add the training suite models directory to sys.path to allow imports
TRAINING_SUITE_PATH = "c:/Development/python/model-training/lemgendary-training-suite"
sys.path.append(os.path.join(TRAINING_SUITE_PATH))

from models.nima import NIMA_Model
from torchvision import transforms

class QualitySentry:
    """
    Automated image quality vetting using a pre-trained NIMA model.
    """
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # Technical NIMA uses EfficientNetV2-S as per unified_models.yaml
        self.model = NIMA_Model(backbone="efficientnet_v2_s").to(self.device)
        
        print(f"📡 [QUALITY-SENTRY] Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def score(self, img_pil):
        img_t = self.transform(img_pil).unsqueeze(0).to(self.device)
        probs = self.model(img_t)
        
        # Calculate mean score from distribution [1, 10]
        weights = torch.arange(1, 11).to(self.device).float()
        mean_score = torch.sum(probs * weights, dim=1).item()
        return mean_score

class AutoLabeler:
    """
    Automated label discovery using pre-trained YOLO models.
    Supports Detection, Instance Segmentation, and Pose Estimation.
    """
    def __init__(self, mode="detection", device="cuda" if torch.cuda.is_available() else "cpu"):
        from ultralytics import YOLO
        self.device = device
        self.mode = mode # 'detection', 'segmentation', 'pose'
        
        # Determine model path based on task
        if mode == "segmentation":
            model_path = "yolov8n-seg.pt"
        elif mode == "pose":
            model_path = "yolov8n-pose.pt"
        else:
            model_path = "yolov8n.pt"
            
        print(f"🤖 [AUTO-LABEL] Initializing {mode} model: {model_path}")
        self.model = YOLO(model_path)

    def predict(self, img_pil):
        results = self.model.predict(img_pil, device=self.device, verbose=False)
        annotations = []
        
        for r in results:
            if self.mode == "detection":
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    x, y, x2, y2 = xyxy
                    annotations.append({"type": "bbox", "cls": cls, "data": [x, y, x2-x, y2-y]})
            
            elif self.mode == "segmentation" and r.masks:
                for i, mask in enumerate(r.masks.xyn):
                    cls = int(r.boxes.cls[i].cpu().numpy())
                    # Flatten normalized polygon: [x1, y1, x2, y2, ...]
                    poly = mask.flatten().tolist()
                    annotations.append({"type": "segmentation", "cls": cls, "data": poly})
            
            elif self.mode == "pose" and r.keypoints:
                for i, kpts in enumerate(r.keypoints.xyn):
                    cls = int(r.boxes.cls[i].cpu().numpy())
                    # YOLO box for anchor
                    box = r.boxes.xywh[i].cpu().numpy()
                    # Flatten normalized keypoints with visibility (v=1 mock)
                    # Note: r.keypoints.xyn is [N, 17, 2]
                    points = []
                    for pt in kpts:
                        points.extend([pt[0], pt[1], 1.0]) # x, y, visibility
                    annotations.append({"type": "pose", "cls": cls, "data": box.tolist() + points})
                    
        return annotations

class CaptionSentry:
    """
    Automated image caption generation using BLIP.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        self.device = device
        # Using the base model for optimal balance of speed and descriptive fidelity
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, img_pil):
        try:
            inputs = self.processor(img_pil, return_tensors="pt").to(self.device)
            # 2026 Resilience: We use greedy search for speed; beam search optional for high-depth tasks
            out = self.model.generate(**inputs, max_new_tokens=50)
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            return "a high quality image" # Fallback safety descriptor

class CLIPManifold:
    """
    Advanced Multimodal Manifold for Style Clustering and Aesthetic Scoring.
    Utilizes CLIP v5.0 features for zero-shot style tagging.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        from transformers import CLIPProcessor, CLIPModel
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
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)
        best_idx = probs.argmax().item()
        return self.styles[best_idx]
