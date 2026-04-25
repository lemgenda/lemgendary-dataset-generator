import torch
from ultralytics import YOLO

class AutoLabeler:
    """
    Automated label discovery using pre-trained YOLO models.
    Supports Detection, Instance Segmentation, and Pose Estimation.
    """
    def __init__(self, mode="detection", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.mode = mode # 'detection', 'segmentation', 'pose'
        
        # Determine model path based on task
        if mode == "segmentation":
            model_path = "yolov8n-seg.pt"
        elif mode == "pose":
            model_path = "yolov8n-pose.pt"
        else:
            model_path = "yolov8n.pt"
            
        # print(f"🤖 [AUTO-LABEL] Initializing {mode} model: {model_path}")
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
                    poly = mask.flatten().tolist()
                    annotations.append({"type": "segmentation", "cls": cls, "data": poly})
            
            elif self.mode == "pose" and r.keypoints:
                for i, kpts in enumerate(r.keypoints.xyn):
                    cls = int(r.boxes.cls[i].cpu().numpy())
                    box = r.boxes.xywh[i].cpu().numpy()
                    points = []
                    for pt in kpts:
                        points.extend([pt[0], pt[1], 1.0])
                    annotations.append({"type": "pose", "cls": cls, "data": box.tolist() + points})
                    
        return annotations
