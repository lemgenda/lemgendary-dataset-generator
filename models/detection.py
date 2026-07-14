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
        
        import os
        base_dir = os.path.dirname(__file__)
        
        # Determine model path based on task
        if mode == "segmentation":
            model_path = os.path.join(base_dir, "yolov8n-seg.pt")
        elif mode == "pose":
            model_path = os.path.join(base_dir, "yolov8n-pose.pt")
        else:
            model_path = os.path.join(base_dir, "yolov8n.pt")
            
        # print(f"🤖 [AUTO-LABEL] Initializing {mode} model: {model_path}")
        self.model = YOLO(model_path)

    def predict(self, img_pil):
        if self.mode == "face_landmarks":
            return self._predict_face_landmarks(img_pil)
            
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

    def _predict_face_landmarks(self, img_pil):
        import numpy as np
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import os
        
        # Initialize detector lazily
        if not hasattr(self, 'face_detector'):
            base_dir = os.path.dirname(__file__)
            # Point to the downloaded face_landmarker.task
            model_path = os.path.join(base_dir, "..", "..", "lemgendary-training-suite", "face_landmarker.task")
            if not os.path.exists(model_path):
                # Fallback to current dir if needed
                model_path = os.path.join(base_dir, "face_landmarker.task")
                
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(base_options=base_options,
                                                output_face_blendshapes=False,
                                                output_facial_transformation_matrixes=False,
                                                num_faces=1)
            self.face_detector = vision.FaceLandmarker.create_from_options(options)
            
        # Convert PIL to mp.Image
        import cv2
        img_np = np.array(img_pil)
        if img_pil.mode == 'RGB':
            # mediapipe expects RGB, np.array from RGB PIL is RGB
            pass
        elif img_pil.mode == 'RGBA':
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        detection_result = self.face_detector.detect(mp_image)
        
        annotations = []
        if detection_result and detection_result.face_landmarks:
            for landmarks in detection_result.face_landmarks:
                h, w = img_np.shape[:2]
                
                # Derive bounding box from landmarks min/max
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                # Add some padding to the bounding box
                pad_x = (x_max - x_min) * 0.1
                pad_y = (y_max - y_min) * 0.1
                x_min = max(0, x_min - pad_x)
                y_min = max(0, y_min - pad_y)
                x_max = min(w, x_max + pad_x)
                y_max = min(h, y_max + pad_y)
                
                # RetinaFace landmarks
                lx, ly = landmarks[468].x, landmarks[468].y
                rx, ry = landmarks[473].x, landmarks[473].y
                nx, ny = landmarks[1].x, landmarks[1].y
                lmx, lmy = landmarks[61].x, landmarks[61].y
                rmx, rmy = landmarks[291].x, landmarks[291].y
                
                kpts = [lx, ly, rx, ry, nx, ny, lmx, lmy, rmx, rmy]
                
                cls = 0 # Face class
                # Note: bbox data here is x, y, width, height (absolute pixels)
                # AutoLabeler returns absolute pixels for bbox in YOLO output format, wait!
                # Actually, in 'detection' mode above, x_min, y_min, w, h are absolute?
                # No! `box.xyxy[0]` is absolute.
                annotations.append({"type": "pose", "cls": cls, "data": [x_min, y_min, x_max-x_min, y_max-y_min] + kpts})
                
        return annotations
