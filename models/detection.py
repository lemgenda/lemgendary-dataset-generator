"""YOLO and MediaPipe auto-labelers for detection, segmentation, pose and face landmarks."""

import os
from typing import Any, cast

import numpy as np
import torch
from ultralytics import YOLO

from core.lazy_imports import load_mediapipe


class AutoLabeler:
    """
    Automated label discovery using pre-trained YOLO models.
    Supports Detection, Instance Segmentation, and Pose Estimation.
    """
    def __init__(self, mode="detection", device=None):
        # None picks CUDA when it is available (resolved per instance, not at import).
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode # 'detection', 'segmentation', 'pose'
        self.face_detector = None  # lazily created by face_landmarks

        base_dir = os.path.dirname(__file__)

        # Determine model path based on task
        if mode == "segmentation":
            model_path = os.path.join(base_dir, "yolov8n-seg.pt")
        elif mode == "pose":
            model_path = os.path.join(base_dir, "yolov8n-pose.pt")
        else:
            model_path = os.path.join(base_dir, "yolov8n.pt")

        # print(f"[AUTO-LABEL] Initializing {mode} model: {model_path}")
        self.model = YOLO(model_path)

    def predict(self, img_pil):
        """Return the annotations (bbox, segmentation, pose or face landmarks) found in an image."""
        if self.mode == "face_landmarks":
            return self.face_landmarks(img_pil)

        raw_results = self.model.predict(img_pil, device=self.device, verbose=False)
        results: list[Any] = cast(list[Any], raw_results)
        extract = _MODE_EXTRACTORS.get(self.mode)
        annotations = []
        if extract is None:
            return annotations

        for r in results:
            annotations.extend(extract(r))

        return annotations

    def _face_detector_instance(self, python, vision):
        """Create the MediaPipe face landmarker on first use and return it."""
        if self.face_detector is not None:
            return self.face_detector
        base_dir = os.path.dirname(__file__)
        # Point to the downloaded face_landmarker.task
        model_path = os.path.join(
            base_dir, "..", "..", "lemgendary-training-suite", "face_landmarker.task"
        )
        if not os.path.exists(model_path):
            # Fallback to current dir if needed
            model_path = os.path.join(base_dir, "face_landmarker.task")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=False,
                                            output_facial_transformation_matrixes=False,
                                            num_faces=1)
        self.face_detector = vision.FaceLandmarker.create_from_options(options)
        return self.face_detector

    def face_landmarks(self, img_pil):
        """Face annotations (box plus landmarks) from MediaPipe, whatever the YOLO mode."""
        mp, python, vision = load_mediapipe()

        # Initialize detector lazily
        detector = self._face_detector_instance(python, vision)

        # Convert PIL to mp.Image (ensure 3-channel RGB)
        img_np = np.array(img_pil.convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        detection_result = detector.detect(mp_image)

        annotations = []
        if detection_result and detection_result.face_landmarks:
            h, w = img_np.shape[:2]
            for landmarks in detection_result.face_landmarks:
                annotations.append(_face_annotation(landmarks, w, h))

        return annotations


def _detection_annotations(r) -> list[dict[str, Any]]:
    """Bounding boxes ``[x, y, w, h]`` (absolute pixels) from one YOLO result."""
    if r.boxes is None:
        return []
    annotations = []
    for box in r.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        x, y, x2, y2 = xyxy
        annotations.append({"type": "bbox", "cls": cls, "data": [x, y, x2-x, y2-y]})
    return annotations


def _segmentation_annotations(r) -> list[dict[str, Any]]:
    """Normalised instance polygons from one YOLO result."""
    if r.masks is None or r.boxes is None:
        return []
    annotations = []
    for i, mask in enumerate(r.masks.xyn):
        cls = int(r.boxes.cls[i].cpu().numpy())
        poly = mask.flatten().tolist()
        annotations.append({"type": "segmentation", "cls": cls, "data": poly})
    return annotations


def _pose_annotations(r) -> list[dict[str, Any]]:
    """Box (xywh) plus normalised keypoints with visibility 1.0 from one YOLO result."""
    if r.keypoints is None or r.boxes is None:
        return []
    annotations = []
    for i, kpts in enumerate(r.keypoints.xyn):
        cls = int(r.boxes.cls[i].cpu().numpy())
        box = r.boxes.xywh[i].cpu().numpy()
        points = []
        for pt in kpts:
            points.extend([pt[0], pt[1], 1.0])
        annotations.append({"type": "pose", "cls": cls, "data": box.tolist() + points})
    return annotations


_MODE_EXTRACTORS = {
    "detection": _detection_annotations,
    "segmentation": _segmentation_annotations,
    "pose": _pose_annotations,
}

# MediaPipe face-mesh indices of the five RetinaFace landmarks:
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
_RETINAFACE_LANDMARKS = (468, 473, 1, 61, 291)


def _padded_face_box(landmarks, w, h) -> tuple[float, float, float, float]:
    """Landmark bounding box padded by 10% per side, clipped to the image."""
    # Derive bounding box from landmarks min/max
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add some padding to the bounding box
    pad_x = (x_max - x_min) * 0.1
    pad_y = (y_max - y_min) * 0.1
    return (
        max(0, x_min - pad_x),
        max(0, y_min - pad_y),
        min(w, x_max + pad_x),
        min(h, y_max + pad_y),
    )


def _face_annotation(landmarks, w, h) -> dict[str, Any]:
    """One face as a pose annotation: absolute ``[x, y, w, h]`` box plus 5 keypoints."""
    x_min, y_min, x_max, y_max = _padded_face_box(landmarks, w, h)

    # RetinaFace landmarks (normalised x, y pairs)
    kpts = []
    for index in _RETINAFACE_LANDMARKS:
        kpts.extend([landmarks[index].x, landmarks[index].y])

    cls = 0 # Face class
    # Note: bbox data here is x, y, width, height (absolute pixels), matching the
    # absolute ``box.xyxy`` values used in 'detection' mode.
    return {
        "type": "pose",
        "cls": cls,
        "data": [x_min, y_min, x_max - x_min, y_max - y_min] + kpts,
    }
