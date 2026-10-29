"""
Task 3: Object Recognition & Comparison.
Implements:
1. Viola-Jones Classical Detection: Haar feature-based Cascade Classifier for frontal faces/eyes.
2. Modern Deep Learning Detection: Ultralytics YOLOv8 for 80 COCO object classes.
3. Comparative Benchmarking: Side-by-side evaluation of detection counts, classes, confidence scores,
   inference latency (runtime), and algorithmic trade-offs without unsupported ground truth claims.
"""

from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from modules.metrics import BenchmarkTimer
from modules.segmentation import to_grayscale
import config


class HaarDetector:
    """Viola-Jones Haar Cascade Detector wrapper."""
    def __init__(self, cascade_path: Optional[str] = None):
        self.cascade_path = cascade_path or config.HAAR_CASCADE_FACE
        self.classifier = cv2.CascadeClassifier(self.cascade_path)
        if self.classifier.empty():
            # Fallback to direct OpenCV data path
            self.cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.classifier = cv2.CascadeClassifier(self.cascade_path)

    @property
    def model_name(self) -> str:
        return Path(self.cascade_path).name

    def detect(
        self,
        image: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30)
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
        """
        Detect target objects (e.g. faces) using Haar Features and Adaboost Cascade.
        Returns:
            (annotated_rgb_image, detections_list, runtime_ms)
        """
        gray = to_grayscale(image)
        annotated = image.copy()

        with BenchmarkTimer("Haar Cascade Detection") as timer:
            rects = self.classifier.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

        detections = []
        for i, (x, y, w, h) in enumerate(rects):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red box for Haar
            cv2.putText(
                annotated,
                f"Face #{i+1}",
                (x, max(15, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
            detections.append({
                "id": i + 1,
                "class": "face",
                "confidence": "N/A (Heuristic Cascade)",
                "bbox": [int(x), int(y), int(w), int(h)]
            })

        return annotated, detections, timer.elapsed_ms


class YOLODetector:
    """Ultralytics YOLOv8 Deep Learning Object Detector wrapper."""
    def __init__(self, model_name: str = config.DEFAULT_YOLO_MODEL):
        self.model_name = model_name
        self.model = None
        self._load_error = None
        self._init_model()

    def _init_model(self):
        try:
            from ultralytics import YOLO
            # Ultralytics will load local weights if present, or download yolov8n.pt on first run
            self.model = YOLO(self.model_name)
        except Exception as e:
            self._load_error = str(e)
            self.model = None

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], float, Optional[str]]:
        """
        Detect multiple COCO classes using YOLOv8 CNN.
        Returns:
            (annotated_rgb_image, detections_list, runtime_ms, error_message)
        """
        if not self.is_available:
            return image.copy(), [], 0.0, f"YOLO model not loaded: {self._load_error}"

        annotated = image.copy()
        detections = []

        try:
            with BenchmarkTimer("YOLO Inference") as timer:
                # Run inference on RGB array
                results = self.model.predict(
                    source=image,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                    device="cpu"
                )

            res = results[0]
            boxes = res.boxes

            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names.get(cls_id, f"Class {cls_id}")

                x1, y1, x2, y2 = xyxy
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for YOLO
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

                detections.append({
                    "id": i + 1,
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                })

            return annotated, detections, timer.elapsed_ms, None

        except Exception as e:
            return image.copy(), [], 0.0, f"YOLO inference error: {str(e)}"


def run_object_detection_comparison(
    image: np.ndarray,
    haar_scale_factor: float = 1.1,
    haar_min_neighbors: int = 5,
    yolo_model_name: str = "yolov8n.pt",
    yolo_conf: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """
    Execute both Viola-Jones and YOLO on the exact same input image.
    Generates side-by-side detection visuals and a factual comparison table.
    """
    # 1. Viola-Jones Haar
    haar = HaarDetector()
    vis_haar, haar_dets, haar_time = haar.detect(
        image, scale_factor=haar_scale_factor, min_neighbors=haar_min_neighbors
    )

    # 2. YOLO
    yolo = YOLODetector(model_name=yolo_model_name)
    vis_yolo, yolo_dets, yolo_time, yolo_err = yolo.detect(image, conf_threshold=yolo_conf)

    # Summarize detections
    haar_classes = [d["class"] for d in haar_dets]
    yolo_classes = [d["class"] for d in yolo_dets]

    haar_conf_summary = "N/A (Rule cascade)"
    if yolo_dets:
        confs = [d["confidence"] for d in yolo_dets if isinstance(d["confidence"], (int, float))]
        yolo_conf_summary = f"{np.mean(confs):.2f} (avg)" if confs else "N/A"
    else:
        yolo_conf_summary = "None detected"

    records = [
        {
            "Framework": "Viola-Jones (OpenCV Haar Cascade)",
            "Target Classes": "Pre-trained Frontal Faces Only",
            "Detections Count": len(haar_dets),
            "Classes Observed": ", ".join(set(haar_classes)) if haar_classes else "None",
            "Confidence Info": haar_conf_summary,
            "Latency (ms)": round(haar_time, 2),
            "Architecture Type": "Integral Image + AdaBoost Cascade (Handcrafted Features)"
        },
        {
            "Framework": f"Ultralytics YOLO ({yolo_model_name})",
            "Target Classes": "80 COCO Generic Object Classes",
            "Detections Count": len(yolo_dets),
            "Classes Observed": ", ".join(set(yolo_classes)) if yolo_classes else "None",
            "Confidence Info": yolo_conf_summary,
            "Latency (ms)": round(yolo_time, 2),
            "Architecture Type": "Deep Convolutional Neural Network (Single-Stage Anchor-Free)"
        }
    ]

    df_comp = pd.DataFrame(records)

    analysis = {
        "haar_count": len(haar_dets),
        "yolo_count": len(yolo_dets),
        "haar_time": round(haar_time, 2),
        "yolo_time": round(yolo_time, 2),
        "yolo_error": yolo_err,
        "yolo_classes": list(set(yolo_classes)),
        "haar_classifier_used": haar.model_name
    }

    return vis_haar, vis_yolo, df_comp, analysis

