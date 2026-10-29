"""
Unit tests for Task 3: Object Recognition (Viola-Jones Haar Cascade and YOLO).
"""

import numpy as np
import pytest
from modules.object_detection import HaarDetector, YOLODetector, run_object_detection_comparison


def test_haar_detector_instantiation_and_run():
    haar = HaarDetector()
    assert haar.model_name.endswith(".xml")

    # Run on synthetic image
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated, detections, latency = haar.detect(blank)
    assert annotated.shape == blank.shape
    assert isinstance(detections, list)
    assert latency >= 0


def test_yolo_detector_availability():
    yolo = YOLODetector()
    assert yolo.is_available or yolo._load_error is not None

    blank = np.zeros((128, 128, 3), dtype=np.uint8)
    annotated, detections, latency, err = yolo.detect(blank)
    assert annotated.shape == blank.shape
    assert isinstance(detections, list)
    assert latency >= 0


def test_comparison_runner():
    blank = np.zeros((128, 128, 3), dtype=np.uint8)
    vis_haar, vis_yolo, df, analysis = run_object_detection_comparison(blank)
    assert len(df) == 2
    assert "Viola-Jones (OpenCV Haar Cascade)" in df["Framework"].values
    assert "Ultralytics YOLO" in df["Framework"].iloc[1]
    assert "haar_count" in analysis

