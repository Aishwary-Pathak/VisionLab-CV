"""
Configuration module for VisionLab CV - CCA 3 Computer Vision Assignment.
Defines project paths, parameters, model weights, and visualization defaults.
"""

from pathlib import Path
import cv2

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Data and report directories
DATA_DIR = BASE_DIR / "data"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"

TASK1_OUT = OUTPUTS_DIR / "task1_restoration_segmentation"
TASK2_OUT = OUTPUTS_DIR / "task2_feature_matching"
TASK3_OUT = OUTPUTS_DIR / "task3_object_recognition"
TASK4_OUT = OUTPUTS_DIR / "task4_motion_estimation"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
for path in [
    INPUTS_DIR,
    OUTPUTS_DIR,
    TASK1_OUT,
    TASK2_OUT,
    TASK3_OUT,
    TASK4_OUT,
    REPORTS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Default Model Paths & Names
DEFAULT_YOLO_MODEL = "yolov8n.pt"  # Lightweight nano model, fast on CPU
HAAR_CASCADE_FACE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
HAAR_CASCADE_EYE = cv2.data.haarcascades + "haarcascade_eye.xml"

# Task 1 Defaults
DEFAULT_GAUSSIAN_SIGMA = 25.0
DEFAULT_SP_NOISE_RATIO = 0.05
DEFAULT_MEDIAN_KERNEL = 5
DEFAULT_GAUSSIAN_KERNEL = 5

# Task 2 Defaults
DEFAULT_SIFT_FEATURES = 1000
DEFAULT_LOWE_RATIO = 0.75
DEFAULT_RANSAC_THRESH = 5.0

# Task 4 Defaults
DEFAULT_MAX_VIDEO_FRAMES = 120
DEFAULT_SAMPLING_RATE = 2  # Process every 2nd frame for speed
DEFAULT_LK_MAX_CORNERS = 150
DEFAULT_LK_QUALITY_LEVEL = 0.03
DEFAULT_LK_MIN_DISTANCE = 7
DEFAULT_LK_BLOCK_SIZE = 7

# UI Theme & Branding
APP_TITLE = "VisionLab CV"
APP_SUBTITLE = "CCA 3 Computer Vision Practical Project (Units 3, 4, 5)"
APP_ICON = "🔬"

