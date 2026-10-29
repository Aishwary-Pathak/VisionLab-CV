# VisionLab CV – CCA 3 Computer Vision Practical Project

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0--contrib-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-red.svg)](https://streamlit.io/)
[![Unit Tests](https://img.shields.io/badge/pytest-22%20passed-brightgreen.svg)](https://docs.pytest.org/)

An academic, interactive, end-to-end Computer Vision engineering dashboard implementing all four required tasks from the **CCA 3 Computer Vision curriculum (Units 3, 4, and 5)** using student personal media datasets.

---

## 🌟 Key Features & Academic Tasks

### 🧪 Task 1: Image Restoration & Segmentation (Unit 3)
- **Stochastic Noise Injection:** Configurable Gaussian noise ($\mu, \sigma$) and impulsive Salt-and-Pepper noise ($p, \text{ratio}$).
- **Spatial Restoration Filters:** Non-linear Median rank filter ($k=3, 5, 7, 9$) vs Linear Gaussian smoothing filter.
- **Fair Quantitative Evaluation:** Direct PSNR (dB), SSIM, and MSE calculated against clean ground-truth reference images.
- **Mathematical Morphology:** Erosion, dilation, opening, closing, and gradient with customizable structuring elements (Rect, Ellipse, Cross).
- **Multi-stage Canny Edge Detection:** Non-maximum suppression and hysteresis thresholding.
- **Marker-Controlled Watershed Segmentation:** Automated Otsu thresholding, Euclidean Distance Transform, topological sure foreground markers, and immersion boundary overlays.

### 🔍 Task 2: Feature Extraction & Matching (Unit 4)
- **Histogram of Oriented Gradients (HOG):** 9-bin gradient histograms across dense $16 \times 16$ spatial cells with visual field rendering.
- **SIFT Scale-Space Invariance:** Difference-of-Gaussian extrema localization and 128-D orientation descriptors.
- **Feature Matching & Geometric Verification:** k-NN matching with Lowe's ratio test ($d_1 < 0.75 \cdot d_2$) and projective Homography estimation via RANSAC.
- **Stress-Test Scenarios:** Side-by-side matching evaluated on 45° rotation and 0.6x downscaling with RANSAC inlier percentage tracking.

### 🎯 Task 3: Object Recognition (Unit 5)
- **Viola–Jones Classical Architecture:** Handcrafted Haar features with Integral Images and AdaBoost attentional cascade for frontal faces.
- **Ultralytics YOLOv8 Deep Learning:** Single-stage anchor-free convolutional neural network trained on 80 COCO object classes.
- **Factual Benchmarking:** Side-by-side visualization, detection counts, confidence distributions, and latency (ms) profiling.

### 🎥 Task 4: Video Motion Estimation (Unit 5)
- **Video Stream Ingestion:** Temporal frame extraction and subsampling of self-recorded student video footage.
- **Consecutive Frame Differencing:** Ultra-fast temporal silhouette detection ($|I_t - I_{t-1}| > T$).
- **GMM Background Subtraction (MOG2):** Statistical Gaussian Mixture Modelling with dynamic background updates and shadow identification.
- **Lucas–Kanade Sparse Optical Flow:** Shi-Tomasi corner detection with pyramidal tracking and continuous motion trajectory trails.
- **Comparative Evaluation:** Structured comparison table assessing FPS, shadow handling, and stationary object response.

---

## 📁 Project Directory Structure

```text
VisionLab-CV/
├── app.py                     # Interactive Streamlit Web Dashboard
├── config.py                  # Project paths, default parameters, and constants
├── conftest.py                # Pytest configuration and root discovery
├── requirements.txt           # Verified, conflict-free dependency specification
├── run_experiments.py         # Batch CLI script to process active dataset
├── README.md                  # Comprehensive documentation and setup guide
├── .gitignore                 # Excludes caches, large weights, and raw videos
│
├── modules/                   # Core computer vision processing modules
│   ├── __init__.py
│   ├── restoration.py         # Task 1A: Noise injection, Median & Gaussian filters
│   ├── segmentation.py        # Task 1B: Morphology, Canny & Watershed
│   ├── feature_extraction.py  # Task 2: HOG, SIFT & RANSAC Homography
│   ├── object_detection.py    # Task 3: Viola-Jones & YOLOv8
│   ├── motion_estimation.py   # Task 4: Differencing, MOG2 & Lucas-Kanade Flow
│   ├── metrics.py             # PSNR, SSIM, MSE & BenchmarkTimer
│   └── visualization.py       # Comparative figure generators
│
├── utils/                     # Input validation and safe file management
│   ├── __init__.py
│   ├── validation.py          # Validates resolution, channels, FPS, and formats
│   └── file_utils.py          # Non-destructive file storage and CSV export
│
├── tests/                     # Automated test suite (22 unit tests)
│   ├── test_restoration.py
│   ├── test_segmentation.py
│   ├── test_feature_extraction.py
│   ├── test_object_detection.py
│   └── test_motion_estimation.py
│
├── data/
│   ├── inputs/                # Immutable original student media
│   │   ├── user_mountain_village.jpg
│   │   └── user_video.mp4
│   └── outputs/               # Generated processed masks, plots, and CSVs
│       ├── task1_restoration_segmentation/
│       ├── task2_feature_matching/
│       ├── task3_object_recognition/
│       └── task4_motion_estimation/
│
└── reports/                   # Academic submission reports and viva guide
    ├── TECHNICAL_REPORT.md    # Complete formal technical report with metrics
    ├── VIVA_PRESENTATION_GUIDE.md  # 5–8 minute oral examination presentation script
    ├── SUBMISSION_CHECKLIST.md     # Google Drive packaging and submission steps
    └── *.csv                  # Aggregated benchmark metric logs
```

---

## 🚀 Setup & Execution Instructions

### 1. Prerequisites
- **Operating System:** macOS (Apple Silicon or Intel) or Linux.
- **Python Version:** Python 3.10, 3.11, or 3.12 (Python 3.12 recommended).

### 2. Create Virtual Environment & Install Dependencies
Open Terminal, navigate to the project directory, and execute:

```bash
# 1. Navigate to project root
cd "/Users/mac/Desktop/CCA-3 C.V"

# 2. Create Python virtual environment
python3.12 -m venv venv

# 3. Activate the environment
source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on OpenCV:** The project requires `opencv-contrib-python-headless`. Do not install `opencv-python` alongside it, as it will cause duplicate namespace conflicts with SIFT and MOG2.

---

### 3. Run Automated Unit Tests
To verify all 22 algorithms before launching the UI:

```bash
source venv/bin/activate
pytest tests/
```
*Expected Output: `22 passed in 4.02s`*

---

### 4. Run Automated Batch Pipeline (Optional)
To execute the complete benchmark pipeline on your active dataset from the command line:

```bash
source venv/bin/activate
python run_experiments.py
```
This generates all processed images, watershed masks, match lines, and CSV metric files in `data/outputs/` and `reports/`.

---

### 5. Launch the Streamlit Dashboard
To launch the interactive dashboard:

```bash
source venv/bin/activate
streamlit run app.py
```
The browser will automatically open to `http://localhost:8501`.

---

## 📊 Summary of Experimental Benchmarks

All metrics below were computed on the genuine student dataset (`user_mountain_village.jpg` and `user_video.mp4`):

| Task | Experiment Scenario | Key Metric / Result | Theoretical Observation |
| :--- | :--- | :--- | :--- |
| **Task 1** | 5% Salt & Pepper Noise | **Median:** PSNR 25.11 dB, SSIM 0.7171<br>**Gaussian:** PSNR 25.01 dB, SSIM 0.6784 | Median filter preserves high-frequency edges by replacing impulse outliers with ranked medians. |
| **Task 2** | SIFT 45° Rotation | **349 RANSAC Inliers (98.6%)** from 354 Lowe's matches | SIFT achieves near-complete rotational invariance via canonical orientation alignment. |
| **Task 2** | SIFT 0.6x Downscaling | **355 RANSAC Inliers (96.7%)** from 367 Lowe's matches | Scale invariance is guaranteed through scale-space Difference-of-Gaussian (DoG) octaves. |
| **Task 3** | Viola-Jones vs YOLOv8 | **Viola-Jones:** 0 faces (166 ms)<br>**YOLOv8:** 1 umbrella (938 ms) | Viola-Jones strictly detects frontal human faces; YOLO recognizes generic COCO scene structures. |
| **Task 4** | Video Motion Estimation | **Differencing:** 185 FPS<br>**MOG2:** 16.7 FPS<br>**Lucas-Kanade:** 189 FPS | Differencing & MOG2 yield spatial masks; Lucas-Kanade yields explicit 2D motion velocity vectors $(u, v)$. |

---

## 🔒 Data Integrity & Ethics Statement
This project adheres to strict academic evaluation standards:
- **No Overwriting:** Source photographs and videos are never modified.
- **Genuine Benchmarks:** All metrics, PSNR values, detection counts, and runtimes reflect real execution data without fabrication.
- **Offline Fallbacks:** Viola-Jones and classical algorithms run completely offline without internet dependencies.

---

## 🎓 Academic Viva & Evaluation
Refer to:
- [`reports/VIVA_PRESENTATION_GUIDE.md`](file:///Users/mac/Desktop/CCA-3%20C.V/reports/VIVA_PRESENTATION_GUIDE.md) for the exact 5–8 minute demonstration script.
- [`reports/TECHNICAL_REPORT.md`](file:///Users/mac/Desktop/CCA-3%20C.V/reports/TECHNICAL_REPORT.md) for the formal written report.
- [`reports/SUBMISSION_CHECKLIST.md`](file:///Users/mac/Desktop/CCA-3%20C.V/reports/SUBMISSION_CHECKLIST.md) for Google Drive submission formatting.

