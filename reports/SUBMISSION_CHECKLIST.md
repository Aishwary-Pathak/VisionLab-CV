# CCA 3 Computer Vision Assignment – Submission Checklist & Instructions

## 📦 Deliverables Checklist

Before final academic submission, ensure all items below are completed and verified:

### Codebase & Implementation
- [x] **Project Directory Structure:** Clean, modular structure (`modules/`, `utils/`, `tests/`, `data/`, `reports/`).
- [x] **Task 1 Implementation:** Gaussian & Salt-and-Pepper noise, Median & Gaussian filters, PSNR/SSIM evaluation, Morphological operators, Canny edge detection, Marker-controlled Watershed.
- [x] **Task 2 Implementation:** HOG extraction, SIFT keypoints & descriptors, Lowe's ratio test, RANSAC Homography verification, Invariance experiments (rotation and scale).
- [x] **Task 3 Implementation:** Viola-Jones Haar Cascade face detector, Ultralytics YOLOv8 detector, comparative benchmark table and observations.
- [x] **Task 4 Implementation:** Video frame sampling, Frame Differencing, GMM MOG2 background subtraction with shadows, Lucas-Kanade optical flow, 3-way motion comparison table.
- [x] **Interactive Dashboard:** Complete Streamlit application (`app.py`) with parameter controls, visualizations, and downloads.
- [x] **Reproducibility Test Suite:** 22/22 unit tests passing in `pytest tests/`.
- [x] **Dependency Specification:** Non-conflicting `requirements.txt` with `opencv-contrib-python-headless`.

### Academic Documentation & Media
- [x] **Technical Report:** Complete report in `reports/TECHNICAL_REPORT.md` populated with real experimental figures and tables.
- [x] **Viva Guide:** 5–8 minute presentation script in `reports/VIVA_PRESENTATION_GUIDE.md`.
- [x] **Original Dataset:** Student photograph (`data/inputs/user_mountain_village.jpg`) and self-recorded video (`data/inputs/user_video.mp4`).
- [x] **Saved Results:** All task output images and CSV logs saved in `data/outputs/` and `reports/`.

---

## ☁️ Google Drive Submission & Archival Guide

Because raw video files and git repositories often have size limits, follow these instructions to package and upload your submission to Google Drive:

### 1. Folder Structure on Google Drive
Create a main folder named:  
`[Your_Roll_Number]_[Your_Name]_CCA3_Computer_Vision/`

Inside, organize the subfolders as follows:
```
[Your_Roll_Number]_[Your_Name]_CCA3_Computer_Vision/
├── 1_Source_Code/
│   ├── app.py
│   ├── config.py
│   ├── requirements.txt
│   ├── run_experiments.py
│   ├── modules/
│   ├── utils/
│   └── tests/
├── 2_Original_Dataset/
│   ├── user_mountain_village.jpg  (Clean high-resolution photograph)
│   └── user_video.mp4             (15-30 second original self-recorded video)
├── 3_Generated_Outputs/
│   ├── task1_restoration_segmentation/
│   ├── task2_feature_matching/
│   ├── task3_object_recognition/
│   └── task4_motion_estimation/
├── 4_Reports_and_Metrics/
│   ├── TECHNICAL_REPORT.md (or exported PDF)
│   ├── VIVA_PRESENTATION_GUIDE.md
│   └── *.csv (All quantitative metric files)
└── 5_Demo_Video/
    └── project_walkthrough.mp4 (Screen recording of Streamlit dashboard in action)
```

### 2. Google Drive Sharing Permissions
1. Right-click the root folder on Google Drive.
2. Select **Share** -> **General access**.
3. Set permission to **"Anyone with the link"** and role as **"Viewer"** (or specify your faculty's institutional email).
4. Copy the link and paste it into your university portal / submission document.

### 3. Creating a Clean Zip for LMS Submission
To create a clean archive of your project excluding large virtual environments or temporary caches:
```bash
cd "/Users/mac/Desktop/CCA-3 C.V"
zip -r visionlab_cv_submission.zip app.py config.py requirements.txt run_experiments.py conftest.py README.md modules/ utils/ tests/ reports/ data/ -x "venv/*" "__pycache__/*" "*.DS_Store" "*/.pytest_cache/*" "yolov8n.pt"
```
Upload this `visionlab_cv_submission.zip` to your course portal alongside your Google Drive link.

