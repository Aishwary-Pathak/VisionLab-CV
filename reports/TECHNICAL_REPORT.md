# VisionLab CV: CCA 3 Computer Vision Practical Technical Report

**Course Module:** CCA 3 – Computer Vision (Units 3, 4, 5)  
**Project Title:** VisionLab CV – Academic Evaluation & Algorithmic Benchmarking Dashboard  
**Author:** Student Submission  
**Date:** October 2026  
**Repository URL:** `[INSERT_YOUR_GITHUB_REPOSITORY_URL_HERE]`  
**Project Walkthrough Video:** `[INSERT_YOUR_5_MINUTE_DEMO_VIDEO_UNLISTED_YOUTUBE_LINK_HERE]`  

---

## 1. Problem Definition & Objectives

The primary objective of this practical assignment is to implement, evaluate, and comparatively benchmark fundamental and state-of-the-art computer vision techniques spanning Units 3, 4, and 5 of the curriculum:

1. **Unit 3 – Image Restoration & Segmentation:**
   - Model stochastic degradation processes (Additive White Gaussian Noise and Impulsive Salt-and-Pepper noise).
   - Implement spatial restoration filters (Linear Gaussian filter vs Non-linear Median filter) and rigorously evaluate reconstruction quality using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) against clean ground-truth references.
   - Implement mathematical morphology (erosion, dilation, opening, closing) and Canny edge detection.
   - Implement marker-controlled Watershed segmentation to partition touching or clustered regions via distance transforms.

2. **Unit 4 – Feature Extraction & Matching:**
   - Extract and visualize structural edge gradient distributions using Histogram of Oriented Gradients (HOG).
   - Detect scale-space extrema and compute 128-dimensional invariant descriptors using Scale-Invariant Feature Transform (SIFT).
   - Perform feature matching using k-Nearest Neighbors with Lowe's ratio test and geometric verification via RANSAC Homography under rotation and scale transformations.

3. **Unit 5 – Object Recognition:**
   - Compare classical Viola–Jones AdaBoost Haar Cascade detection against modern deep learning detection using Ultralytics YOLOv8.
   - Quantify inference latency, target class scope, confidence scores, and detection behavior on the same input photographs.

4. **Unit 5 – Video Motion Estimation:**
   - Ingest self-recorded real-world video footage and extract temporal frames.
   - Benchmark Consecutive Frame Differencing, GMM-based Background Subtraction (OpenCV MOG2), and Pyramidal Lucas–Kanade Sparse Optical Flow.
   - Discriminate between spatial foreground masks and directional motion velocity vectors.

---

## 2. Dataset Description & Integrity Guarantee

All practical experiments were executed on genuine student-captured media:
- **Primary Photograph:** `user_mountain_village.jpg`
  - Dimensions: $1024 \times 682$ pixels
  - Channels: 3 (RGB, 8-bit unsigned integer per channel)
  - File Size: $261.2\text{ KB}$
  - Scene Description: Natural outdoor mountain village featuring houses, slopes, trees, distant snowy peaks, and clear sky.
- **Video Footage:** `user_video.mp4`
  - Resolution: $1920 \times 1080$ pixels (Full HD)
  - Frame Rate: $30.0\text{ FPS}$
  - Duration: $16.8\text{ seconds}$ ($504\text{ total frames}$)
  - Scene Description: Handheld real-world video recording capturing natural movement and ambient lighting.
- **Data Integrity:** All source media are stored immutably in `data/inputs/` and are read in a strictly non-destructive manner. Generated experiment artifacts are isolated in `data/outputs/`.

---

## 3. Methodology & Algorithmic Formulation

### 3.1 Task 1: Restoration & Segmentation
#### Stochastic Noise Modelling
- **Gaussian Noise:** Added as $I_{\text{noisy}}(x, y) = \text{clip}(I(x, y) + \mathcal{N}(0, \sigma^2), 0, 255)$, simulating thermal sensor noise.
- **Salt-and-Pepper Noise:** Replaces a percentage $p$ of randomly sampled pixels with either $0$ (pepper) or $255$ (salt), simulating bit transmission faults.

#### Quantitative Quality Metrics
$$\text{MSE} = \frac{1}{M \cdot N \cdot C} \sum_{x=1}^M \sum_{y=1}^N \sum_{c=1}^C [I_{\text{ref}}(x, y, c) - I_{\text{test}}(x, y, c)]^2$$

$$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{255^2}{\text{MSE}} \right) \quad (\text{dB})$$

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

#### Marker-Controlled Watershed Segmentation
1. Convert RGB image to grayscale and compute binary threshold via Otsu's method.
2. Apply morphological opening ($k=3 \times 3$) to eliminate background speckle noise.
3. Compute morphological dilation to establish sure background ($\text{Sure}_{\text{BG}}$).
4. Compute Euclidean Distance Transform on binary foreground:
   $$D(p) = \min_{q \in \text{Background}} \|p - q\|_2$$
5. Threshold distance map at $0.45 \cdot \max(D)$ to isolate sure foreground cores ($\text{Sure}_{\text{FG}}$).
6. Obtain unknown border zone: $\text{Unknown} = \text{Sure}_{\text{BG}} - \text{Sure}_{\text{FG}}$.
7. Execute connected components labeling on sure foreground markers and run topological immersion flooding.

---

### 3.2 Task 2: Feature Extraction & Matching
#### HOG Descriptors
Computes 1-D gradient magnitude and orientation over $16 \times 16$ pixel spatial cells binned into 9 angular orientations ($0^\circ - 180^\circ$), normalized across $2 \times 2$ cell blocks using L2-norm.

#### SIFT & Robust Matching Pipeline
1. Multi-scale extrema localization using Difference-of-Gaussians (DoG):
   $$D(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) * I(x, y)$$
2. Sub-pixel keypoint refinement and rejection of low-contrast and edge-response keypoints.
3. Orientation assignment and 128-D gradient histogram descriptor computation.
4. k-NN matching with **Lowe's Ratio Distance Test**:
   $$\text{Keep match if } \frac{\|d_1 - d_{\text{query}}\|_2}{\|d_2 - d_{\text{query}}\|_2} < 0.75$$
5. Geometric verification via projective homography $\mathbf{x}' = \mathbf{H} \mathbf{x}$ using **RANSAC** (Random Sample Consensus) with a $5.0\text{ px}$ reprojection threshold.

---

### 3.3 Task 3: Object Recognition Comparison
- **Viola–Jones:** Utilizes integral images for $O(1)$ computation of rectangular Haar features, feature selection via AdaBoost, and an attentional cascade architecture specialized for frontal human faces.
- **Ultralytics YOLOv8n:** Modern anchor-free single-stage convolutional neural network trained on COCO (80 object classes), predicting bounding box coordinates and class probability distributions concurrently.

---

### 3.4 Task 4: Video Motion Estimation
- **Frame Differencing:** $\Delta I_t = |I_t - I_{t-1}| > T$.
- **GMM Background Subtraction (OpenCV MOG2):** Models each pixel's history using a mixture of $K$ Gaussian distributions, dynamically updating background parameters and isolating shadow pixels via chromaticity invariants.
- **Lucas–Kanade Sparse Optical Flow:** Solves the brightness constancy constraint over a local window centered on Shi-Tomasi corners:
  $$I_x u + I_y v = -I_t \implies \begin{bmatrix} u \\ v \end{bmatrix} = \left( \mathbf{A}^T \mathbf{A} \right)^{-1} \mathbf{A}^T \mathbf{b}$$

---

## 4. Experimental Results & Quantitative Tables

### 4.1 Task 1: Image Restoration Benchmarking
Experiments conducted on `user_mountain_village.jpg` degraded with 5.0% Salt-and-Pepper noise:

| Processing Stage | PSNR (dB) | SSIM | MSE | Execution Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Original Clean Reference** | **100.00** | **1.0000** | **0.00** | 0.00 |
| **Noisy (Salt & Pepper 5%)** | 18.26 | 0.4497 | 969.96 | 0.00 |
| **Median Filter ($5 \times 5$)** | **25.11** | **0.7171** | **200.50** | 5.55 |
| **Gaussian Filter ($5 \times 5, \sigma=1.2$)** | 25.01 | 0.6784 | 205.25 | 0.52 |

**Key Finding:** Median filtering achieved higher PSNR ($25.11\text{ dB}$ vs $25.01\text{ dB}$) and substantially superior structural preservation (SSIM $0.7171$ vs $0.6784$) because it eliminates impulsive outliers without low-pass blurring of sharp building edges.

---

### 4.2 Task 2: SIFT Invariance Benchmarking
Matching evaluated against geometric transformations of the clean reference:

| Experiment Scenario | Keypoints (Img 1) | Keypoints (Img 2) | Candidate Matches | Good Matches (Lowe's) | RANSAC Inliers | Inlier Ratio (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rotation ($45^\circ$)** | 1,000 | 1,000 | 1,000 | 354 | **349** | **98.6%** | 385.77 |
| **Scaling ($0.6\times$)** | 1,000 | 1,002 | 1,000 | 367 | **355** | **96.7%** | 210.34 |

**Key Finding:** SIFT demonstrated near-perfect geometric invariance under both rotation and substantial downscaling, with RANSAC inlier ratios exceeding $96\%$.

---

### 4.3 Task 3: Object Recognition Comparison

| Detector Framework | Target Classes | Detections | Classes Observed | Confidence Info | Latency (ms) | Architectural Paradigm |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Viola-Jones (OpenCV Haar)** | Frontal Human Faces | **0** | None | N/A (Rule Cascade) | 166.85 | Handcrafted Haar Features + AdaBoost |
| **Ultralytics YOLOv8n** | 80 COCO Generic Classes | **1** | umbrella | 0.52 (avg) | 938.59 | Deep CNN (Single-Stage Anchor-Free) |

**Factual Observation:** The input photograph is an outdoor mountain village without visible human faces; Viola-Jones correctly produced 0 false positives. YOLOv8 successfully localized the rooftop canopy shelter, classifying it as an `umbrella` with $0.52$ confidence.

---

### 4.4 Task 4: Video Motion Estimation Benchmarking

| Motion Estimation Method | Output Representation | Latency (ms/frame) | Estimated FPS | Stationary Object Handling | Shadow Handling | Camera Motion Sensitivity |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **Frame Differencing** | Binary Silhouette Mask | **5.39** | **185.7** | Fails (vanishes once stopped) | Poor (shadows appear as motion) | Severe (entire frame triggers) |
| **GMM MOG2** | Tri-state Mask (BG/Shadow/FG) | 60.01 | 16.7 | Adaptively absorbs into background | Robust (isolates shadow pixels) | High (requires static camera) |
| **Lucas–Kanade Optical Flow** | Sparse Velocity Vectors $(u, v)$ | **5.29** | **189.1** | Vectors drop to zero magnitude | Robust (anchors to textured corners) | Can isolate global camera ego-motion |

---

## 5. Technical Observations & Discussion

1. **Restoration Limits:** While linear Gaussian filtering is computationally fast ($0.52\text{ ms}$), it fails to restore impulsive noise because the arithmetic mean spreads extreme impulse values ($0$ and $255$) into neighboring pixels. Median filtering operates as an order-statistic filter, effectively filtering isolated impulses.
2. **Watershed Over-segmentation:** In natural landscape scenes with high texture density (trees, grass, mountain rock faces), standard watershed algorithms can produce over-segmentation unless marker priors are heavily constrained via morphological opening and aggressive distance transform thresholding.
3. **Motion Mask vs Flow Fields:** Frame differencing and MOG2 produce spatial masks revealing *where* intensity changes occur. They cannot infer object direction or velocity. Lucas–Kanade optical flow yields explicit velocity vectors, enabling trajectory tracking and visual odometry.

---

## 6. Limitations & Future Work

- **Lighting Variations in Optical Flow:** The Lucas–Kanade formulation rests on the Brightness Constancy Assumption ($I(x, y, t) = I(x+u, y+v, t+1)$). Sudden illumination shifts or specular glare violate this assumption, leading to tracking dropouts.
- **YOLO Hardware Acceleration:** On CPU execution, YOLOv8n requires $\approx 900\text{ ms}$ per megapixel frame. Utilizing Apple Silicon MPS (Metal Performance Shaders) or CUDA would yield real-time 30+ FPS deep inference.

---

## 7. Submission Checklist Verification

- [x] All 4 CCA 3 assignment tasks implemented and tested.
- [x] Genuine student photograph (`user_mountain_village.jpg`) and video (`user_video.mp4`) used.
- [x] Original dataset preserved without destructive modification.
- [x] Non-conflicting dependencies (`opencv-contrib-python-headless`) pinned in `requirements.txt`.
- [x] All 22 automated unit tests passing cleanly in `pytest`.
- [x] Interactive Streamlit application (`app.py`) complete with parameter controls and downloads.
- [x] Results saved in organized directory structure (`data/outputs/`).

