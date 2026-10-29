# VisionLab CV: Viva Voce & Faculty Demonstration Guide

**Project:** VisionLab CV – CCA 3 Computer Vision Practical Project  
**Target Duration:** 5 to 8 Minutes  
**Audience:** Computer Vision Evaluator / Professor / External Examiner  

---

## 🎯 Viva Strategy & Presentation Mindset
- **Be Confident & Direct:** Start immediately with what you implemented and show running results.
- **Emphasize Theory + Practical Rigor:** Explain *why* algorithms behave the way they do (e.g. why Median filter beats Gaussian for impulse noise).
- **Show Genuine Outputs:** Highlight that all experiments ran on your own photograph and video, with strict data integrity (non-destructive pipeline).

---

## ⏱️ Step-by-Step Chronological Speaking Script

### Minute 0:00 – 1:00 | Project Overview & Dataset Verification
> **What to Say:**  
> *"Good morning/afternoon Professor. Today I am presenting **VisionLab CV**, my practical implementation for the CCA 3 Computer Vision assignment covering Units 3, 4, and 5.*  
> *For this project, I used my own original dataset: a high-resolution photograph taken in a mountain village, and a self-recorded 16-second Full HD video.*  
> *All algorithms run through a modular, reproducible Python architecture built on OpenCV, NumPy, scikit-image, and PyTorch/Ultralytics, integrated into an interactive Streamlit engineering dashboard.*  
> *Importantly, the pipeline is strictly non-destructive: all source photographs and videos are preserved in `data/inputs/`, while all processed masks, descriptors, and benchmark CSVs are stored systematically in `data/outputs/`."*

**Action on Screen:**
- Open the dashboard at `http://localhost:8501`.
- Click on **📁 Dataset Manager & Validation**.
- Show the detected image metadata ($1024 \times 682$, 3 channels) and video metadata ($1920 \times 1080$, $30\text{ FPS}$).

---

### Minute 1:00 – 2:30 | Task 1: Image Restoration & Segmentation
> **What to Say:**  
> *"In Task 1, I implemented image restoration under two different stochastic degradation models: Gaussian noise and Salt-and-Pepper impulsive noise.*  
> *When we evaluate restoration against the clean reference image, our quantitative metrics clearly validate computer vision theory:*  
> *On 5% Salt-and-Pepper noise, the **Median filter** achieves a PSNR of **25.11 dB** and an SSIM of **0.7171**, significantly outperforming the Gaussian filter's SSIM of **0.6784**.*  
> *This is because the Median filter is an order-statistic rank filter that substitutes outliers with local medians without blurring edge gradients.*  
> *Next, in segmentation, I implemented mathematical morphology—erosion, dilation, opening, and closing—along with Canny edge detection.*  
> *Finally, I implemented **Marker-Controlled Watershed Segmentation**: we compute Otsu thresholding, morphological opening for noise removal, and calculate the Euclidean Distance Transform. By thresholding the distance peaks at 45%, we obtain sure foreground markers and flood the gradient basins to separate touching regions."*

**Action on Screen:**
- Navigate to **🧪 Task 1: Restoration & Segmentation**.
- Show the 4-panel comparison figure (Clean, Noisy, Median, Gaussian) and the metric table.
- Flip to the **Watershed tab** and show the distance transform (Jet colormap) and boundary overlays.

---

### Minute 2:30 – 4:00 | Task 2: Feature Extraction & Matching (HOG & SIFT)
> **What to Say:**  
> *"In Task 2, we explore local and global feature representations.*  
> *First, I implemented **Histogram of Oriented Gradients (HOG)** with 9 angular orientation bins across $16 \times 16$ spatial cells. The visualization demonstrates how HOG captures structural shape boundaries invariant to illumination.*  
> *Second, I implemented **Scale-Invariant Feature Transform (SIFT)**, which identifies scale-space extrema across Difference-of-Gaussian octaves and computes 128-dimensional orientation descriptors.*  
> *To empirically prove SIFT's invariance, I established two geometric stress tests:*  
> *1. When rotating the photograph by **45 degrees**, Lowe's ratio test identified 354 good matches, and **RANSAC Homography verification confirmed 349 inliers**, yielding a **98.6% inlier ratio**.*  
> *2. When downscaling the photograph to **0.6x**, SIFT achieved a **96.7% RANSAC inlier ratio**.*  
> *This rigorously proves SIFT's scale and rotational invariance."*

**Action on Screen:**
- Navigate to **🔍 Task 2: Features & Matching**.
- Show HOG gradient field visualization.
- Under **Invariance Experiments**, display the side-by-side match lines with green RANSAC inliers for the 45° rotation test.

---

### Minute 4:00 – 5:30 | Task 3: Object Recognition (Viola–Jones vs YOLO)
> **What to Say:**  
> *"Task 3 provides a head-to-head empirical comparison between classical computer vision and modern deep learning.*  
> *I benchmarked **Viola–Jones Haar Cascade** against **Ultralytics YOLOv8n** on the exact same input photograph.*  
> *Viola–Jones uses rectangular Haar features computed in constant time via integral images with an AdaBoost attentional cascade. Because the cascade is specialized strictly for frontal human faces, it correctly yielded 0 detections on our mountain landscape photograph.*  
> *In contrast, YOLOv8n is an end-to-end deep convolutional neural network trained on COCO's 80 generic object classes. It successfully recognized the rooftop canopy structure in the village as an `umbrella` with a 0.52 confidence score.*  
> *In terms of latency, Viola–Jones executed in **166 ms**, while YOLOv8n required **938 ms** on CPU.*  
> *Scientifically, we do not claim one is arbitrarily 'more accurate' overall without task-specific ground truth annotations; rather, they serve completely different operational regimes: specialized fast face scanning vs open-vocabulary multi-class scene understanding."*

**Action on Screen:**
- Navigate to **🎯 Task 3: Object Recognition**.
- Show the side-by-side detection visual and the quantitative comparison table.

---

### Minute 5:30 – 7:00 | Task 4: Video Motion Estimation & Optical Flow
> **What to Say:**  
> *"Task 4 examines temporal motion estimation across consecutive frames of my self-recorded video.*  
> *I implemented and compared three distinct motion paradigms:*  
> *1. **Frame Differencing:** Calculates absolute pixel difference $|I_t - I_{t-1}| > T$. It is ultra-fast at **185 FPS**, but it cannot distinguish object motion from camera jitter, and stopped objects disappear immediately.*  
> *2. **GMM Background Subtraction (MOG2):** Models each pixel's history using Gaussian mixtures. It adaptively updates background parameters and isolates shadows (marked in blue) from true moving foreground (marked in green).*  
> *3. **Lucas–Kanade Pyramidal Optical Flow:** Tracks Shi-Tomasi corner features across spatial intensity pyramids, generating explicit 2D motion velocity vectors $(u, v)$.*  
> *As a critical engineering distinction: Differencing and MOG2 produce spatial binary masks telling us **where** intensity changed, whereas Lucas–Kanade produces vector fields telling us **direction and speed** of movement."*

**Action on Screen:**
- Navigate to **🎥 Task 4: Video Motion Estimation**.
- Move the frame slider to display the 4-quadrant view (Original, Differencing, MOG2 shadow overlay, Lucas-Kanade flow trails).
- Highlight the 3-way results table.

---

### Minute 7:00 – 8:00 | Conclusion, Code Quality & Q&A
> **What to Say:**  
> *"To ensure academic reproducibility, I have included a full test suite with 22 automated unit tests passing cleanly in `pytest`, modularized code across `modules/`, and complete CSV export capabilities.*  
> *Thank you Professor. I welcome any questions regarding the mathematical formulations or implementation details."*

---

## ❓ Frequently Asked Viva Questions & Model Answers

**Q1: Why does Median filter work so well on Salt-and-Pepper noise compared to Gaussian?**  
> *Answer:* Gaussian filtering computes a weighted arithmetic mean of local neighbors. If an extreme impulse pixel ($0$ or $255$) is present, it gets averaged into the output, blurring the defect across the neighborhood and softening edge gradients. In contrast, Median filtering sorts pixel intensities in the window and picks the middle value; since impulsive noise pixels are extreme outliers, they never occupy the median position and are cleanly discarded while sharp edges are preserved.

**Q2: What is the purpose of Lowe's Ratio Test in SIFT matching?**  
> *Answer:* For each descriptor, we find its two nearest neighbors in Euclidean descriptor space ($d_1$ and $d_2$). If $d_1 / d_2 < 0.75$, it means the best match is significantly closer than the second-best match, indicating high distinctiveness. If $d_1$ and $d_2$ are nearly equal, the point likely belongs to a repetitive texture (e.g., foliage or brick patterns) and is discarded to avoid false positive correspondences.

**Q3: What assumption does Lucas–Kanade optical flow make, and when does it fail?**  
> *Answer:* Lucas–Kanade relies on three core assumptions:  
> 1. Brightness constancy: the intensity of a moving surface patch remains invariant over short intervals ($I(x, y, t) = I(x+u, y+v, t+1)$).  
> 2. Small motion: displacements between consecutive frames are small enough for first-order Taylor series approximation.  
> 3. Spatial coherence: neighboring pixels belong to the same surface and share common motion.  
> It fails during abrupt illumination changes, specular reflections, or very rapid motions exceeding the pyramid search window.

**Q4: Why does Watershed require markers?**  
> *Answer:* Direct watershed transform treats image gradient magnitudes as topographic relief. In real-world images, subtle texture variations produce thousands of tiny local minima, causing severe over-segmentation. Marker-controlled watershed eliminates this by specifying explicit seed markers for foreground and background, flooding only from these verified topological seeds.

