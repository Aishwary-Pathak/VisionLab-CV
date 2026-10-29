"""
VisionLab CV – CCA 3 Computer Vision Assignment Dashboard.
A complete, academic, interactive Streamlit application covering Units 3, 4, and 5.
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import config
from utils.validation import validate_image_file, validate_video_file
from utils.file_utils import (
    save_bytes_to_file,
    copy_file_safe,
    list_files_by_extensions,
    save_image,
    export_dataframe_csv
)
from modules.metrics import evaluate_restoration_quality, BenchmarkTimer
from modules.restoration import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    apply_gaussian_filter,
    apply_median_filter,
    run_restoration_comparison
)
from modules.segmentation import (
    to_grayscale,
    apply_thresholding,
    apply_morphology,
    apply_canny_edge,
    apply_watershed_segmentation
)
from modules.feature_extraction import (
    extract_hog_features,
    detect_and_compute_sift,
    match_sift_descriptors,
    verify_matches_homography,
    draw_feature_matches,
    rotate_image,
    resize_image,
    run_single_matching_experiment
)
from modules.object_detection import (
    HaarDetector,
    YOLODetector,
    run_object_detection_comparison
)
from modules.motion_estimation import (
    extract_sampled_frames,
    compute_frame_differencing,
    run_mog2_background_subtraction,
    run_lucas_kanade_optical_flow,
    build_motion_comparison_table
)
from modules.visualization import (
    create_figure_restoration,
    create_figure_morphology,
    create_figure_watershed,
    create_figure_detection_comparison
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="VisionLab CV – CCA 3 Assignment",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 12px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #064E3B;
        color: #34D399;
    }
    .badge-info {
        background-color: #1E3A8A;
        color: #93C5FD;
    }
    .badge-warn {
        background-color: #78350F;
        color: #FCD34D;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "primary_image_path" not in st.session_state:
    # Auto-detect existing sample in data/inputs
    sample_images = list_files_by_extensions(config.INPUTS_DIR, [".jpg", ".jpeg", ".png"])
    st.session_state.primary_image_path = sample_images[0] if sample_images else None

if "secondary_image_path" not in st.session_state:
    sample_images = list_files_by_extensions(config.INPUTS_DIR, [".jpg", ".jpeg", ".png"])
    st.session_state.secondary_image_path = sample_images[1] if len(sample_images) > 1 else None

if "video_path" not in st.session_state:
    sample_videos = list_files_by_extensions(config.INPUTS_DIR, [".mp4", ".avi", ".mov"])
    st.session_state.video_path = sample_videos[0] if sample_videos else None


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔬 VisionLab CV")
    st.caption("CCA 3 Computer Vision Assignment")
    st.markdown('<span class="status-badge">● Engine Ready</span>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home & Overview",
            "📁 Dataset Manager & Validation",
            "🧪 Task 1: Restoration & Segmentation",
            "🔍 Task 2: Features & Matching (HOG/SIFT)",
            "🎯 Task 3: Object Recognition (Viola-Jones vs YOLO)",
            "🎥 Task 4: Video Motion Estimation",
            "🖼️ Results Gallery",
            "📊 Comparative Benchmark Reports",
            "📖 Viva Guide & Documentation"
        ]
    )

    st.markdown("---")
    st.markdown("#### 📂 Active Dataset Status")
    if st.session_state.primary_image_path:
        st.success(f"📷 Photo: {st.session_state.primary_image_path.name}")
    else:
        st.warning("📷 Photo: Missing")

    if st.session_state.video_path:
        st.success(f"🎥 Video: {st.session_state.video_path.name}")
    else:
        st.warning("🎥 Video: Missing")

    st.markdown("---")
    st.caption("Developed for Academic Evaluation • Python 3.12 • OpenCV 4.9-contrib")


# Helper to load primary image
def get_primary_image_rgb() -> Optional[np.ndarray]:
    if st.session_state.primary_image_path and Path(st.session_state.primary_image_path).exists():
        bgr = cv2.imread(str(st.session_state.primary_image_path))
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return None


def get_secondary_image_rgb() -> Optional[np.ndarray]:
    if st.session_state.secondary_image_path and Path(st.session_state.secondary_image_path).exists():
        bgr = cv2.imread(str(st.session_state.secondary_image_path))
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return None


# =========================================================
# PAGE 1: HOME & OVERVIEW
# =========================================================
if page == "🏠 Home & Overview":
    st.markdown('<div class="main-header">VisionLab CV – CCA 3 Practical Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive Computer Vision Engineering Platform • Units 3, 4 & 5</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Project Objective
        This dashboard implements and benchmarks all four required practical tasks from the **CCA 3 Computer Vision curriculum**:
        
        1. **Task 1 (Unit 3): Image Restoration & Segmentation**
           - Configurable Gaussian & Salt-and-Pepper noise synthesis.
           - Median and Gaussian spatial filtering with PSNR & SSIM quantitative benchmarking.
           - Morphological operations (erosion, dilation, opening, closing) and Canny edge detection.
           - Marker-controlled Watershed segmentation for separating touching regions.
        
        2. **Task 2 (Unit 4): Feature Extraction & Image Matching**
           - Histogram of Oriented Gradients (HOG) feature descriptors & gradient field visualization.
           - Scale-Invariant Feature Transform (SIFT) 128-D keypoint detection and descriptor extraction.
           - FLANN / BFMatcher k-NN matching with Lowe's ratio test and RANSAC Homography verification.
           - Multi-condition invariance experiments (original vs second photo, rotated, resized).
        
        3. **Task 3 (Unit 5): Object Recognition**
           - Viola–Jones classical detection using OpenCV's AdaBoost Haar Cascade (`haarcascade_frontalface_default.xml`).
           - Deep Learning detection using Ultralytics YOLOv8 for 80 COCO generic object categories.
           - Head-to-head architectural comparison (latency, detection counts, confidence, bounding boxes).
        
        4. **Task 4 (Unit 5): Video Motion Estimation**
           - Temporal consecutive frame differencing.
           - Gaussian Mixture Model (GMM MOG2) background subtraction with shadow identification.
           - Lucas–Kanade pyramidal sparse optical flow with Shi-Tomasi corners and trajectory trails.
           - Comparative performance metrics and motion analytics.
        """)

    with col2:
        st.markdown("### 📋 System Architecture")
        st.info("""
        **Environment Specifications:**
        - **Language:** Python 3.12 (macOS / Linux compatible)
        - **Computer Vision Core:** OpenCV 4.9.0-contrib-headless (SIFT, MOG2 enabled)
        - **Numerical & Scientific:** NumPy, SciPy, scikit-image
        - **Deep Learning:** PyTorch 2.2.2, Ultralytics YOLOv8
        - **GUI Dashboard:** Streamlit 1.66
        - **Data Integrity:** Strict non-destructive pipeline (originals never modified)
        """)

        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("""
        1. Go to **📁 Dataset Manager** to view or upload your original photo & video.
        2. Navigate to each **Task tab** to adjust parameters and trigger processing.
        3. Examine quantitative comparison tables and generated figures.
        4. View and download results from the **🖼️ Results Gallery** and **📊 Comparative Reports**.
        5. Check **📖 Viva Guide** for presentation talking points.
        """)


# =========================================================
# PAGE 2: DATASET MANAGER & VALIDATION
# =========================================================
elif page == "📁 Dataset Manager & Validation":
    st.markdown('<div class="main-header">Dataset Management & Input Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Manage and validate student personal photograph and 15–30s video footage</div>', unsafe_allow_html=True)

    tab_img, tab_vid = st.tabs(["📷 Photograph Management", "🎥 Video Footage Management"])

    with tab_img:
        st.markdown("### 1. Primary Photograph (Clean Reference)")
        st.caption("This image serves as the benchmark reference for Task 1, Task 2, and Task 3.")

        col_up1, col_info1 = st.columns([1, 1])
        with col_up1:
            uploaded_img = st.file_uploader(
                "Upload student original photograph (JPG, PNG)",
                type=["jpg", "jpeg", "png"],
                key="primary_img_uploader"
            )
            if uploaded_img:
                saved_path = save_bytes_to_file(uploaded_img.getvalue(), config.INPUTS_DIR, uploaded_img.name)
                st.session_state.primary_image_path = saved_path
                st.success(f"Saved to inputs: {saved_path.name}")

            # Selector from existing inputs
            avail_images = list_files_by_extensions(config.INPUTS_DIR, [".jpg", ".jpeg", ".png"])
            if avail_images:
                curr_idx = 0
                if st.session_state.primary_image_path in avail_images:
                    curr_idx = avail_images.index(st.session_state.primary_image_path)
                selected_primary = st.selectbox(
                    "Select active primary photograph:",
                    avail_images,
                    index=curr_idx,
                    format_func=lambda x: x.name,
                    key="primary_img_selector"
                )
                st.session_state.primary_image_path = selected_primary

        with col_info1:
            if st.session_state.primary_image_path:
                is_valid, msg, meta = validate_image_file(st.session_state.primary_image_path)
                if is_valid:
                    st.success(f"✅ {msg}")
                    st.json(meta)
                    img = get_primary_image_rgb()
                    if img is not None:
                        st.image(img, caption=f"Primary Photograph: {st.session_state.primary_image_path.name}", use_container_width=True)
                else:
                    st.error(f"❌ {msg}")
            else:
                st.info("No primary image loaded yet. Please upload an image.")

        st.markdown("---")
        st.markdown("### 2. Secondary Photograph (Optional for Task 2 Matching)")
        st.caption("Upload a second shot of the same scene from a different perspective, angle, or distance.")

        col_up2, col_info2 = st.columns([1, 1])
        with col_up2:
            uploaded_sec = st.file_uploader(
                "Upload secondary photograph (JPG, PNG)",
                type=["jpg", "jpeg", "png"],
                key="sec_img_uploader"
            )
            if uploaded_sec:
                saved_path2 = save_bytes_to_file(uploaded_sec.getvalue(), config.INPUTS_DIR, uploaded_sec.name)
                st.session_state.secondary_image_path = saved_path2
                st.success(f"Saved secondary image: {saved_path2.name}")

            if avail_images:
                sec_opts = [None] + avail_images
                sec_idx = 0
                if st.session_state.secondary_image_path in sec_opts:
                    sec_idx = sec_opts.index(st.session_state.secondary_image_path)
                chosen_sec = st.selectbox(
                    "Select active secondary photograph:",
                    sec_opts,
                    index=sec_idx,
                    format_func=lambda x: "None (Will use synthetic transformations)" if x is None else x.name,
                    key="sec_img_selector"
                )
                st.session_state.secondary_image_path = chosen_sec

        with col_info2:
            if st.session_state.secondary_image_path:
                is_valid2, msg2, meta2 = validate_image_file(st.session_state.secondary_image_path)
                if is_valid2:
                    st.success(f"✅ {msg2}")
                    st.json(meta2)
                    sec_img = get_secondary_image_rgb()
                    if sec_img is not None:
                        st.image(sec_img, caption=f"Secondary Photograph: {st.session_state.secondary_image_path.name}", use_container_width=True)
                else:
                    st.error(f"❌ {msg2}")
            else:
                st.info("No secondary image selected. Task 2 will evaluate rotation and scale invariance on the primary image.")

    with tab_vid:
        st.markdown("### Self-Recorded Video (15–30 Seconds)")
        st.caption("Used as input stream for Task 4 motion estimation and optical flow.")

        col_v1, col_v2 = st.columns([1, 1])
        with col_v1:
            uploaded_vid = st.file_uploader(
                "Upload student original video (MP4, AVI)",
                type=["mp4", "avi", "mov"],
                key="video_uploader"
            )
            if uploaded_vid:
                saved_v_path = save_bytes_to_file(uploaded_vid.getvalue(), config.INPUTS_DIR, uploaded_vid.name)
                st.session_state.video_path = saved_v_path
                st.success(f"Saved video to inputs: {saved_v_path.name}")

            avail_vids = list_files_by_extensions(config.INPUTS_DIR, [".mp4", ".avi", ".mov"])
            if avail_vids:
                v_idx = 0
                if st.session_state.video_path in avail_vids:
                    v_idx = avail_vids.index(st.session_state.video_path)
                selected_v = st.selectbox(
                    "Select active video footage:",
                    avail_vids,
                    index=v_idx,
                    format_func=lambda x: x.name,
                    key="video_selector"
                )
                st.session_state.video_path = selected_v

        with col_v2:
            if st.session_state.video_path:
                is_v_valid, v_msg, v_meta = validate_video_file(st.session_state.video_path)
                if is_v_valid:
                    st.success(f"✅ {v_msg}")
                    st.json(v_meta)
                    st.video(str(st.session_state.video_path))
                else:
                    st.error(f"❌ {v_msg}")
            else:
                st.warning("No video file selected. Please upload an MP4 or AVI video.")


# =========================================================
# PAGE 3: TASK 1 (RESTORATION & SEGMENTATION)
# =========================================================
elif page == "🧪 Task 1: Restoration & Segmentation":
    st.markdown('<div class="main-header">Task 1: Image Restoration & Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unit 3 Practical • Noise Modelling, Median/Gaussian Filtering, Morphology & Marker Watershed</div>', unsafe_allow_html=True)

    img_rgb = get_primary_image_rgb()
    if img_rgb is None:
        st.warning("⚠️ Please select or upload a primary photograph in the Dataset Manager first.")
        st.stop()

    tab1, tab2, tab3 = st.tabs([
        "A. Noise & Spatial Filtering (PSNR & SSIM)",
        "B. Morphological Operations & Canny Edge",
        "C. Marker-Controlled Watershed Segmentation"
    ])

    # -----------------------------------------------------
    # TAB 1: NOISE & RESTORATION
    # -----------------------------------------------------
    with tab1:
        st.markdown("### Part A: Noise Degradation and Spatial Restoration")
        st.markdown("""
        Evaluates **Gaussian noise** vs **Salt-and-Pepper (impulsive) noise** and compares the restoration efficacy
        of **Median rank filtering** against **Linear Gaussian smoothing** with fair PSNR and SSIM calculation against the clean reference image.
        """)

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            noise_type = st.selectbox("Noise Type:", ["Salt & Pepper", "Gaussian"], key="t1_noise_type")
            if noise_type == "Gaussian":
                noise_sigma = st.slider("Gaussian Sigma (Standard Deviation \u03c3):", 5.0, 60.0, 25.0, 2.5)
                noise_param = noise_sigma
            else:
                noise_amount = st.slider("Noise Ratio (Impulse Density):", 0.01, 0.25, 0.06, 0.01)
                noise_param = noise_amount

        with col_ctrl2:
            median_ksize = st.select_slider("Median Kernel Size (must be odd):", options=[3, 5, 7, 9], value=5)
            gaussian_ksize = st.select_slider("Gaussian Kernel Size (must be odd):", options=[3, 5, 7, 9], value=5)

        with col_ctrl3:
            gaussian_sigma = st.slider("Gaussian Filter \u03c3 (Smoothing Intensity):", 0.5, 3.0, 1.2, 0.1)
            random_seed = st.number_input("Reproducibility Seed:", value=42, step=1)

        if st.button("🚀 Run Restoration Experiment", key="btn_run_t1_restoration"):
            with st.spinner("Injecting noise and calculating restoration metrics..."):
                noisy_img, med_restored, gauss_restored, metrics_df = run_restoration_comparison(
                    clean_image=img_rgb,
                    noise_type=noise_type.lower().replace(" & ", "_").replace(" ", "_"),
                    noise_param=noise_param,
                    gaussian_ksize=gaussian_ksize,
                    gaussian_sigma=gaussian_sigma,
                    median_ksize=median_ksize,
                    seed=int(random_seed)
                )

                # Save generated images
                save_image(noisy_img, config.TASK1_OUT / "noisy_input.jpg")
                save_image(med_restored, config.TASK1_OUT / "median_restored.jpg")
                save_image(gauss_restored, config.TASK1_OUT / "gaussian_restored.jpg")
                csv_path = export_dataframe_csv(metrics_df, config.TASK1_OUT / "restoration_metrics.csv")

                st.session_state["t1_noisy"] = noisy_img
                st.session_state["t1_med"] = med_restored
                st.session_state["t1_gauss"] = gauss_restored
                st.session_state["t1_df"] = metrics_df

        if "t1_noisy" in st.session_state:
            st.markdown("#### Visual Results Comparison Grid")
            c1, c2 = st.columns(2)
            with c1:
                st.image(img_rgb, caption="1. Clean Reference Image", use_container_width=True)
                st.image(st.session_state["t1_med"], caption=f"3. Median Restored (k={median_ksize}x{median_ksize})", use_container_width=True)
            with c2:
                st.image(st.session_state["t1_noisy"], caption=f"2. Noisy Image ({noise_type})", use_container_width=True)
                st.image(st.session_state["t1_gauss"], caption=f"4. Gaussian Restored (k={gaussian_ksize}x{gaussian_ksize})", use_container_width=True)

            st.markdown("#### Quantitative Metric Evaluation (Fair Clean Reference Comparison)")
            st.dataframe(st.session_state["t1_df"], use_container_width=True)

            # Technical takeaways callout
            st.info("""
            **Technical Observation & Theoretical Explanation:**
            - **Salt & Pepper Noise:** Median filtering consistently outperforms Gaussian filtering in PSNR and SSIM because it is an order-statistic rank filter. It replaces outlier impulse pixels with local median values without averaging across sharp image edges.
            - **Gaussian Smoothing:** Operates as a linear low-pass filter. While it suppresses high-frequency Gaussian noise, it blurs salient high-frequency edge gradients, resulting in reduced structural similarity (SSIM).
            """)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                csv_bytes = st.session_state["t1_df"].to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Restoration Metrics (CSV)",
                    data=csv_bytes,
                    file_name="task1_restoration_metrics.csv",
                    mime="text/csv"
                )
            with col_dl2:
                fig = create_figure_restoration(
                    img_rgb,
                    st.session_state["t1_noisy"],
                    st.session_state["t1_med"],
                    st.session_state["t1_gauss"]
                )
                fig_path = config.TASK1_OUT / "restoration_comparison_quad.png"
                fig.savefig(fig_path, dpi=200, bbox_inches='tight')
                with open(fig_path, "rb") as f:
                    st.download_button(
                        "📥 Download High-Resolution Quad Figure (PNG)",
                        data=f.read(),
                        file_name="task1_restoration_figure.png",
                        mime="image/png"
                    )

    # -----------------------------------------------------
    # TAB 2: MORPHOLOGY & CANNY
    # -----------------------------------------------------
    with tab2:
        st.markdown("### Part B: Morphological Operations & Canny Edge Detection")
        st.markdown("""
        Demonstrates non-linear shape-based processing: **Erosion**, **Dilation**, **Opening**, **Closing**,
        and multi-stage **Canny Edge Detection** with gradient suppression and hysteresis.
        """)

        gray_img = to_grayscale(img_rgb)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            thresh_method = st.selectbox(
                "Thresholding Algorithm:",
                ["Otsu (Automatic Bimodal)", "Manual Global", "Adaptive Gaussian", "Adaptive Mean"],
                key="t1_thresh_method"
            )
            invert_mask = st.checkbox("Invert Binary Mask", value=False)

        with col_m2:
            if "Manual" in thresh_method:
                manual_thresh = st.slider("Manual Threshold Cutoff:", 0, 255, 128)
            else:
                manual_thresh = 128

            struct_shape = st.selectbox("Structuring Element Shape:", ["rect", "ellipse", "cross"])
            struct_size = st.select_slider("Kernel Size:", options=[3, 5, 7, 9, 11], value=3)

        with col_m3:
            morph_iters = st.slider("Morphological Iterations:", 1, 5, 1)
            canny_low = st.slider("Canny Low Threshold (Weak Edges):", 10, 150, 50)
            canny_high = st.slider("Canny High Threshold (Strong Edges):", 50, 300, 150)

        # Compute Binary Mask
        m_type = "otsu" if "Otsu" in thresh_method else ("manual" if "Manual" in thresh_method else ("adaptive_gaussian" if "Gaussian" in thresh_method else "adaptive_mean"))
        bin_mask, thresh_val = apply_thresholding(gray_img, method=m_type, manual_thresh=manual_thresh, invert=invert_mask)

        # Compute Morphological Operations
        eroded = apply_morphology(bin_mask, "erosion", kernel_shape=struct_shape, kernel_size=struct_size, iterations=morph_iters)
        dilated = apply_morphology(bin_mask, "dilation", kernel_shape=struct_shape, kernel_size=struct_size, iterations=morph_iters)
        opened = apply_morphology(bin_mask, "opening", kernel_shape=struct_shape, kernel_size=struct_size, iterations=morph_iters)
        closed = apply_morphology(bin_mask, "closing", kernel_shape=struct_shape, kernel_size=struct_size, iterations=morph_iters)
        canny_edges = apply_canny_edge(img_rgb, low_thresh=canny_low, high_thresh=canny_high)

        # Save to disk
        save_image(bin_mask, config.TASK1_OUT / "binary_mask.png")
        save_image(eroded, config.TASK1_OUT / "eroded.png")
        save_image(dilated, config.TASK1_OUT / "dilated.png")
        save_image(opened, config.TASK1_OUT / "opened.png")
        save_image(closed, config.TASK1_OUT / "closed.png")
        save_image(canny_edges, config.TASK1_OUT / "canny_edges.png")

        st.markdown(f"**Computed Threshold Value:** `{thresh_val:.1f}`" if thresh_val >= 0 else "**Computed Threshold:** Local Adaptive Window")

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.image(gray_img, caption="1. Grayscale Input", use_container_width=True)
            st.image(eroded, caption=f"4. Erosion ({struct_shape} {struct_size}x{struct_size})", use_container_width=True)
        with m_col2:
            st.image(bin_mask, caption=f"2. Binary Mask ({thresh_method})", use_container_width=True)
            st.image(dilated, caption=f"5. Dilation ({struct_shape} {struct_size}x{struct_size})", use_container_width=True)
        with m_col3:
            st.image(opened, caption="3. Opening (Erosion -> Dilation)", use_container_width=True)
            st.image(closed, caption="6. Closing (Dilation -> Erosion)", use_container_width=True)

        st.markdown("#### Canny Multi-Stage Edge Map")
        st.image(canny_edges, caption=f"Canny Edge Map (T_low={canny_low}, T_high={canny_high})", use_container_width=True)

        st.info("""
        **Morphology Concepts:**
        - **Erosion:** Strips away boundary pixels; eliminates small foreground noise and separates thin bridges.
        - **Dilation:** Expands foreground borders; joins split fragments and fills tiny interior voids.
        - **Opening:** Erosion followed by dilation; eliminates small foreground protrusions while preserving object scale.
        - **Closing:** Dilation followed by erosion; closes holes and bridges small fractures.
        """)

    # -----------------------------------------------------
    # TAB 3: WATERSHED SEGMENTATION
    # -----------------------------------------------------
    with tab3:
        st.markdown("### Part C: Marker-Controlled Watershed Segmentation")
        st.markdown("""
        Separates touching or clustered objects using mathematical morphology, Euclidean distance transforms,
        connected component topological markers, and immersion flooding.
        """)

        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            dist_ratio = st.slider("Distance Transform Threshold Ratio:", 0.1, 0.9, 0.45, 0.05,
                                  help="Higher values isolate only the deepest object cores as sure foreground markers.")
        with col_w2:
            ws_kernel = st.select_slider("Morphology Clean Kernel:", options=[3, 5, 7], value=3)
        with col_w3:
            invert_ws = st.checkbox("Invert Foreground Prior", value=True,
                                    help="Toggle depending on whether foreground objects are brighter or darker than background.")

        if st.button("🚀 Execute Watershed Segmentation", key="btn_run_watershed"):
            with st.spinner("Calculating distance transform and watershed markers..."):
                ws_res = apply_watershed_segmentation(
                    img_rgb,
                    dist_thresh_ratio=dist_ratio,
                    morph_kernel_size=ws_kernel,
                    invert_binary=invert_ws
                )

                st.session_state["ws_res"] = ws_res

                # Save intermediate and final outputs
                save_image(ws_res["dist_transform_vis"], config.TASK1_OUT / "watershed_dist_transform.jpg")
                save_image(ws_res["sure_fg"], config.TASK1_OUT / "watershed_sure_fg.png")
                save_image(ws_res["segmented_overlay"], config.TASK1_OUT / "watershed_boundary_overlay.jpg")
                save_image(ws_res["segmented_colored"], config.TASK1_OUT / "watershed_segmented_colored.jpg")

        if "ws_res" in st.session_state:
            ws_res = st.session_state["ws_res"]

            # Suitability diagnosis card
            if ws_res["num_objects"] > 40:
                st.warning(f"⚠️ **Suitability Assessment:** {ws_res['suitability_note']}")
            else:
                st.success(f"✅ **Suitability Assessment:** {ws_res['suitability_note']}")

            st.markdown("#### Step-by-Step Intermediate Representations")
            w1, w2, w3 = st.columns(3)
            with w1:
                st.image(ws_res["binary_mask"], caption="1. Initial Binary Mask", use_container_width=True)
                st.image(ws_res["sure_fg"], caption="4. Sure Foreground Markers", use_container_width=True)
            with w2:
                st.image(ws_res["dist_transform_vis"], caption="2. Distance Transform (Jet Colormap)", use_container_width=True)
                st.image(ws_res["segmented_colored"], caption="5. Region Segment Labels", use_container_width=True)
            with w3:
                st.image(ws_res["sure_bg"], caption="3. Sure Background Region", use_container_width=True)
                st.image(ws_res["segmented_overlay"], caption=f"6. Watershed Boundaries in Red ({ws_res['num_objects']} regions)", use_container_width=True)


# =========================================================
# PAGE 4: TASK 2 (FEATURE EXTRACTION & MATCHING)
# =========================================================
elif page == "🔍 Task 2: Features & Matching (HOG/SIFT)":
    st.markdown('<div class="main-header">Task 2: Feature Extraction & Image Matching</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unit 4 Practical • HOG Descriptors, SIFT Scale-Space Invariance & RANSAC Geometric Homography</div>', unsafe_allow_html=True)

    img_rgb = get_primary_image_rgb()
    if img_rgb is None:
        st.warning("⚠️ Please select or upload a primary photograph in the Dataset Manager first.")
        st.stop()

    tab_hog, tab_sift, tab_match = st.tabs([
        "A. HOG Feature Extraction",
        "B. SIFT Keypoints & Descriptors",
        "C. Feature Matching & Invariance Experiments"
    ])

    # -----------------------------------------------------
    # TAB 1: HOG
    # -----------------------------------------------------
    with tab_hog:
        st.markdown("### Part A: Histogram of Oriented Gradients (HOG)")
        st.markdown("""
        HOG captures structural edge gradients by accumulating orientation histograms across dense spatial cells,
        normalized over overlapping blocks for local contrast invariance.
        """)

        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            hog_orient = st.slider("Gradient Angular Bins (Orientations):", 6, 12, 9, 1)
        with col_h2:
            ppc = st.select_slider("Pixels Per Cell:", options=[8, 16, 24, 32], value=16)
        with col_h3:
            cpb = st.select_slider("Cells Per Block:", options=[1, 2, 3], value=2)

        if st.button("🚀 Extract & Visualize HOG Features", key="btn_run_hog"):
            with st.spinner("Computing spatial gradient histograms..."):
                feat_vec, hog_vis, hog_info = extract_hog_features(
                    img_rgb,
                    orientations=hog_orient,
                    pixels_per_cell=(ppc, ppc),
                    cells_per_block=(cpb, cpb)
                )

                save_image(hog_vis, config.TASK2_OUT / "hog_visualization.jpg")
                st.session_state["hog_vis"] = hog_vis
                st.session_state["hog_info"] = hog_info
                st.session_state["hog_feat_len"] = len(feat_vec)

        if "hog_vis" in st.session_state:
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(img_rgb, caption="Input RGB Photograph", use_container_width=True)
            with col_img2:
                st.image(st.session_state["hog_vis"], caption=f"HOG Gradient Vector Field (Magma Colormap, {st.session_state['hog_feat_len']:,} dimensions)", use_container_width=True)

            st.json(st.session_state["hog_info"])

    # -----------------------------------------------------
    # TAB 2: SIFT
    # -----------------------------------------------------
    with tab_sift:
        st.markdown("### Part B: SIFT Keypoint Detection and 128-D Descriptors")
        st.markdown("""
        Scale-Invariant Feature Transform detects extrema across Difference-of-Gaussian (DoG) scale-space octaves,
        assigns canonical orientations, and computes 128-dimensional local gradient descriptors.
        """)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sift_nfeat = st.slider("Maximum Keypoints:", 200, 2500, 1000, 100)
        with col_s2:
            sift_contrast = st.slider("Contrast Threshold (DoG Filter):", 0.01, 0.10, 0.04, 0.01)
        with col_s3:
            sift_edge = st.slider("Edge Threshold (Principal Curvature):", 5.0, 20.0, 10.0, 1.0)

        if st.button("🚀 Detect SIFT Keypoints", key="btn_run_sift"):
            with st.spinner("Extracting scale-space extrema and descriptors..."):
                kp, desc, vis_sift, sift_ms = detect_and_compute_sift(
                    img_rgb,
                    n_features=sift_nfeat,
                    contrast_thresh=sift_contrast,
                    edge_thresh=sift_edge
                )

                save_image(vis_sift, config.TASK2_OUT / "sift_keypoints.jpg")
                st.session_state["sift_vis"] = vis_sift
                st.session_state["sift_kp_len"] = len(kp)
                st.session_state["sift_desc_shape"] = desc.shape if desc is not None else (0, 0)
                st.session_state["sift_ms"] = sift_ms

        if "sift_vis" in st.session_state:
            st.image(st.session_state["sift_vis"], caption=f"SIFT Detected Keypoints ({st.session_state['sift_kp_len']} keypoints with scale & orientation)", use_container_width=True)
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Detected Keypoints", st.session_state["sift_kp_len"])
            col_m2.metric("Descriptor Dimensions", f"{st.session_state['sift_desc_shape'][1]}-D" if st.session_state['sift_desc_shape'][0] > 0 else "None")
            col_m3.metric("Computation Latency", f"{st.session_state['sift_ms']:.1f} ms")

    # -----------------------------------------------------
    # TAB 3: MATCHING EXPERIMENTS
    # -----------------------------------------------------
    with tab_match:
        st.markdown("### Part C: Image Matching & Geometric Invariance Experiments")
        st.markdown(r"""
        Evaluates matching accuracy using **k-NN with Lowe's Ratio Test** ($d_1 < 0.75 \cdot d_2$)
        and verifies spatial consistency using **RANSAC Homography Estimation**.
        """)

        exp_type = st.radio(
            "Select Invariance Experiment Scenario:",
            [
                "Scenario 1: Original vs Rotated Image (Rotation Invariance)",
                "Scenario 2: Original vs Resized Image (Scale Invariance)",
                "Scenario 3: Original vs Second Image (Viewpoint / Perspective Invariance)"
            ]
        )

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            if "Rotated" in exp_type:
                rot_angle = st.slider("Rotation Angle (Degrees):", -180, 180, 45, 15)
                target_img2 = rotate_image(img_rgb, rot_angle)
                exp_label = f"Rotated ({rot_angle}°)"
            elif "Resized" in exp_type:
                scale_factor = st.slider("Scaling Factor:", 0.3, 2.0, 0.6, 0.1)
                target_img2 = resize_image(img_rgb, scale_factor)
                exp_label = f"Rescaled ({scale_factor}x)"
            else:
                sec = get_secondary_image_rgb()
                if sec is not None:
                    target_img2 = sec
                    exp_label = f"Second Photo ({st.session_state.secondary_image_path.name})"
                else:
                    st.warning("No secondary photo found. Using 30° rotation fallback.")
                    target_img2 = rotate_image(img_rgb, 30)
                    exp_label = "Synthesized Variation (30° Rotation)"

        with col_p2:
            lowe_ratio = st.slider("Lowe's Ratio Test Threshold:", 0.50, 0.90, 0.75, 0.05,
                                  help="Discard matches where second closest neighbor distance is close to first.")
        with col_p3:
            ransac_threshold = st.slider("RANSAC Inlier Reprojection Distance (px):", 1.0, 10.0, 5.0, 0.5)

        if st.button("🚀 Run Matching Experiment", key="btn_run_matching"):
            with st.spinner("Matching descriptors and computing RANSAC homography..."):
                match_results = run_single_matching_experiment(
                    img1=img_rgb,
                    img2=target_img2,
                    experiment_name=exp_label,
                    ratio_thresh=lowe_ratio,
                    ransac_thresh=ransac_threshold
                )

                st.session_state["match_results"] = match_results
                # Save visualization
                save_image(match_results["matches_vis"], config.TASK2_OUT / f"matches_{exp_label.replace(' ', '_')}.jpg")

        if "match_results" in st.session_state:
            res = st.session_state["match_results"]
            st.image(res["matches_vis"], caption=f"SIFT Correspondence Matches with RANSAC Geometric Verification ({exp_label})", use_container_width=True)

            m_df = pd.DataFrame([res["metrics"]])
            st.dataframe(m_df, use_container_width=True)

            st.info(f"""
            **Theoretical Analysis for {exp_label}:**
            - **Candidate Matches:** Initial k-NN matches before filtering.
            - **Lowe's Ratio Test:** Successfully pruned ambiguous keypoints lacking distinctiveness.
            - **RANSAC Inliers:** {res['inliers_count']} geometrically consistent correspondences ({res['metrics']['Inlier Ratio']*100:.1f}% inlier ratio).
            - **SIFT Invariance:** Confirms SIFT's robustness against rotation and scale transformations through orientation assignment and multi-scale Gaussian pyramid representations.
            """)


# =========================================================
# PAGE 5: TASK 3 (OBJECT RECOGNITION)
# =========================================================
elif page == "🎯 Task 3: Object Recognition (Viola-Jones vs YOLO)":
    st.markdown('<div class="main-header">Task 3: Object Recognition & Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unit 5 Practical • Viola–Jones AdaBoost Cascade vs Ultralytics YOLOv8 Deep CNN</div>', unsafe_allow_html=True)

    img_rgb = get_primary_image_rgb()
    if img_rgb is None:
        st.warning("⚠️ Please select or upload a primary photograph in the Dataset Manager first.")
        st.stop()

    st.markdown("""
    This task provides a fair, empirical comparison between:
    1. **Viola-Jones:** Hand-crafted rectangular Haar-like features + Integral Image + AdaBoost cascade (specialized for frontal human faces).
    2. **Ultralytics YOLOv8:** End-to-end deep convolutional neural network trained on COCO (detects 80 generic object categories including people, vehicles, animals, and common indoor/outdoor items).
    """)

    col_det1, col_det2 = st.columns(2)
    with col_det1:
        st.markdown("#### 1. Viola-Jones (Haar Cascade) Controls")
        haar_scale = st.slider("Scale Factor:", 1.05, 1.40, 1.15, 0.05,
                               help="Specifies how much the image size is reduced at each image scale.")
        haar_neighbors = st.slider("Min Neighbors:", 2, 10, 5, 1,
                                   help="Specifies how many neighbors each candidate rectangle should retain.")

    with col_det2:
        st.markdown("#### 2. Ultralytics YOLO Controls")
        yolo_model_choice = st.selectbox("YOLO Model Architecture:", ["yolov8n.pt", "yolov8s.pt"], index=0,
                                         help="yolov8n is ultra-fast nano model; yolov8s has higher capacity.")
        yolo_conf = st.slider("Confidence Threshold:", 0.15, 0.90, 0.25, 0.05)

    if st.button("🚀 Run Object Recognition Benchmark", key="btn_run_detection"):
        with st.spinner("Executing Viola-Jones cascade and YOLO inference..."):
            vis_haar, vis_yolo, comp_df, analysis = run_object_detection_comparison(
                image=img_rgb,
                haar_scale_factor=haar_scale,
                haar_min_neighbors=haar_neighbors,
                yolo_model_name=yolo_model_choice,
                yolo_conf=yolo_conf
            )

            # Save to disk
            save_image(vis_haar, config.TASK3_OUT / "viola_jones_detections.jpg")
            save_image(vis_yolo, config.TASK3_OUT / "yolo_detections.jpg")
            export_dataframe_csv(comp_df, config.TASK3_OUT / "detection_comparison.csv")

            st.session_state["det_haar"] = vis_haar
            st.session_state["det_yolo"] = vis_yolo
            st.session_state["det_df"] = comp_df
            st.session_state["det_analysis"] = analysis

    if "det_haar" in st.session_state:
        st.markdown("#### Head-to-Head Visual Detection Comparison")
        c1, c2 = st.columns(2)
        with c1:
            st.image(st.session_state["det_haar"], caption="Viola-Jones Haar Cascade (Frontal Faces in Red)", use_container_width=True)
        with c2:
            st.image(st.session_state["det_yolo"], caption=f"Ultralytics YOLO ({yolo_model_choice}) (COCO Classes in Green)", use_container_width=True)

        st.markdown("#### Quantitative Benchmark Table")
        st.dataframe(st.session_state["det_df"], use_container_width=True)

        # Observations
        analysis = st.session_state["det_analysis"]
        st.info(f"""
        **Factual Engineering Observations:**
        - **Classifier Specificity:** Viola-Jones detected `{analysis['haar_count']}` face regions. It relies on handcrafted intensity contrasts (like the bridge of the nose being brighter than the eye sockets) and cannot detect generic objects.
        - **Multiclass Generalization:** YOLO detected `{analysis['yolo_count']}` bounding boxes across classes: `{', '.join(analysis['yolo_classes']) if analysis['yolo_classes'] else 'None above threshold'}`.
        - **Inference Latency:** Viola-Jones ran in `{analysis['haar_time']} ms` vs YOLO in `{analysis['yolo_time']} ms`.
        - **Ground Truth Note:** Quantitative accuracy (e.g. mAP or IoU precision) cannot be claimed without human-annotated ground-truth bounding box labels for this specific photograph.
        """)


# =========================================================
# PAGE 6: TASK 4 (VIDEO MOTION ESTIMATION)
# =========================================================
elif page == "🎥 Task 4: Video Motion Estimation":
    st.markdown('<div class="main-header">Task 4: Video Motion Estimation & Tracking</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unit 5 Practical • Consecutive Frame Differencing, GMM Background Subtraction & Lucas–Kanade Optical Flow</div>', unsafe_allow_html=True)

    if not st.session_state.video_path or not Path(st.session_state.video_path).exists():
        st.warning("⚠️ Please select or upload a student video file in the Dataset Manager first.")
        st.stop()

    st.markdown(r"""
    Evaluates three distinct paradigms of motion analysis on student video footage:
    1. **Frame Differencing:** High-speed temporal pixel difference $|I_t - I_{t-1}| > T$.
    2. **GMM Background Subtraction (MOG2):** Statistical Gaussian mixture modelling with adaptive background update and shadow discrimination.
    3. **Lucas–Kanade Optical Flow:** Spatial intensity gradient tracking of Shi-Tomasi corner features across consecutive pyramidal frames.
    """)

    col_vm1, col_vm2, col_vm3 = st.columns(3)
    with col_vm1:
        st.markdown("**1. Video Sampling & Differencing**")
        max_frames = st.slider("Max Frames to Process:", 20, 150, 60, 10,
                               help="Safeguard against high processing times on long HD videos.")
        sampling_step = st.select_slider("Temporal Sampling Step:", options=[1, 2, 3, 4], value=2,
                                         help="Process every Nth frame to accelerate analysis.")
        diff_threshold = st.slider("Differencing Threshold:", 10, 80, 25)

    with col_vm2:
        st.markdown("**2. GMM Subtraction (MOG2)**")
        mog_history = st.slider("MOG2 History Frames:", 30, 200, 80)
        mog_var = st.slider("MOG2 Variance Threshold:", 8.0, 40.0, 16.0)
        mog_shadows = st.checkbox("Detect & Mark Shadows (Blue tint)", value=True)

    with col_vm3:
        st.markdown("**3. Lucas–Kanade Optical Flow**")
        lk_corners = st.slider("Max Tracked Corners:", 30, 250, 100)
        lk_quality = st.slider("Corner Quality Level:", 0.01, 0.10, 0.03, 0.01)
        lk_window = st.select_slider("LK Window Size (px):", options=[11, 15, 21, 31], value=15)

    if st.button("🚀 Execute Video Motion Analysis", key="btn_run_motion"):
        with st.spinner("Extracting video frames and processing motion pipelines..."):
            prog_bar = st.progress(0.1, text="1/4 Reading and sampling video stream...")

            frames, v_meta = extract_sampled_frames(
                st.session_state.video_path,
                max_frames=max_frames,
                sampling_step=sampling_step
            )

            prog_bar.progress(0.35, text="2/4 Processing temporal frame differencing...")
            diff_masks = []
            diff_times = []
            for i in range(1, len(frames)):
                with BenchmarkTimer() as t_d:
                    _, mask, _ = compute_frame_differencing(frames[i-1], frames[i], threshold=diff_threshold)
                diff_masks.append(mask)
                diff_times.append(t_d.elapsed_ms)
            avg_diff_ms = float(np.mean(diff_times)) if diff_times else 0.0

            prog_bar.progress(0.65, text="3/4 Applying GMM Background Subtraction (MOG2)...")
            fg_masks, mog_overlays, avg_mog_ms = run_mog2_background_subtraction(
                frames, history=mog_history, var_threshold=mog_var, detect_shadows=mog_shadows
            )

            prog_bar.progress(0.90, text="4/4 Tracking Lucas-Kanade Pyramidal Optical Flow...")
            flow_frames, avg_lk_ms, flow_stats = run_lucas_kanade_optical_flow(
                frames, max_corners=lk_corners, quality_level=lk_quality, win_size=lk_window
            )

            prog_bar.progress(1.0, text="Motion analysis completed successfully!")

            # Build comparison table
            motion_df = build_motion_comparison_table(avg_diff_ms, avg_mog_ms, avg_lk_ms)
            export_dataframe_csv(motion_df, config.TASK4_OUT / "motion_comparison.csv")

            # Save representative samples
            if len(frames) > 2:
                mid = len(frames) // 2
                save_image(frames[mid], config.TASK4_OUT / "sample_original_frame.jpg")
                save_image(diff_masks[min(mid, len(diff_masks)-1)], config.TASK4_OUT / "sample_differencing_mask.png")
                save_image(mog_overlays[mid], config.TASK4_OUT / "sample_mog2_overlay.jpg")
                save_image(flow_frames[mid], config.TASK4_OUT / "sample_lk_flow_frame.jpg")

            st.session_state["motion_frames"] = frames
            st.session_state["diff_masks"] = diff_masks
            st.session_state["mog_overlays"] = mog_overlays
            st.session_state["flow_frames"] = flow_frames
            st.session_state["motion_df"] = motion_df
            st.session_state["flow_stats"] = flow_stats

    if "motion_frames" in st.session_state:
        frames = st.session_state["motion_frames"]
        diff_masks = st.session_state["diff_masks"]
        mog_overlays = st.session_state["mog_overlays"]
        flow_frames = st.session_state["flow_frames"]

        st.markdown("#### Interactive Frame Inspector")
        frame_slider = st.slider("Select Frame Index to Inspect:", 1, len(frames) - 1, len(frames) // 2)

        c1, c2 = st.columns(2)
        with c1:
            st.image(frames[frame_slider], caption=f"Original Frame #{frame_slider}", use_container_width=True)
            st.image(mog_overlays[frame_slider], caption="MOG2 Background Subtraction (Green=Foreground, Blue=Shadow)", use_container_width=True)
        with c2:
            st.image(diff_masks[frame_slider - 1], caption=f"Frame Differencing Mask (|I_{frame_slider} - I_{frame_slider-1}|)", use_container_width=True)
            st.image(flow_frames[frame_slider], caption="Lucas-Kanade Pyramidal Optical Flow Vectors", use_container_width=True)

        st.markdown("#### Quantitative Motion Paradigm Benchmarking")
        st.dataframe(st.session_state["motion_df"], use_container_width=True)

        st.info(r"""
        **Crucial Engineering Distinctions:**
        - **Foreground Segmentation (Differencing & MOG2):** Generates binary or tri-state spatial silhouettes indicating *where* motion occurred, but does *not* provide trajectory direction or velocity.
        - **Optical Flow (Lucas–Kanade):** Estimates explicit 2D motion velocity vectors $(u, v)$ for feature points, enabling velocity estimation and trajectory tracking.
        - **Shadow Vulnerability:** Frame differencing falsely classifies shifting illumination and shadows as moving objects, whereas MOG2 explicitly models chromaticity changes to isolate cast shadows.
        """)


# =========================================================
# PAGE 7: RESULTS GALLERY
# =========================================================
elif page == "🖼️ Results Gallery":
    st.markdown('<div class="main-header">Experimental Results Gallery</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Browse and download all saved processed artifacts from completed tasks</div>', unsafe_allow_html=True)

    task_filter = st.selectbox(
        "Select Task Artifact Folder:",
        [
            "All Tasks",
            "Task 1: Restoration & Segmentation",
            "Task 2: Feature Extraction & Matching",
            "Task 3: Object Recognition",
            "Task 4: Motion Estimation"
        ]
    )

    folders_to_scan = []
    if task_filter in ["All Tasks", "Task 1: Restoration & Segmentation"]:
        folders_to_scan.append(config.TASK1_OUT)
    if task_filter in ["All Tasks", "Task 2: Feature Extraction & Matching"]:
        folders_to_scan.append(config.TASK2_OUT)
    if task_filter in ["All Tasks", "Task 3: Object Recognition"]:
        folders_to_scan.append(config.TASK3_OUT)
    if task_filter in ["All Tasks", "Task 4: Motion Estimation"]:
        folders_to_scan.append(config.TASK4_OUT)

    all_files = []
    for fld in folders_to_scan:
        all_files.extend(list_files_by_extensions(fld, [".jpg", ".jpeg", ".png"]))

    if not all_files:
        st.info("No saved output images found yet. Run tasks from the sidebar to populate the gallery!")
    else:
        st.success(f"Found {len(all_files)} saved output artifacts:")
        cols = st.columns(3)
        for i, file_path in enumerate(all_files):
            col = cols[i % 3]
            with col:
                img = Image.open(str(file_path))
                st.image(img, caption=f"{file_path.parent.name}/{file_path.name}", use_container_width=True)
                with open(file_path, "rb") as f:
                    st.download_button(
                        f"📥 Download {file_path.name}",
                        data=f.read(),
                        file_name=file_path.name,
                        key=f"dl_gal_{i}"
                    )


# =========================================================
# PAGE 8: COMPARATIVE BENCHMARK REPORTS
# =========================================================
elif page == "📊 Comparative Benchmark Reports":
    st.markdown('<div class="main-header">Comparative Benchmark Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Quantitative tables and structured metric logs across all assignment experiments</div>', unsafe_allow_html=True)

    csv_files = []
    for fld in [config.TASK1_OUT, config.TASK2_OUT, config.TASK3_OUT, config.TASK4_OUT, config.REPORTS_DIR]:
        csv_files.extend(list_files_by_extensions(fld, [".csv"]))

    if not csv_files:
        st.info("No CSV experiment logs found yet. Execute tasks to generate benchmark logs!")
    else:
        for csv_path in csv_files:
            st.markdown(f"### 📄 `{csv_path.parent.name} / {csv_path.name}`")
            try:
                df = pd.read_csv(csv_path)
                st.dataframe(df, use_container_width=True)
                with open(csv_path, "rb") as f:
                    st.download_button(
                        f"📥 Download {csv_path.name}",
                        data=f.read(),
                        file_name=csv_path.name,
                        key=f"dl_csv_{csv_path.name}"
                    )
            except Exception as e:
                st.error(f"Error reading {csv_path.name}: {e}")
            st.markdown("---")


# =========================================================
# PAGE 9: VIVA GUIDE & DOCUMENTATION
# =========================================================
elif page == "📖 Viva Guide & Documentation":
    st.markdown('<div class="main-header">Viva Voce Presentation & Assignment Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">How to demonstrate your practical work to Computer Vision faculty</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎙️ 5–8 Minute Faculty Demonstration Script
    Use this chronological structure during your viva or project presentation:

    #### Step 1: Dataset Presentation (1 Minute)
    - Show your **original personal photograph** and **self-recorded video**.
    - Explain that no synthetic or stock images were used.
    - Highlight the metadata validation: resolution, channels, FPS, frame count, and non-destructive storage in `data/inputs`.

    #### Step 2: Task 1 – Restoration & Segmentation (2 Minutes)
    - **Noise Degradation:** Explain Gaussian noise as thermal sensor fluctuation and Salt & Pepper as transmission bit error.
    - **Filter Demonstration:** Show how Median filter completely eliminates Salt & Pepper noise with high PSNR/SSIM, while Gaussian filter blurs edges.
    - **Watershed Segmentation:** Explain the mathematical pipeline: Otsu threshold -> Morphological opening -> Euclidean Distance Transform -> Sure Foreground Markers -> Watershed immersion flooding.

    #### Step 3: Task 2 – Feature Matching (1.5 Minutes)
    - **HOG:** Explain dense spatial cell gradient histograms for structural shape representation.
    - **SIFT:** Explain scale-space Difference-of-Gaussian extrema detection and 128-D orientation descriptors.
    - **Invariance:** Demonstrate matching under 45° rotation and 0.6x scaling, showing how Lowe's ratio test and RANSAC Homography eliminate spurious correspondences.

    #### Step 4: Task 3 – Object Recognition (1.5 Minutes)
    - **Viola-Jones vs YOLO:** Explain that Viola-Jones uses handcrafted Haar features with AdaBoost cascade for frontal faces, whereas YOLOv8 is a deep CNN detecting 80 COCO categories.
    - Show the side-by-side detections and runtime comparison table.

    #### Step 5: Task 4 – Motion Estimation (1.5 Minutes)
    - **Three-way Comparison:** Explain Frame Differencing ($|I_t - I_{t-1}|$), GMM MOG2 background subtraction with shadow identification, and Lucas-Kanade optical flow vector tracking.
    - Explain the difference between spatial motion masks and directional motion velocity vectors.

    ---
    ### 📚 Algorithm Mathematical Formulations
    """)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown(r"""
        **1. Peak Signal-to-Noise Ratio (PSNR):**
        $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right)$$

        **2. Structural Similarity Index (SSIM):**
        $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
        """)
    with col_f2:
        st.markdown(r"""
        **3. Lucas–Kanade Optical Flow Constraint:**
        $$I_x u + I_y v + I_t = 0$$
        $$\begin{bmatrix} u \\ v \end{bmatrix} = (A^T A)^{-1} A^T b$$

        **4. Projective Homography Matrix:**
        $$\mathbf{x}' = \mathbf{H} \mathbf{x} \quad \text{where } \mathbf{H} \in \mathbb{R}^{3 \times 3}$$
        """)
