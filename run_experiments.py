"""
Automated experiment execution script for VisionLab CV.
Processes active student dataset (user_mountain_village.jpg and user_video.mp4)
and populates output directories and benchmark reports with genuine experimental results.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd

import config
from modules.restoration import run_restoration_comparison
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
    rotate_image,
    resize_image,
    run_single_matching_experiment
)
from modules.object_detection import run_object_detection_comparison
from modules.motion_estimation import (
    extract_sampled_frames,
    compute_frame_differencing,
    run_mog2_background_subtraction,
    run_lucas_kanade_optical_flow,
    build_motion_comparison_table
)
from utils.file_utils import save_image, export_dataframe_csv
from modules.metrics import BenchmarkTimer


def main():
    print("=" * 60)
    print("VisionLab CV – Executing Full Assignment Practical Pipeline")
    print("=" * 60)

    # 1. Load Primary Input Image
    img_path = config.INPUTS_DIR / "user_mountain_village.jpg"
    if not img_path.exists():
        print(f"Error: {img_path} not found.")
        return

    bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    print(f"Loaded Primary Photograph: {img_path.name} ({w}x{h})")

    # ---------------------------------------------------------
    # Task 1: Image Restoration & Segmentation
    # ---------------------------------------------------------
    print("\n[Task 1/4] Running Image Restoration & Segmentation...")
    # Part A: Noise & Filtering
    noisy_sp, med_sp, gauss_sp, df_sp = run_restoration_comparison(
        rgb, noise_type="salt_pepper", noise_param=0.05, gaussian_ksize=5, median_ksize=5, seed=42
    )
    save_image(noisy_sp, config.TASK1_OUT / "noisy_salt_pepper.jpg")
    save_image(med_sp, config.TASK1_OUT / "median_restored_sp.jpg")
    save_image(gauss_sp, config.TASK1_OUT / "gaussian_restored_sp.jpg")
    export_dataframe_csv(df_sp, config.TASK1_OUT / "restoration_metrics_salt_pepper.csv")
    export_dataframe_csv(df_sp, config.REPORTS_DIR / "task1_restoration_metrics.csv")

    noisy_g, med_g, gauss_g, df_g = run_restoration_comparison(
        rgb, noise_type="gaussian", noise_param=25.0, gaussian_ksize=5, median_ksize=5, seed=42
    )
    save_image(noisy_g, config.TASK1_OUT / "noisy_gaussian.jpg")
    save_image(med_g, config.TASK1_OUT / "median_restored_gaussian.jpg")
    save_image(gauss_g, config.TASK1_OUT / "gaussian_restored_gaussian.jpg")
    export_dataframe_csv(df_g, config.TASK1_OUT / "restoration_metrics_gaussian.csv")

    # Part B: Morphology & Canny
    gray = to_grayscale(rgb)
    bin_mask, _ = apply_thresholding(gray, method="otsu")
    eroded = apply_morphology(bin_mask, "erosion", kernel_size=3)
    dilated = apply_morphology(bin_mask, "dilation", kernel_size=3)
    opened = apply_morphology(bin_mask, "opening", kernel_size=3)
    closed = apply_morphology(bin_mask, "closing", kernel_size=3)
    canny = apply_canny_edge(rgb, low_thresh=50, high_thresh=150)

    save_image(bin_mask, config.TASK1_OUT / "binary_mask_otsu.png")
    save_image(eroded, config.TASK1_OUT / "morphology_erosion.png")
    save_image(dilated, config.TASK1_OUT / "morphology_dilation.png")
    save_image(opened, config.TASK1_OUT / "morphology_opening.png")
    save_image(closed, config.TASK1_OUT / "morphology_closing.png")
    save_image(canny, config.TASK1_OUT / "canny_edges.png")

    # Part C: Watershed
    ws_res = apply_watershed_segmentation(rgb, dist_thresh_ratio=0.45, morph_kernel_size=3)
    save_image(ws_res["dist_transform_vis"], config.TASK1_OUT / "watershed_dist_transform.jpg")
    save_image(ws_res["sure_fg"], config.TASK1_OUT / "watershed_sure_fg.png")
    save_image(ws_res["segmented_overlay"], config.TASK1_OUT / "watershed_boundary_overlay.jpg")
    save_image(ws_res["segmented_colored"], config.TASK1_OUT / "watershed_segmented_colored.jpg")
    print(f" Task 1 Completed: {ws_res['num_objects']} regions segmented.")

    # ---------------------------------------------------------
    # Task 2: Feature Extraction & Matching
    # ---------------------------------------------------------
    print("\n[Task 2/4] Running Feature Extraction & Matching...")
    # HOG
    _, hog_vis, hog_info = extract_hog_features(rgb, orientations=9, pixels_per_cell=(16, 16))
    save_image(hog_vis, config.TASK2_OUT / "hog_gradient_visualization.jpg")

    # SIFT
    kp, desc, sift_vis, _ = detect_and_compute_sift(rgb, n_features=1000)
    save_image(sift_vis, config.TASK2_OUT / "sift_keypoints.jpg")

    # Matching Invariance Experiments
    # Experiment 1: Rotation (45 degrees)
    rot_45 = rotate_image(rgb, 45)
    res_rot = run_single_matching_experiment(rgb, rot_45, "Rotation (45 deg)", ratio_thresh=0.75)
    save_image(res_rot["matches_vis"], config.TASK2_OUT / "matches_rotation_45deg.jpg")

    # Experiment 2: Scale (0.6x)
    scale_06 = resize_image(rgb, 0.6)
    res_scale = run_single_matching_experiment(rgb, scale_06, "Scaling (0.6x)", ratio_thresh=0.75)
    save_image(res_scale["matches_vis"], config.TASK2_OUT / "matches_scale_0.6x.jpg")

    matching_df = pd.DataFrame([res_rot["metrics"], res_scale["metrics"]])
    export_dataframe_csv(matching_df, config.TASK2_OUT / "matching_invariance_metrics.csv")
    export_dataframe_csv(matching_df, config.REPORTS_DIR / "task2_feature_matching_metrics.csv")
    print(f" Task 2 Completed: {len(kp)} SIFT keypoints, HOG features, and invariance matches verified.")

    # ---------------------------------------------------------
    # Task 3: Object Recognition
    # ---------------------------------------------------------
    print("\n[Task 3/4] Running Object Recognition Benchmark (Viola-Jones vs YOLOv8)...")
    vis_haar, vis_yolo, df_det, analysis = run_object_detection_comparison(
        rgb, haar_scale_factor=1.15, haar_min_neighbors=5, yolo_model_name="yolov8n.pt", yolo_conf=0.25
    )
    save_image(vis_haar, config.TASK3_OUT / "viola_jones_detections.jpg")
    save_image(vis_yolo, config.TASK3_OUT / "yolo_detections.jpg")
    export_dataframe_csv(df_det, config.TASK3_OUT / "detection_comparison.csv")
    export_dataframe_csv(df_det, config.REPORTS_DIR / "task3_detection_comparison.csv")
    print(f" Task 3 Completed: Haar={analysis['haar_count']} faces, YOLO={analysis['yolo_count']} objects {analysis['yolo_classes']}.")

    # ---------------------------------------------------------
    # Task 4: Video Motion Estimation
    # ---------------------------------------------------------
    vid_path = config.INPUTS_DIR / "user_video.mp4"
    if vid_path.exists():
        print(f"\n[Task 4/4] Processing Video Motion Estimation on {vid_path.name}...")
        frames, v_meta = extract_sampled_frames(vid_path, max_frames=50, sampling_step=2)
        print(f" Extracted {len(frames)} sampled frames from video.")

        # Differencing
        diff_times = []
        diff_masks = []
        for i in range(1, len(frames)):
            with BenchmarkTimer() as td:
                _, mask, _ = compute_frame_differencing(frames[i-1], frames[i], threshold=25)
            diff_times.append(td.elapsed_ms)
            diff_masks.append(mask)
        avg_diff_ms = float(np.mean(diff_times))

        # MOG2
        fg_masks, mog_overlays, avg_mog_ms = run_mog2_background_subtraction(frames, history=80, var_threshold=16.0)

        # Lucas-Kanade
        flow_frames, avg_lk_ms, flow_stats = run_lucas_kanade_optical_flow(frames, max_corners=120)

        # Save sample outputs
        mid = len(frames) // 2
        save_image(frames[mid], config.TASK4_OUT / "original_sample_frame.jpg")
        save_image(diff_masks[mid], config.TASK4_OUT / "differencing_motion_mask.png")
        save_image(mog_overlays[mid], config.TASK4_OUT / "mog2_background_subtraction_overlay.jpg")
        save_image(flow_frames[mid], config.TASK4_OUT / "lucas_kanade_optical_flow_vectors.jpg")

        motion_df = build_motion_comparison_table(avg_diff_ms, avg_mog_ms, avg_lk_ms)
        export_dataframe_csv(motion_df, config.TASK4_OUT / "motion_comparison.csv")
        export_dataframe_csv(motion_df, config.REPORTS_DIR / "task4_motion_comparison.csv")
        print(" Task 4 Completed: Differencing, MOG2, and Lucas-Kanade benchmarked.")
    else:
        print(" Warning: Video file not found, skipping Task 4 sample execution.")

    print("\n" + "=" * 60)
    print("✅ All Task Benchmarks Executed and Outputs Populated Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

