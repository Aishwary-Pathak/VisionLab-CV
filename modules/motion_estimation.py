"""
Task 4: Video Motion Estimation & Tracking.
Implements:
1. Video stream ingestion and controlled temporal sampling.
2. Consecutive Frame Differencing with binary motion mask generation.
3. GMM-based Background Subtraction (OpenCV MOG2) with shadow separation.
4. Lucas-Kanade Pyramidal Sparse Optical Flow with feature tracking and motion vector trails.
5. Motion analytics and comparative evaluation across all three paradigms.
"""

from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from modules.metrics import BenchmarkTimer
from modules.segmentation import to_grayscale


def extract_sampled_frames(
    video_path: Path,
    max_frames: int = 100,
    sampling_step: int = 2
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Read video stream and extract sampled RGB frames for motion analysis.
    Safely limits frame count to prevent memory exhaustion on long videos.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx % sampling_step == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            if len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()

    meta = {
        "video_name": video_path.name,
        "width": width,
        "height": height,
        "original_fps": round(fps, 1),
        "total_source_frames": total_frames,
        "sampled_frames_count": len(frames),
        "sampling_step": sampling_step,
        "duration_sec": round(total_frames / fps, 2) if fps > 0 else 0.0
    }

    return frames, meta


def compute_frame_differencing(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold: int = 30
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Temporal Frame Differencing between consecutive frames t-1 and t.
    Returns:
        (diff_gray, binary_motion_mask, motion_percentage)
    """
    gray_prev = to_grayscale(prev_frame)
    gray_curr = to_grayscale(curr_frame)

    # 1. Compute absolute pixel difference
    diff = cv2.absdiff(gray_prev, gray_curr)

    # 2. Smooth slightly to eliminate high-frequency camera noise
    blurred_diff = cv2.GaussianBlur(diff, (5, 5), 0)

    # 3. Threshold to binary mask
    _, mask = cv2.threshold(blurred_diff, threshold, 255, cv2.THRESH_BINARY)

    # 4. Clean with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    motion_pixels = np.count_nonzero(clean_mask)
    total_pixels = clean_mask.size
    motion_pct = round((motion_pixels / total_pixels) * 100.0, 2)

    return diff, clean_mask, motion_pct


def run_mog2_background_subtraction(
    frames: List[np.ndarray],
    history: int = 120,
    var_threshold: float = 16.0,
    detect_shadows: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """
    Gaussian Mixture Model (MOG2) Background Subtraction across sequence.
    Detects moving foreground while distinguishing illumination shadows.
    Returns:
        (foreground_masks, visual_overlays, avg_latency_ms)
    """
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows
    )

    fg_masks = []
    overlays = []
    times = []

    for frame in frames:
        with BenchmarkTimer("MOG2 Step") as t:
            # OpenCV background subtractors expect BGR or RGB
            mask = subtractor.apply(frame)

        times.append(t.elapsed_ms)
        fg_masks.append(mask)

        # Create overlay: foreground in green, shadow in grey
        overlay = frame.copy()
        # Shadows are marked with value 127 when detectShadows=True
        shadow_pixels = (mask == 127)
        fg_pixels = (mask == 255)

        overlay[shadow_pixels] = [100, 100, 200]  # Blue-tinted shadow
        overlay[fg_pixels] = [0, 255, 0]          # Bright green foreground
        blended = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
        overlays.append(blended)

    avg_ms = float(np.mean(times)) if times else 0.0
    return fg_masks, overlays, avg_ms


def run_lucas_kanade_optical_flow(
    frames: List[np.ndarray],
    max_corners: int = 150,
    quality_level: float = 0.03,
    min_distance: int = 7,
    win_size: int = 15
) -> Tuple[List[np.ndarray], float, Dict[str, Any]]:
    """
    Lucas-Kanade Pyramidal Sparse Optical Flow tracker.
    Tracks Shi-Tomasi corner points through consecutive frames and renders motion vector vectors.
    """
    if len(frames) < 2:
        return frames, 0.0, {}

    feature_params = dict(
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7
    )

    lk_params = dict(
        winSize=(win_size, win_size),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Initial frame points
    prev_gray = to_grayscale(frames[0])
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # Persistent mask canvas for drawing motion trails
    h, w = frames[0].shape[:2]
    canvas_mask = np.zeros_like(frames[0])

    flow_frames = [frames[0].copy()]
    times = []
    tracked_counts = []
    displacements = []

    # Assign distinct colors to tracks
    np.random.seed(42)
    colors = np.random.randint(0, 255, (max_corners + 50, 3)).tolist()

    for i in range(1, len(frames)):
        curr_frame = frames[i]
        curr_gray = to_grayscale(curr_frame)

        with BenchmarkTimer("LK Flow Step") as t:
            if p0 is not None and len(p0) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, p0, None, **lk_params
                )
            else:
                p1, st = None, None

        times.append(t.elapsed_ms)

        # Select good points
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            tracked_counts.append(len(good_new))

            # Draw tracking trails and flow vectors
            display_frame = curr_frame.copy()
            for j, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                col = colors[j % len(colors)]

                # Motion magnitude
                dx = a - c
                dy = b - d
                mag = np.hypot(dx, dy)
                displacements.append(mag)

                # Draw trajectory vector line
                cv2.line(canvas_mask, (a, b), (c, d), col, 2)
                cv2.circle(display_frame, (a, b), 4, col, -1)

            # Combine canvas trail with current frame
            combined = cv2.add(display_frame, canvas_mask)
            flow_frames.append(combined)

            # Re-seed if feature count dropped below 10
            if len(good_new) < 15:
                p0 = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
            else:
                p0 = good_new.reshape(-1, 1, 2)
        else:
            flow_frames.append(curr_frame.copy())
            tracked_counts.append(0)
            p0 = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)

        prev_gray = curr_gray.copy()

    avg_ms = float(np.mean(times)) if times else 0.0
    avg_speed = float(np.mean(displacements)) if displacements else 0.0

    stats = {
        "avg_points_tracked": round(float(np.mean(tracked_counts)), 1) if tracked_counts else 0,
        "max_points_tracked": max(tracked_counts) if tracked_counts else 0,
        "avg_displacement_px": round(avg_speed, 2),
        "total_displacements_computed": len(displacements)
    }

    return flow_frames, avg_ms, stats


def build_motion_comparison_table(
    differencing_ms: float,
    mog2_ms: float,
    lk_flow_ms: float
) -> pd.DataFrame:
    """
    Build structured comparison table evaluating the three motion estimation paradigms.
    """
    records = [
        {
            "Method": "Consecutive Frame Differencing",
            "Output Representation": "Binary Motion Mask (|I_t - I_{t-1}| > T)",
            "Avg Latency (ms/frame)": round(differencing_ms, 2),
            "Estimated FPS": round(1000.0 / differencing_ms, 1) if differencing_ms > 0 else "N/A",
            "Handling of Stationary Objects": "Fails (fades immediately to background once motionless)",
            "Shadow Handling": "Vulnerable (shadows treated as moving foreground)",
            "Camera Motion Sensitivity": "Extremely High (entire frame becomes motion mask)",
            "Primary Academic Application": "Fast initial motion trigger / temporal activity cue"
        },
        {
            "Method": "GMM Background Subtraction (MOG2)",
            "Output Representation": "Tri-state Mask (Background / Shadow / Foreground)",
            "Avg Latency (ms/frame)": round(mog2_ms, 2),
            "Estimated FPS": round(1000.0 / mog2_ms, 1) if mog2_ms > 0 else "N/A",
            "Handling of Stationary Objects": "Gradually absorbs stopped objects into background model",
            "Shadow Handling": "Robust (distinguishes chromaticity shift from luminance)",
            "Camera Motion Sensitivity": "High (requires static camera or global registration)",
            "Primary Academic Application": "Surveillance, traffic monitoring, occupancy counting"
        },
        {
            "Method": "Lucas-Kanade Sparse Optical Flow",
            "Output Representation": "Motion Velocity Vectors (u, v) at corner keypoints",
            "Avg Latency (ms/frame)": round(lk_flow_ms, 2),
            "Estimated FPS": round(1000.0 / lk_flow_ms, 1) if lk_flow_ms > 0 else "N/A",
            "Handling of Stationary Objects": "Stationary points yield zero-length flow vectors",
            "Shadow Handling": "Focuses on high-contrast textured corners, less distracted by diffuse shadows",
            "Camera Motion Sensitivity": "Can isolate camera motion via global vector field consensus",
            "Primary Academic Application": "Visual odometry, action recognition, object tracking"
        }
    ]
    return pd.DataFrame(records)

