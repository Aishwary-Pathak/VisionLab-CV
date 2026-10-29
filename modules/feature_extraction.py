"""
Task 2: Feature Extraction and Image Matching.
Implements:
1. Histogram of Oriented Gradients (HOG) extraction and gradient visualization.
2. Scale-Invariant Feature Transform (SIFT) keypoint detection and 128-D descriptor computation.
3. FLANN / Brute-Force feature matching with Lowe's ratio distance test.
4. Robust geometric verification using Homography with RANSAC outlier rejection.
5. Invariance experiments:
   - Original vs Second Image
   - Original vs Rotated Image
   - Original vs Resized Image
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import cv2
from skimage.feature import hog
from skimage import exposure
import pandas as pd
from modules.metrics import BenchmarkTimer
from modules.segmentation import to_grayscale


def extract_hog_features(
    image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (16, 16),
    cells_per_block: Tuple[int, int] = (2, 2)
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute Histogram of Oriented Gradients (HOG) descriptors and spatial visualization.
    Returns:
        (features_1d, hog_image_u8, info_dict)
    """
    gray = to_grayscale(image)

    with BenchmarkTimer("HOG Extraction") as timer:
        features, hog_vis = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True,
            channel_axis=None
        )

    # Enhance contrast of the gradient visualization
    hog_rescaled = exposure.rescale_intensity(hog_vis, in_range=(0, 10))
    hog_u8 = (hog_rescaled * 255).astype(np.uint8)
    hog_color = cv2.applyColorMap(hog_u8, cv2.COLORMAP_MAGMA)

    info = {
        "num_features": int(features.size),
        "orientations": orientations,
        "pixels_per_cell": f"{pixels_per_cell[0]}x{pixels_per_cell[1]}",
        "cells_per_block": f"{cells_per_block[0]}x{cells_per_block[1]}",
        "runtime_ms": round(timer.elapsed_ms, 2)
    }

    return features, hog_color, info


def detect_and_compute_sift(
    image: np.ndarray,
    n_features: int = 1000,
    contrast_thresh: float = 0.04,
    edge_thresh: float = 10.0,
    sigma: float = 1.6
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray], np.ndarray, float]:
    """
    Detect SIFT keypoints and compute 128-dimensional invariant descriptors.
    Returns:
        (keypoints, descriptors, visualized_image, runtime_ms)
    """
    gray = to_grayscale(image)
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_thresh,
        edgeThreshold=edge_thresh,
        sigma=sigma
    )

    with BenchmarkTimer("SIFT Detect & Compute") as timer:
        keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Visualize keypoints with scale and orientation circles
    vis_img = image.copy()
    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)

    # Draw rich keypoints (size + orientation)
    vis_img = cv2.drawKeypoints(
        vis_img,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return keypoints, descriptors, vis_img, timer.elapsed_ms


def match_sift_descriptors(
    desc1: Optional[np.ndarray],
    desc2: Optional[np.ndarray],
    matcher_type: str = "flann",
    ratio_thresh: float = 0.75
) -> Tuple[List[cv2.DMatch], List[List[cv2.DMatch]], float]:
    """
    Match two sets of SIFT descriptors using k-NN with Lowe's ratio test.
    Returns:
        (good_matches, all_knn_matches, runtime_ms)
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return [], [], 0.0

    desc1_f = desc1.astype(np.float32)
    desc2_f = desc2.astype(np.float32)

    with BenchmarkTimer("Descriptor Matching") as timer:
        if matcher_type.lower() == "flann":
            # FLANN parameters for SIFT (KDTree)
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        knn_matches = matcher.knnMatch(desc1_f, desc2_f, k=2)

        # Apply Lowe's Ratio Test
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

    return good_matches, knn_matches, timer.elapsed_ms


def verify_matches_homography(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    good_matches: List[cv2.DMatch],
    ransac_thresh: float = 5.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, float]:
    """
    Estimate Homography matrix using RANSAC to geometrically verify matches and reject outliers.
    Returns:
        (H_matrix, inliers_mask, num_inliers, inlier_ratio)
    """
    if len(good_matches) < 4:
        return None, None, 0, 0.0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    if mask is not None:
        inliers_count = int(np.sum(mask))
        inlier_ratio = round(inliers_count / len(good_matches), 3) if len(good_matches) > 0 else 0.0
    else:
        inliers_count = 0
        inlier_ratio = 0.0

    return H, mask, inliers_count, inlier_ratio


def draw_feature_matches(
    img1: np.ndarray,
    kp1: List[cv2.KeyPoint],
    img2: np.ndarray,
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    mask: Optional[np.ndarray] = None,
    max_to_draw: int = 100
) -> np.ndarray:
    """Draw matched feature lines side-by-side with inliers highlighted in green."""
    if mask is not None:
        matches_mask = mask.ravel().tolist()
    else:
        matches_mask = [1] * len(matches)

    # Limit drawing density to avoid clutter
    if len(matches) > max_to_draw:
        matches_subset = matches[:max_to_draw]
        matches_mask_subset = matches_mask[:max_to_draw]
    else:
        matches_subset = matches
        matches_mask_subset = matches_mask

    draw_params = dict(
        matchColor=(0, 255, 0),  # Inliers: green
        singlePointColor=(255, 0, 0),  # Keypoints: blue
        matchesMask=matches_mask_subset,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    matched_vis = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches_subset,
        None,
        **draw_params
    )
    return matched_vis


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image around its center by angle_deg degrees without truncation."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    # Compute new bounding dimensions
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Scale image uniformly by scale factor."""
    h, w = image.shape[:2]
    new_w = max(32, int(w * scale))
    new_h = max(32, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)


def run_single_matching_experiment(
    img1: np.ndarray,
    img2: np.ndarray,
    experiment_name: str,
    n_features: int = 1000,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 5.0
) -> Dict[str, Any]:
    """
    Run full SIFT detect -> compute -> match -> RANSAC pipeline between two images.
    Returns:
        Dict with keypoint visuals, match visualization, and metrics.
    """
    kp1, desc1, vis1, t_sift1 = detect_and_compute_sift(img1, n_features=n_features)
    kp2, desc2, vis2, t_sift2 = detect_and_compute_sift(img2, n_features=n_features)

    good_matches, all_matches, t_match = match_sift_descriptors(desc1, desc2, ratio_thresh=ratio_thresh)
    H, mask, inliers, inlier_ratio = verify_matches_homography(kp1, kp2, good_matches, ransac_thresh=ransac_thresh)

    vis_matches = draw_feature_matches(img1, kp1, img2, kp2, good_matches, mask=mask)

    total_time_ms = round(t_sift1 + t_sift2 + t_match, 2)

    metrics = {
        "Experiment": experiment_name,
        "Image 1 Keypoints": len(kp1),
        "Image 2 Keypoints": len(kp2),
        "Candidate Matches": len(all_matches),
        "Good Matches (Lowe's)": len(good_matches),
        "RANSAC Inliers": inliers,
        "Inlier Ratio": inlier_ratio,
        "Runtime (ms)": total_time_ms
    }

    return {
        "metrics": metrics,
        "kp1_vis": vis1,
        "kp2_vis": vis2,
        "matches_vis": vis_matches,
        "homography": H,
        "inliers_count": inliers,
        "good_matches_count": len(good_matches)
    }

