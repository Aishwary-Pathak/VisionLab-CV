"""
Unit tests for Task 2: Feature Extraction and Matching (HOG, SIFT, Homography/RANSAC).
"""

import numpy as np
import cv2
import pytest
from modules.feature_extraction import (
    extract_hog_features,
    detect_and_compute_sift,
    match_sift_descriptors,
    verify_matches_homography,
    rotate_image,
    resize_image,
    run_single_matching_experiment
)


def create_feature_rich_image(size=150):
    """Create a rich geometric image with corners, lines and text."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (70, 70), (255, 255, 255), -1)
    cv2.circle(img, (110, 110), 25, (180, 180, 180), -1)
    cv2.line(img, (10, 140), (140, 10), (200, 200, 200), 3)
    cv2.putText(img, "CV", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return img


def test_hog_extraction():
    img = create_feature_rich_image()
    features, hog_vis, info = extract_hog_features(img, orientations=9, pixels_per_cell=(16, 16))
    assert features.ndim == 1
    assert len(features) > 0
    assert hog_vis.shape[:2] == img.shape[:2]
    assert "num_features" in info


def test_sift_detection_and_descriptor():
    img = create_feature_rich_image()
    kp, desc, vis, elapsed = detect_and_compute_sift(img, n_features=500)
    assert len(kp) > 0
    assert desc is not None
    assert desc.shape[1] == 128  # 128-D SIFT descriptor
    assert vis.shape == img.shape


def test_image_transformations():
    img = create_feature_rich_image()
    rot = rotate_image(img, 45)
    assert rot.shape[0] > 0 and rot.shape[1] > 0

    resized = resize_image(img, 0.5)
    assert resized.shape[0] == img.shape[0] // 2
    assert resized.shape[1] == img.shape[1] // 2


def test_sift_matching_and_homography():
    img1 = create_feature_rich_image()
    # Create slight variation
    img2 = rotate_image(img1, 15)

    kp1, desc1, _, _ = detect_and_compute_sift(img1)
    kp2, desc2, _, _ = detect_and_compute_sift(img2)

    good_matches, _, _ = match_sift_descriptors(desc1, desc2, ratio_thresh=0.8)
    assert len(good_matches) > 0

    H, mask, inliers, ratio = verify_matches_homography(kp1, kp2, good_matches)
    assert inliers >= 0
    assert 0.0 <= ratio <= 1.0


def test_full_matching_experiment():
    img1 = create_feature_rich_image()
    img2 = resize_image(img1, 0.8)

    res = run_single_matching_experiment(img1, img2, "Resized 0.8x")
    assert "metrics" in res
    assert "matches_vis" in res
    assert res["metrics"]["Good Matches (Lowe's)"] >= 0

