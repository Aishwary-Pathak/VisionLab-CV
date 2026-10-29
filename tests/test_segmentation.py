"""
Unit tests for Task 1 (Part B): Morphology, Canny Edge, and Watershed Segmentation.
"""

import numpy as np
import pytest
from modules.segmentation import (
    to_grayscale,
    apply_thresholding,
    apply_morphology,
    apply_canny_edge,
    apply_watershed_segmentation
)


def create_binary_circle_image(size=120):
    """Create test binary image with two touching circles."""
    img = np.zeros((size, size), dtype=np.uint8)
    import cv2
    cv2.circle(img, (40, 60), 25, 255, -1)
    cv2.circle(img, (80, 60), 25, 255, -1)
    return img


def test_to_grayscale():
    rgb = np.zeros((50, 50, 3), dtype=np.uint8)
    gray = to_grayscale(rgb)
    assert len(gray.shape) == 2
    assert gray.shape == (50, 50)


def test_thresholding_methods():
    # Create test pattern with two distinct intensity modes (0 and 200)
    test_pattern = np.zeros((100, 100), dtype=np.uint8)
    test_pattern[20:80, 20:80] = 200

    # Otsu
    otsu_mask, otsu_val = apply_thresholding(test_pattern, method="otsu")
    assert otsu_mask.shape == test_pattern.shape
    assert otsu_val >= 0.0

    # Manual
    manual_mask, m_val = apply_thresholding(test_pattern, method="manual", manual_thresh=100)
    assert manual_mask.shape == test_pattern.shape
    assert m_val == 100.0


def test_morphological_operations():
    circle = create_binary_circle_image()
    eroded = apply_morphology(circle, "erosion", kernel_size=3)
    dilated = apply_morphology(circle, "dilation", kernel_size=3)
    opened = apply_morphology(circle, "opening", kernel_size=3)
    closed = apply_morphology(circle, "closing", kernel_size=3)

    # Eroded area must be strictly less than dilated area
    assert np.count_nonzero(eroded) < np.count_nonzero(dilated)
    assert opened.shape == circle.shape
    assert closed.shape == circle.shape


def test_canny_edge():
    circle = create_binary_circle_image()
    edges = apply_canny_edge(circle, low_thresh=50, high_thresh=150)
    assert edges.shape == circle.shape
    assert np.any(edges == 255)


def test_watershed_segmentation():
    circle_gray = create_binary_circle_image(120)
    circle_rgb = np.stack([circle_gray]*3, axis=-1)
    results = apply_watershed_segmentation(circle_rgb, dist_thresh_ratio=0.4, invert_binary=False)

    assert "binary_mask" in results
    assert "sure_bg" in results
    assert "sure_fg" in results
    assert "segmented_overlay" in results
    assert "num_objects" in results
    assert results["num_objects"] >= 1
