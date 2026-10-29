"""
Visualization utilities for generating comparative figures, galleries, and plots.
"""

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np


def create_figure_restoration(
    clean: np.ndarray,
    noisy: np.ndarray,
    median_restored: np.ndarray,
    gaussian_restored: np.ndarray,
    title_suffix: str = ""
) -> plt.Figure:
    """Generate 2x2 comparison figure of restoration filters."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.patch.set_facecolor('#0E1117')

    def setup_ax(ax, img, title):
        ax.set_facecolor('#0E1117')
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=8)
        ax.axis('off')

    setup_ax(axes[0, 0], clean, "1. Clean Reference Image")
    setup_ax(axes[0, 1], noisy, f"2. Noisy Image {title_suffix}")
    setup_ax(axes[1, 0], median_restored, "3. Median Filter Restored")
    setup_ax(axes[1, 1], gaussian_restored, "4. Gaussian Filter Restored")

    plt.tight_layout()
    return fig


def create_figure_morphology(
    original: np.ndarray,
    binary: np.ndarray,
    eroded: np.ndarray,
    dilated: np.ndarray,
    opened: np.ndarray,
    closed: np.ndarray
) -> plt.Figure:
    """Generate 2x3 panel of morphological operations."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.patch.set_facecolor('#0E1117')

    imgs = [
        (original, "Original Grayscale", 'gray'),
        (binary, "Otsu Binary Mask", 'gray'),
        (eroded, "Erosion (Shrink)", 'gray'),
        (dilated, "Dilation (Expand)", 'gray'),
        (opened, "Opening (Erode->Dilate)", 'gray'),
        (closed, "Closing (Dilate->Erode)", 'gray'),
    ]

    for ax, (im, title, cmap) in zip(axes.flat, imgs):
        ax.set_facecolor('#0E1117')
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=6)
        ax.axis('off')

    plt.tight_layout()
    return fig


def create_figure_watershed(
    original_rgb: np.ndarray,
    binary_mask: np.ndarray,
    dist_transform_vis: np.ndarray,
    sure_fg: np.ndarray,
    segmented_colored: np.ndarray,
    final_overlay: np.ndarray
) -> plt.Figure:
    """Generate 2x3 progression of Marker-Controlled Watershed pipeline."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.patch.set_facecolor('#0E1117')

    steps = [
        (original_rgb, "1. Input RGB Image", None),
        (binary_mask, "2. Threshold Mask", 'gray'),
        (dist_transform_vis, "3. Distance Transform (Jet)", None),
        (sure_fg, "4. Sure Foreground Markers", 'gray'),
        (segmented_colored, "5. Segment Labels", None),
        (final_overlay, "6. Watershed Boundaries (Red)", None)
    ]

    for ax, (im, title, cmap) in zip(axes.flat, steps):
        ax.set_facecolor('#0E1117')
        if cmap:
            ax.imshow(im, cmap=cmap)
        else:
            ax.imshow(im)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=6)
        ax.axis('off')

    plt.tight_layout()
    return fig


def create_figure_detection_comparison(
    orig_rgb: np.ndarray,
    haar_rgb: np.ndarray,
    yolo_rgb: np.ndarray
) -> plt.Figure:
    """Generate side-by-side detection comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0E1117')

    panels = [
        (orig_rgb, "Input Photograph"),
        (haar_rgb, "Viola-Jones (Haar Cascade)"),
        (yolo_rgb, "Ultralytics YOLO (Deep CNN)")
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.set_facecolor('#0E1117')
        ax.imshow(img)
        ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=8)
        ax.axis('off')

    plt.tight_layout()
    return fig

