"""
Unit tests for Task 1: Image Restoration and Quantitative Metrics.
"""

import numpy as np
import pytest
from modules.restoration import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    apply_gaussian_filter,
    apply_median_filter,
    run_restoration_comparison
)
from modules.metrics import evaluate_restoration_quality, calculate_mse


def create_synthetic_image(h=100, w=100):
    """Create reproducible synthetic checkerboard pattern with geometric gradient."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:50, :50] = [200, 50, 50]
    img[50:, :50] = [50, 200, 50]
    img[:50, 50:] = [50, 50, 200]
    img[50:, 50:] = [220, 220, 220]
    return img


def test_gaussian_noise_shape_and_range():
    clean = create_synthetic_image()
    noisy = add_gaussian_noise(clean, mean=0.0, sigma=20.0, seed=42)
    assert noisy.shape == clean.shape
    assert noisy.dtype == np.uint8
    assert np.all(noisy >= 0) and np.all(noisy <= 255)
    # Noisy image should differ from clean
    assert not np.array_equal(clean, noisy)


def test_salt_and_pepper_noise():
    clean = create_synthetic_image()
    noisy = add_salt_and_pepper_noise(clean, amount=0.1, salt_vs_pepper=0.5, seed=42)
    assert noisy.shape == clean.shape
    assert noisy.dtype == np.uint8
    # Should have some 0 and 255 pixels
    assert np.any(noisy == 0)
    assert np.any(noisy == 255)


def test_median_filter_effectiveness_on_salt_pepper():
    clean = create_synthetic_image()
    noisy = add_salt_and_pepper_noise(clean, amount=0.05, seed=42)
    restored_med = apply_median_filter(noisy, kernel_size=3)
    restored_gauss = apply_gaussian_filter(noisy, kernel_size=3, sigma=1.0)

    # Median filter should achieve lower MSE than Gaussian filter on impulse noise
    mse_med = calculate_mse(clean, restored_med)
    mse_gauss = calculate_mse(clean, restored_gauss)
    assert mse_med < mse_gauss, f"Expected median MSE ({mse_med}) < gaussian MSE ({mse_gauss})"


def test_evaluate_restoration_metrics():
    clean = create_synthetic_image()
    noisy = add_gaussian_noise(clean, sigma=15.0, seed=42)

    clean_metrics = evaluate_restoration_quality(clean, clean, "Clean")
    assert clean_metrics["PSNR (dB)"] == 100.0
    assert clean_metrics["SSIM"] == 1.0
    assert clean_metrics["MSE"] == 0.0

    noisy_metrics = evaluate_restoration_quality(clean, noisy, "Noisy")
    assert noisy_metrics["PSNR (dB)"] < 100.0
    assert noisy_metrics["SSIM"] < 1.0
    assert noisy_metrics["MSE"] > 0.0


def test_run_restoration_comparison_pipeline():
    clean = create_synthetic_image()
    noisy, med, gauss, df = run_restoration_comparison(
        clean, noise_type="gaussian", noise_param=20.0, gaussian_ksize=5, median_ksize=5
    )
    assert noisy.shape == clean.shape
    assert med.shape == clean.shape
    assert gauss.shape == clean.shape
    assert len(df) == 4
    assert "PSNR (dB)" in df.columns
    assert "SSIM" in df.columns

