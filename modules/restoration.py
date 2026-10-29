"""
Task 1 (Part A): Image Noise Injection and Restoration Filters.
Implements Gaussian noise, Salt & Pepper noise, Median filtering, and Gaussian filtering
with quantitative metric evaluation (PSNR and SSIM) against clean reference images.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
import pandas as pd
from modules.metrics import evaluate_restoration_quality, BenchmarkTimer


def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    sigma: float = 25.0,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Add additive Gaussian white noise to an image.
    Formula: I_noisy(x, y) = clip(I_clean(x, y) + N(mean, sigma^2), 0, 255)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    gauss = rng.normal(mean, sigma, image.shape)
    noisy = image.astype(np.float64) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(
    image: np.ndarray,
    amount: float = 0.05,
    salt_vs_pepper: float = 0.5,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Add impulsive Salt-and-Pepper noise to an image.
    - amount: proportion of image pixels replaced by noise (0.0 to 1.0)
    - salt_vs_pepper: ratio of salt (white=255) vs pepper (black=0)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    noisy = image.copy()
    h, w = image.shape[:2]
    num_noisy_pixels = int(amount * h * w)

    # Generate random coordinates
    coords_y = rng.integers(0, h, num_noisy_pixels)
    coords_x = rng.integers(0, w, num_noisy_pixels)

    # Salt (white pixels)
    num_salt = int(num_noisy_pixels * salt_vs_pepper)
    if len(image.shape) == 3:
        noisy[coords_y[:num_salt], coords_x[:num_salt], :] = 255
        noisy[coords_y[num_salt:], coords_x[num_salt:], :] = 0
    else:
        noisy[coords_y[:num_salt], coords_x[:num_salt]] = 255
        noisy[coords_y[num_salt:], coords_x[num_salt]] = 0

    return noisy


def apply_gaussian_filter(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply linear Gaussian low-pass spatial smoothing filter.
    Kernel size must be a positive odd integer.
    """
    # Enforce odd kernel
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)


def apply_median_filter(
    image: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Apply non-linear Median rank filter.
    Highly effective for impulsive (salt-and-pepper) noise reduction without blurring edges.
    Kernel size must be a positive odd integer.
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.medianBlur(image, k)


def run_restoration_comparison(
    clean_image: np.ndarray,
    noise_type: str = "gaussian",
    noise_param: float = 25.0,
    gaussian_ksize: int = 5,
    gaussian_sigma: float = 1.2,
    median_ksize: int = 5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Perform a complete restoration pipeline experiment:
    1. Injects selected noise (Gaussian or Salt & Pepper).
    2. Restores using Median filter.
    3. Restores using Gaussian filter.
    4. Computes fair quantitative metrics (PSNR, SSIM, MSE, Runtime) against original clean image.

    Returns:
        (noisy_img, median_restored, gaussian_restored, metrics_df)
    """
    # 1. Inject noise
    if noise_type.lower() == "gaussian":
        noisy_img = add_gaussian_noise(clean_image, mean=0.0, sigma=noise_param, seed=seed)
        noise_label = f"Gaussian Noise (\u03c3={noise_param})"
    else:
        noisy_img = add_salt_and_pepper_noise(clean_image, amount=noise_param, salt_vs_pepper=0.5, seed=seed)
        noise_label = f"Salt & Pepper Noise (amount={noise_param*100:.1f}%)"

    # Evaluate noisy baseline
    noisy_metrics = evaluate_restoration_quality(clean_image, noisy_img, label=f"Noisy ({noise_label})")
    noisy_metrics["Runtime (ms)"] = 0.0

    # 2. Median Filter
    with BenchmarkTimer("Median Filter") as t_med:
        median_restored = apply_median_filter(noisy_img, kernel_size=median_ksize)
    med_metrics = evaluate_restoration_quality(
        clean_image, median_restored, label=f"Median Filter (k={median_ksize}x{median_ksize})"
    )
    med_metrics["Runtime (ms)"] = round(t_med.elapsed_ms, 2)

    # 3. Gaussian Filter
    with BenchmarkTimer("Gaussian Filter") as t_gauss:
        gaussian_restored = apply_gaussian_filter(noisy_img, kernel_size=gaussian_ksize, sigma=gaussian_sigma)
    gauss_metrics = evaluate_restoration_quality(
        clean_image, gaussian_restored, label=f"Gaussian Filter (k={gaussian_ksize}x{gaussian_ksize}, \u03c3={gaussian_sigma})"
    )
    gauss_metrics["Runtime (ms)"] = round(t_gauss.elapsed_ms, 2)

    # Clean Reference baseline
    clean_metrics = evaluate_restoration_quality(clean_image, clean_image, label="Original Clean Image (Reference)")
    clean_metrics["Runtime (ms)"] = 0.0

    records = [clean_metrics, noisy_metrics, med_metrics, gauss_metrics]
    df = pd.DataFrame(records)

    return noisy_img, median_restored, gaussian_restored, df

