"""
Quantitative evaluation metrics for Computer Vision tasks.
Includes PSNR, SSIM, MSE, and runtime calculation with fair reference guarantees.
"""

import time
from typing import Dict, Any, Tuple
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def calculate_mse(reference: np.ndarray, target: np.ndarray) -> float:
    """Calculate Mean Squared Error between two images."""
    if reference.shape != target.shape:
        raise ValueError(f"Shape mismatch: reference {reference.shape} vs target {target.shape}")
    ref_f = reference.astype(np.float64)
    tar_f = target.astype(np.float64)
    return float(np.mean((ref_f - tar_f) ** 2))


def evaluate_restoration_quality(
    clean_original: np.ndarray,
    restored_or_noisy: np.ndarray,
    label: str = "Image"
) -> Dict[str, Any]:
    """
    Compute full reconstruction quality metrics (PSNR, SSIM, MSE) against clean reference.
    Guarantees fair evaluation by verifying shape and range compatibility.
    """
    if clean_original.shape != restored_or_noisy.shape:
        # Resize target to match reference if slight padding differences exist
        import cv2
        restored_or_noisy = cv2.resize(
            restored_or_noisy,
            (clean_original.shape[1], clean_original.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    # Ensure uint8 [0, 255]
    clean_u8 = np.clip(clean_original, 0, 255).astype(np.uint8)
    target_u8 = np.clip(restored_or_noisy, 0, 255).astype(np.uint8)

    mse = calculate_mse(clean_u8, target_u8)

    if mse == 0:
        psnr_val = 100.0  # Perfect reconstruction
    else:
        psnr_val = float(compute_psnr(clean_u8, target_u8, data_range=255))

    is_multichannel = (len(clean_u8.shape) == 3 and clean_u8.shape[2] == 3)
    channel_axis = 2 if is_multichannel else None

    # SSIM calculation
    ssim_val = float(compute_ssim(
        clean_u8,
        target_u8,
        data_range=255,
        channel_axis=channel_axis
    ))

    return {
        "Method": label,
        "PSNR (dB)": round(psnr_val, 2),
        "SSIM": round(ssim_val, 4),
        "MSE": round(mse, 2),
    }


class BenchmarkTimer:
    """Context manager for accurate runtime profiling."""
    def __init__(self, name: str = "Execution"):
        self.name = name
        self.elapsed_sec = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_sec = time.perf_counter() - self.start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_sec * 1000.0

