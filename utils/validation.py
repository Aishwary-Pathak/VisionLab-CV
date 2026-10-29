"""
Input validation utilities for images and videos.
Performs verification of file formats, dimensions, channels, frame rates, and durations.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


def validate_image_file(image_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate that an image exists, is readable by OpenCV, and has valid dimensions.

    Returns:
        (is_valid, error_or_success_message, metadata_dict)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return False, f"Image file not found: {image_path.name}", {}

    # Read image using OpenCV
    img = cv2.imread(str(image_path))
    if img is None or img.size == 0:
        return False, f"Could not decode image file '{image_path.name}'. Format may be corrupt or unsupported.", {}

    h, w = img.shape[:2]
    channels = 1 if len(img.shape) == 2 else img.shape[2]
    file_size_kb = round(image_path.stat().st_size / 1024, 2)

    metadata = {
        "filename": image_path.name,
        "width": int(w),
        "height": int(h),
        "channels": int(channels),
        "size_kb": file_size_kb,
        "aspect_ratio": round(w / h, 2) if h > 0 else 0,
        "dtype": str(img.dtype),
    }

    if w < 32 or h < 32:
        return False, f"Image dimensions ({w}x{h}) are too small for CV tasks (minimum 32x32 required).", metadata

    return True, f"Image successfully validated: {w}x{h}, {channels} channels, {file_size_kb} KB", metadata


def validate_video_file(video_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate that a video exists, is readable by OpenCV, has valid FPS and frame count.

    Returns:
        (is_valid, error_or_success_message, metadata_dict)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return False, f"Video file not found: {video_path.name}", {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, f"Unable to open video stream: '{video_path.name}'. Codec may be unsupported by OpenCV.", {}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the first frame to confirm stream validity
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return False, f"Failed to read any frames from video: '{video_path.name}'", {}

    fps = fps if (fps and fps > 0) else 30.0
    duration_sec = round(total_frames / fps, 2) if (fps > 0 and total_frames > 0) else 0.0
    file_size_mb = round(video_path.stat().st_size / (1024 * 1024), 2)

    metadata = {
        "filename": video_path.name,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "file_size_mb": file_size_mb,
    }

    if width < 64 or height < 64:
        return False, f"Video resolution ({width}x{height}) is too small for motion analysis.", metadata

    return True, f"Video validated: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames ({duration_sec}s, {file_size_mb} MB)", metadata

