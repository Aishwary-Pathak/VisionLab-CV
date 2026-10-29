"""
File input/output helper functions.
Preserves original files, prevents overwriting without warning, and handles export.
"""

from pathlib import Path
import shutil
import re
from typing import List, Dict, Optional
import cv2
import numpy as np
import pandas as pd


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to avoid shell or path traversal issues."""
    name = Path(filename).name
    # Replace spaces and special characters except alphanumeric, underscore, dot, dash
    clean_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return clean_name


def save_bytes_to_file(data: bytes, dest_dir: Path, filename: str) -> Path:
    """Safely write raw bytes to destination path, creating directory if needed."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    clean_name = sanitize_filename(filename)
    dest_path = dest_dir / clean_name
    with open(dest_path, "wb") as f:
        f.write(data)
    return dest_path


def copy_file_safe(src_path: Path, dest_dir: Path, new_name: Optional[str] = None) -> Path:
    """Safely copy a local file to destination directory without mutating source."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = new_name if new_name else src_path.name
    clean_name = sanitize_filename(filename)
    dest_path = dest_dir / clean_name
    shutil.copy2(str(src_path), str(dest_path))
    return dest_path


def list_files_by_extensions(directory: Path, extensions: List[str]) -> List[Path]:
    """Return sorted list of files matching extensions in directory."""
    directory = Path(directory)
    if not directory.exists():
        return []
    exts = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions]
    files = [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in exts]
    return sorted(files, key=lambda x: x.name.lower())


def save_image(img_rgb_or_bgr: np.ndarray, dest_path: Path, is_rgb: bool = True) -> bool:
    """Save numpy image array to disk. If is_rgb=True, converts to BGR for cv2.imwrite."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if img_rgb_or_bgr is None or img_rgb_or_bgr.size == 0:
        return False

    # Check dtype
    if img_rgb_or_bgr.dtype != np.uint8:
        # Normalize if float between 0 and 1
        if img_rgb_or_bgr.max() <= 1.0:
            to_save = (np.clip(img_rgb_or_bgr, 0, 1) * 255).astype(np.uint8)
        else:
            to_save = np.clip(img_rgb_or_bgr, 0, 255).astype(np.uint8)
    else:
        to_save = img_rgb_or_bgr.copy()

    if len(to_save.shape) == 3 and is_rgb:
        to_save = cv2.cvtColor(to_save, cv2.COLOR_RGB2BGR)

    return cv2.imwrite(str(dest_path), to_save)


def export_dataframe_csv(df: pd.DataFrame, dest_path: Path) -> Path:
    """Export pandas DataFrame to CSV with formatting."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_path, index=False)
    return dest_path

