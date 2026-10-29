"""
Unit tests for Task 4: Video Motion Estimation (Differencing, MOG2, Lucas-Kanade).
"""

import numpy as np
import pytest
from modules.motion_estimation import (
    compute_frame_differencing,
    run_mog2_background_subtraction,
    run_lucas_kanade_optical_flow,
    build_motion_comparison_table
)


def create_synthetic_moving_sequence(num_frames=6, h=100, w=100):
    """Generate simple synthetic sequence of a moving bright square."""
    frames = []
    import cv2
    for i in range(num_frames):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        # Static background object
        cv2.rectangle(f, (10, 10), (30, 30), (120, 120, 120), -1)
        # Moving foreground object (shifts 5px right each frame)
        x_pos = 20 + i * 8
        cv2.rectangle(f, (x_pos, 50), (x_pos + 20, 70), (255, 255, 255), -1)
        frames.append(f)
    return frames


def test_frame_differencing():
    frames = create_synthetic_moving_sequence(3)
    diff, mask, motion_pct = compute_frame_differencing(frames[0], frames[1], threshold=20)
    assert diff.shape == (100, 100)
    assert mask.shape == (100, 100)
    assert motion_pct > 0.0  # Motion was present


def test_mog2_background_subtraction():
    frames = create_synthetic_moving_sequence(6)
    masks, overlays, avg_ms = run_mog2_background_subtraction(frames, history=5)
    assert len(masks) == 6
    assert len(overlays) == 6
    assert avg_ms >= 0


def test_lucas_kanade_optical_flow():
    frames = create_synthetic_moving_sequence(6)
    flow_frames, avg_ms, stats = run_lucas_kanade_optical_flow(frames, max_corners=50)
    assert len(flow_frames) == 6
    assert avg_ms >= 0
    assert "avg_points_tracked" in stats


def test_build_motion_comparison_table():
    df = build_motion_comparison_table(5.2, 8.4, 12.1)
    assert len(df) == 3
    assert "Consecutive Frame Differencing" in df["Method"].values
    assert "GMM Background Subtraction (MOG2)" in df["Method"].values
    assert "Lucas-Kanade Sparse Optical Flow" in df["Method"].values

