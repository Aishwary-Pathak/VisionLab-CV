"""
Task 1 (Part B): Morphological Operations, Edge Detection, and Watershed Segmentation.
Includes thresholding (Otsu/Manual/Adaptive), morphological kernels, Canny edge detection,
and distance-transform based marker Watershed segmentation for separating touching regions.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB or BGR image to single-channel 8-bit grayscale."""
    if len(image.shape) == 2:
        return image.copy()
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_thresholding(
    gray_image: np.ndarray,
    method: str = "otsu",
    manual_thresh: int = 127,
    adaptive_block_size: int = 11,
    adaptive_c: int = 2,
    invert: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Produce binary mask via Otsu's thresholding, manual global thresholding, or adaptive thresholding.
    Returns: (binary_mask, computed_threshold_value)
    """
    gray = to_grayscale(gray_image)
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    if method.lower() == "otsu":
        otsu_val, mask = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
        return mask, float(otsu_val)
    elif method.lower() == "adaptive_mean":
        bs = max(3, adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1)
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, thresh_type, bs, adaptive_c
        )
        return mask, -1.0
    elif method.lower() == "adaptive_gaussian":
        bs = max(3, adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1)
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, bs, adaptive_c
        )
        return mask, -1.0
    else:  # Manual
        _, mask = cv2.threshold(gray, manual_thresh, 255, thresh_type)
        return mask, float(manual_thresh)


def get_structuring_element(shape: str = "rect", size: int = 3) -> np.ndarray:
    """Create OpenCV structuring element of given shape (rect, ellipse, cross) and odd size."""
    ksize = max(1, size if size % 2 == 1 else size + 1)
    s_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS
    }
    morph_shape = s_map.get(shape.lower(), cv2.MORPH_RECT)
    return cv2.getStructuringElement(morph_shape, (ksize, ksize))


def apply_morphology(
    binary_mask: np.ndarray,
    operation: str = "opening",
    kernel_shape: str = "rect",
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Apply mathematical morphological operators:
    - erosion: shrinks bright foreground regions; removes small isolated noise.
    - dilation: expands bright foreground regions; fills tiny holes.
    - opening: erosion followed by dilation; eliminates small foreground objects/bridges.
    - closing: dilation followed by erosion; fills small holes and gaps within objects.
    - gradient: difference between dilation and erosion; extracts outer boundaries.
    """
    element = get_structuring_element(shape=kernel_shape, size=kernel_size)
    op = operation.lower()

    if op == "erosion":
        return cv2.erode(binary_mask, element, iterations=iterations)
    elif op == "dilation":
        return cv2.dilate(binary_mask, element, iterations=iterations)
    elif op == "opening":
        return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, element, iterations=iterations)
    elif op == "closing":
        return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, element, iterations=iterations)
    elif op == "gradient":
        return cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, element, iterations=iterations)
    else:
        raise ValueError(f"Unknown morphology operation '{operation}'")


def apply_canny_edge(
    image: np.ndarray,
    low_thresh: int = 50,
    high_thresh: int = 150,
    aperture_size: int = 3,
    l2_gradient: bool = True
) -> np.ndarray:
    """
    Canny Edge Detector with non-maximum suppression and hysteresis thresholding.
    - low_thresh: lower hysteresis threshold for weak edges.
    - high_thresh: upper hysteresis threshold for strong edges.
    - aperture_size: Sobel kernel size (must be 3, 5, or 7).
    """
    gray = to_grayscale(image)
    # Pre-smooth with small Gaussian blur to reduce high frequency noise
    smoothed = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(
        smoothed,
        threshold1=low_thresh,
        threshold2=high_thresh,
        apertureSize=aperture_size,
        L2gradient=l2_gradient
    )
    return edges


def apply_watershed_segmentation(
    rgb_image: np.ndarray,
    dist_thresh_ratio: float = 0.5,
    morph_kernel_size: int = 3,
    invert_binary: bool = True
) -> Dict[str, Any]:
    """
    Marker-controlled Watershed Segmentation algorithm:
    1. Grayscale & Otsu thresholding to produce binary foreground mask.
    2. Morphological opening to remove background speckle noise.
    3. Dilation to obtain sure background.
    4. Distance Transform on foreground to isolate object centers.
    5. Thresholding distance transform to produce sure foreground markers.
    6. Subtraction to find unknown border regions.
    7. Connected components labeling on sure foreground markers.
    8. Watershed flooding on RGB image.
    9. Boundary coloring and segmented visual overlay.

    Returns:
        Dictionary containing all intermediate representations and final segmented RGB.
    """
    gray = to_grayscale(rgb_image)
    # Pre-filter
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Otsu threshold
    thresh_flag = cv2.THRESH_BINARY_INV if invert_binary else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(blurred, 0, 255, thresh_flag + cv2.THRESH_OTSU)

    # 2. Morphological opening (noise removal)
    kernel = get_structuring_element("rect", morph_kernel_size)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Sure background via dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 4. Distance transform for sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_dist = dist_transform.max()

    # 5. Threshold distance transform to isolate sure foreground centers
    if max_dist > 0:
        _, sure_fg = cv2.threshold(dist_transform, dist_thresh_ratio * max_dist, 255, 0)
    else:
        sure_fg = np.zeros_like(opening)
    sure_fg = np.uint8(sure_fg)

    # 6. Unknown region = sure_bg - sure_fg
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7. Marker labeling
    num_markers, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so sure background is not 0, but 1
    markers = markers + 1
    # Mark unknown region with 0
    markers[unknown == 255] = 0

    # 8. Apply Watershed
    # OpenCV watershed requires BGR/RGB image in 8-bit
    watershed_img = rgb_image.copy()
    markers = cv2.watershed(watershed_img, markers)

    # 9. Create color segmented visualization
    # Color each region distinctly
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_markers + 2, 3), dtype=np.uint8)
    colors[1] = [40, 40, 40]  # Background dark grey

    segmented_colored = np.zeros_like(rgb_image)
    for label_idx in range(1, num_markers + 1):
        segmented_colored[markers == label_idx] = colors[label_idx]

    # Draw boundaries: watershed marks boundaries as -1
    result_overlay = rgb_image.copy()
    boundary_mask = (markers == -1)
    result_overlay[boundary_mask] = [255, 0, 0]  # Bright red boundaries
    segmented_colored[boundary_mask] = [255, 255, 255]

    # Normalize distance transform for clean visualization
    dist_display = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dist_display_color = cv2.applyColorMap(dist_display, cv2.COLORMAP_JET)

    # Diagnose suitability
    detected_objects = max(0, num_markers - 1)
    is_suitable = 1 <= detected_objects <= 50
    suitability_msg = (
        f"Detected {detected_objects} segmented regions. "
        "Separation completed successfully." if is_suitable else
        f"Image produced {detected_objects} regions. Complex textured scenes (like mountains/trees) "
        "often over-segment in Watershed without strong marker priors. High-contrast objects (coins, cells, pebbles) produce optimal results."
    )

    return {
        "binary_mask": thresh,
        "morph_clean": opening,
        "sure_bg": sure_bg,
        "dist_transform_vis": dist_display_color,
        "sure_fg": sure_fg,
        "unknown": unknown,
        "markers_initial": markers.copy(),
        "segmented_overlay": result_overlay,
        "segmented_colored": segmented_colored,
        "num_objects": detected_objects,
        "suitability_note": suitability_msg,
    }

