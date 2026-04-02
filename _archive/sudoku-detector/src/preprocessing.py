"""Image preprocessing functions for sudoku grid detection."""

from typing import Optional, Tuple

import cv2
import numpy as np

from .config import DetectionConfig


def load_and_validate(
    filepath: str, min_size: int = 200
) -> Optional[np.ndarray]:
    """Load an image from disk and validate it meets minimum requirements.

    Args:
        filepath: Path to the image file.
        min_size: Minimum dimension (width or height) required.

    Returns:
        The loaded image as a numpy array (BGR format), or None if:
        - File cannot be read
        - Image is too small (either dimension < min_size)
    """
    img = cv2.imread(filepath)

    if img is None:
        return None

    height, width = img.shape[:2]
    if height < min_size or width < min_size:
        return None

    return img


def resize_image(
    img: np.ndarray, target_size: int
) -> Tuple[np.ndarray, float]:
    """Resize image to fit within target size while preserving aspect ratio.

    The image is only resized if its largest dimension exceeds target_size.
    The aspect ratio is always preserved.

    Args:
        img: Input image (BGR or grayscale).
        target_size: Maximum dimension (width or height) for the output.

    Returns:
        Tuple of (resized_image, scale_factor) where scale_factor is the
        ratio of new size to original size (1.0 if no resize needed).
    """
    height, width = img.shape[:2]
    max_dim = max(height, width)

    if max_dim <= target_size:
        return img.copy(), 1.0

    scale_factor = target_size / max_dim
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    resized = cv2.resize(
        img, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    return resized, scale_factor


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale, handling already-gray images.

    Args:
        img: Input image (BGR or grayscale).

    Returns:
        Grayscale image.
    """
    if len(img.shape) == 2:
        return img.copy()

    if img.shape[2] == 1:
        return img.squeeze(axis=2).copy()

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    CLAHE improves local contrast and handles variable lighting conditions.

    Args:
        gray: Grayscale input image.
        clip_limit: Threshold for contrast limiting.
        tile_size: Size of grid for histogram equalization.

    Returns:
        CLAHE-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)


def adaptive_threshold(
    img: np.ndarray, block_size: int = 11, c: int = 2
) -> np.ndarray:
    """Apply adaptive Gaussian thresholding to create binary image.

    Uses THRESH_BINARY_INV so that grid lines appear WHITE on black background.

    Args:
        img: Grayscale input image.
        block_size: Size of pixel neighborhood for threshold calculation.
            Must be odd.
        c: Constant subtracted from the mean.

    Returns:
        Binary image with grid lines as white (255) on black (0) background.
    """
    return cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )


def apply_morphological(
    binary: np.ndarray, kernel_size: int = 3
) -> np.ndarray:
    """Apply morphological closing to fill small gaps in lines.

    Closing is dilation followed by erosion, which fills small holes
    and gaps while preserving the overall shape.

    Args:
        binary: Binary input image.
        kernel_size: Size of the morphological kernel.

    Returns:
        Binary image after morphological closing.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def get_edges(
    img: np.ndarray,
    low: int = 50,
    high: int = 150,
    blur_size: int = 5,
) -> np.ndarray:
    """Apply Gaussian blur followed by Canny edge detection.

    Args:
        img: Grayscale input image.
        low: Lower threshold for Canny hysteresis.
        high: Upper threshold for Canny hysteresis.
        blur_size: Kernel size for Gaussian blur (must be odd).

    Returns:
        Binary edge image.
    """
    blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    return cv2.Canny(blurred, low, high)


def preprocess_for_contour(
    img: np.ndarray, config: DetectionConfig
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Full preprocessing pipeline for contour-based detection.

    Pipeline:
    1. Resize to target size
    2. Convert to grayscale
    3. Apply CLAHE
    4. Adaptive threshold
    5. Optional morphological closing

    Args:
        img: Input image (BGR).
        config: Detection configuration parameters.

    Returns:
        Tuple of (binary_image, resized_color_image, scale_factor).
        The binary image has grid lines as white on black.
        The resized color image is for later perspective correction.
    """
    resized, scale_factor = resize_image(img, config.target_size)
    gray = to_grayscale(resized)
    enhanced = apply_clahe(gray, config.clahe_clip_limit, config.clahe_tile_size)
    binary = adaptive_threshold(
        enhanced, config.adaptive_block_size, config.adaptive_c
    )

    if config.use_morphological:
        binary = apply_morphological(binary, config.morph_kernel_size)

    return binary, resized, scale_factor


def preprocess_for_hough(
    img: np.ndarray, config: DetectionConfig
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Full preprocessing pipeline for Hough line detection.

    Pipeline:
    1. Resize to target size
    2. Convert to grayscale
    3. Apply CLAHE
    4. Gaussian blur + Canny edge detection

    Args:
        img: Input image (BGR).
        config: Detection configuration parameters.

    Returns:
        Tuple of (edge_image, resized_color_image, scale_factor).
        The edge image contains detected edges for Hough transform.
        The resized color image is for later perspective correction.
    """
    resized, scale_factor = resize_image(img, config.target_size)
    gray = to_grayscale(resized)
    enhanced = apply_clahe(gray, config.clahe_clip_limit, config.clahe_tile_size)
    edges = get_edges(enhanced, config.canny_low, config.canny_high)

    return edges, resized, scale_factor
