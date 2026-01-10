"""
Image preprocessing module for Sudoku grid detection.

Each function is designed to be independently testable and returns
intermediate results suitable for visualization.

Mathematical operations:
- Grayscale: G = 0.299R + 0.587G + 0.114B
- Gaussian Blur: B = G * K_σ (convolution with Gaussian kernel)
- Adaptive Threshold: T_{i,j} = 255 if B_{i,j} < μ_{i,j} - C else 0
- Canny Edge: Gradient magnitude with non-max suppression and hysteresis
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessingResult:
    """Container for all preprocessing intermediate results."""

    original: np.ndarray
    grayscale: np.ndarray
    blurred: np.ndarray
    threshold: np.ndarray
    edges: np.ndarray


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale.

    Formula: G = 0.299R + 0.587G + 0.114B

    These weights reflect human eye sensitivity:
    - Green: Most sensitive (0.587)
    - Red: Medium (0.299)
    - Blue: Least sensitive (0.114)

    Args:
        image: BGR image (H x W x 3)

    Returns:
        Grayscale image (H x W)
    """
    if len(image.shape) == 2:
        return image  # Already grayscale

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0
) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise.

    Gaussian kernel formula:
    K_σ(x, y) = (1 / 2πσ²) * exp(-(x² + y²) / 2σ²)

    The blur helps:
    - Reduce high-frequency noise
    - Smooth out small imperfections
    - Prepare image for edge detection

    Args:
        image: Grayscale image (H x W)
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation (0 = auto-calculate from kernel size)

    Returns:
        Blurred image (H x W)
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 11,
    c: int = 2,
    method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
) -> np.ndarray:
    """
    Apply adaptive thresholding to create binary image.

    For each pixel (i, j):
    - Calculate local mean μ in a block_size x block_size window
    - T_{i,j} = 255 if image_{i,j} < μ_{i,j} - C else 0

    Adaptive thresholding handles:
    - Uneven lighting
    - Shadows across the image
    - Different paper colors

    Args:
        image: Grayscale image (H x W)
        block_size: Size of local neighborhood (must be odd)
        c: Constant subtracted from mean
        method: ADAPTIVE_THRESH_GAUSSIAN_C or ADAPTIVE_THRESH_MEAN_C

    Returns:
        Binary image (H x W) with values 0 or 255
    """
    return cv2.adaptiveThreshold(
        image, 255, method, cv2.THRESH_BINARY_INV, block_size, c
    )


def detect_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3
) -> np.ndarray:
    """
    Detect edges using Canny edge detector.

    Canny algorithm steps:
    1. Compute gradient magnitude and direction (Sobel)
    2. Non-maximum suppression (thin edges to 1 pixel)
    3. Hysteresis thresholding:
       - Strong edges: gradient > high_threshold (kept)
       - Weak edges: low_threshold < gradient < high_threshold
       - Weak edges kept only if connected to strong edges

    Args:
        image: Grayscale image (H x W)
        low_threshold: Lower bound for hysteresis
        high_threshold: Upper bound for hysteresis
        aperture_size: Size of Sobel kernel

    Returns:
        Edge map (H x W) with values 0 or 255
    """
    return cv2.Canny(image, low_threshold, high_threshold, apertureSize=aperture_size)


def preprocess_for_hough(
    image: np.ndarray,
    blur_kernel: int = 5,
    threshold_block: int = 11,
    threshold_c: int = 2
) -> np.ndarray:
    """
    Full preprocessing pipeline for Standard Hough Transform (HoughLinesP).

    Pipeline: Original → Grayscale → Blur → Adaptive Threshold

    Args:
        image: BGR image
        blur_kernel: Gaussian blur kernel size
        threshold_block: Adaptive threshold block size
        threshold_c: Adaptive threshold constant

    Returns:
        Binary threshold image ready for Hough line detection
    """
    gray = to_grayscale(image)
    blurred = apply_blur(gray, blur_kernel)
    thresh = adaptive_threshold(blurred, threshold_block, threshold_c)
    return thresh


def preprocess_for_hough_polar(
    image: np.ndarray,
    canny_low: int = 90,
    canny_high: int = 150,
    dilate_kernel_size: int = 3,
    erode_kernel_size: int = 5
) -> np.ndarray:
    """
    Preprocess image for Standard Hough Transform (polar form with HoughLines).

    Pipeline: Original → Grayscale → Canny → Dilate → Erode

    Why morphological operations:
    - Dilation (3x3): Expands edges, connecting broken line segments
    - Erosion (5x5): Shrinks edges, removing small noise artifacts
    - Order matters: dilate first to connect, then erode to clean up

    Args:
        image: BGR image
        canny_low: Canny low threshold (default 90)
        canny_high: Canny high threshold (default 150)
        dilate_kernel_size: Size of dilation kernel (default 3)
        erode_kernel_size: Size of erosion kernel (default 5)

    Returns:
        Edge map ready for cv2.HoughLines() detection
    """
    gray = to_grayscale(image)
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)

    # Morphological operations to enhance edges
    kernel_dilate = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilated = cv2.dilate(edges, kernel_dilate, iterations=1)

    kernel_erode = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    eroded = cv2.erode(dilated, kernel_erode, iterations=1)

    return eroded


@dataclass
class PolarPreprocessingResult:
    """Container for polar preprocessing intermediate results."""

    original: np.ndarray
    grayscale: np.ndarray
    edges: np.ndarray      # After Canny
    dilated: np.ndarray    # After dilation (connects gaps)
    eroded: np.ndarray     # After erosion (final output)


def preprocess_for_hough_polar_full(
    image: np.ndarray,
    canny_low: int = 90,
    canny_high: int = 150,
    dilate_kernel_size: int = 3,
    erode_kernel_size: int = 5
) -> PolarPreprocessingResult:
    """
    Run polar preprocessing and return all intermediate results.

    Useful for visualization and debugging.

    Args:
        image: BGR image
        canny_low: Canny low threshold
        canny_high: Canny high threshold
        dilate_kernel_size: Size of dilation kernel
        erode_kernel_size: Size of erosion kernel

    Returns:
        PolarPreprocessingResult with all intermediate images
    """
    gray = to_grayscale(image)
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)

    kernel_dilate = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilated = cv2.dilate(edges, kernel_dilate, iterations=1)

    kernel_erode = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    eroded = cv2.erode(dilated, kernel_erode, iterations=1)

    return PolarPreprocessingResult(
        original=image,
        grayscale=gray,
        edges=edges,
        dilated=dilated,
        eroded=eroded
    )


def preprocess_for_ght(
    image: np.ndarray,
    blur_kernel: int = 5,
    canny_low: int = 50,
    canny_high: int = 150
) -> np.ndarray:
    """
    Full preprocessing pipeline for Generalized Hough Transform.

    Pipeline: Original → Grayscale → Blur → Canny Edges

    GHT works better with edge maps than binary threshold because
    it uses gradient direction at each edge pixel.

    Args:
        image: BGR image
        blur_kernel: Gaussian blur kernel size
        canny_low: Canny low threshold
        canny_high: Canny high threshold

    Returns:
        Edge map ready for GHT
    """
    gray = to_grayscale(image)
    blurred = apply_blur(gray, blur_kernel)
    edges = detect_edges(blurred, canny_low, canny_high)
    return edges


def preprocess_full(
    image: np.ndarray,
    blur_kernel: int = 5,
    threshold_block: int = 11,
    threshold_c: int = 2,
    canny_low: int = 50,
    canny_high: int = 150
) -> PreprocessingResult:
    """
    Run full preprocessing pipeline and return all intermediate results.

    Useful for visualization and debugging.

    Args:
        image: BGR image
        blur_kernel: Gaussian blur kernel size
        threshold_block: Adaptive threshold block size
        threshold_c: Adaptive threshold constant
        canny_low: Canny low threshold
        canny_high: Canny high threshold

    Returns:
        PreprocessingResult with all intermediate images
    """
    gray = to_grayscale(image)
    blurred = apply_blur(gray, blur_kernel)
    thresh = adaptive_threshold(blurred, threshold_block, threshold_c)
    edges = detect_edges(blurred, canny_low, canny_high)

    return PreprocessingResult(
        original=image,
        grayscale=gray,
        blurred=blurred,
        threshold=thresh,
        edges=edges
    )


def compute_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradients using Sobel operator.

    Sobel kernels:
    Gx = [[-1, 0, 1],      Gy = [[-1, -2, -1],
          [-2, 0, 2],            [ 0,  0,  0],
          [-1, 0, 1]]            [ 1,  2,  1]]

    Used for:
    - Gradient direction in GHT
    - Edge orientation analysis

    Args:
        image: Grayscale image

    Returns:
        Tuple of (gradient_x, gradient_y) as float64 arrays
    """
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return gx, gy


def compute_gradient_direction(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Compute gradient direction at each pixel.

    θ = arctan2(gy, gx)

    Result is in radians, range [-π, π]

    Args:
        gx: Gradient in x direction
        gy: Gradient in y direction

    Returns:
        Gradient direction in radians
    """
    return np.arctan2(gy, gx)


def compute_gradient_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Compute gradient magnitude at each pixel.

    |G| = sqrt(gx² + gy²)

    Args:
        gx: Gradient in x direction
        gy: Gradient in y direction

    Returns:
        Gradient magnitude
    """
    return np.sqrt(gx**2 + gy**2)
