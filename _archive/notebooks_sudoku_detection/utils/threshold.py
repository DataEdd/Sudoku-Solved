"""
Thresholding operations implemented from scratch using numpy.

This module provides:
- Global thresholding
- Otsu's automatic threshold selection
- Adaptive (local) thresholding
"""

import numpy as np
from typing import Tuple


def global_threshold(image: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply global (fixed) thresholding.

    Pixels above threshold become 255 (white), below become 0 (black).

    Args:
        image: Grayscale (2D) or color (3D) image
        threshold: Threshold value (0-255)

    Returns:
        Binary image (same shape as input)

    Note:
        For color images, thresholding is applied per-channel.
    """
    return (image > threshold).astype(np.uint8) * 255


def otsu_threshold(image: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute optimal threshold using Otsu's method.

    Otsu's method finds the threshold that minimizes intra-class variance
    (equivalently, maximizes inter-class variance).

    Mathematical basis:
    For threshold t, we have two classes:
    - C0: pixels with intensity <= t (background)
    - C1: pixels with intensity > t (foreground)

    Inter-class variance:
        sigma_b^2 = w0 * w1 * (mu0 - mu1)^2

    where:
    - w0, w1 = class probabilities (weights)
    - mu0, mu1 = class means

    We find t that maximizes sigma_b^2.

    Args:
        image: Grayscale (2D) or color (3D) image (uint8)

    Returns:
        Tuple of (optimal threshold, binary image)

    Note:
        Color images are converted to grayscale using luminosity method.

    Example:
        >>> threshold, binary = otsu_threshold(image)
        >>> print(f"Otsu threshold: {threshold}")
    """
    # Handle color images by converting to grayscale
    if len(image.shape) == 3:
        # Luminosity method: 0.299*R + 0.587*G + 0.114*B
        # Note: OpenCV uses BGR order
        image = (0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]).astype(np.uint8)

    # Compute histogram
    hist = np.zeros(256, dtype=np.float64)
    for val in image.flatten():
        hist[int(val)] += 1

    # Normalize to get probabilities
    hist = hist / hist.sum()

    # Cumulative sums
    cumsum = np.cumsum(hist)  # w0(t) = P(X <= t)
    cumsum_mean = np.cumsum(np.arange(256) * hist)  # sum of i * p(i) for i <= t

    # Total mean
    total_mean = cumsum_mean[-1]

    # Find threshold that maximizes inter-class variance
    best_t = 0
    best_variance = 0

    for t in range(256):
        w0 = cumsum[t]
        w1 = 1 - w0

        if w0 == 0 or w1 == 0:
            continue

        # Class means
        mu0 = cumsum_mean[t] / w0
        mu1 = (total_mean - cumsum_mean[t]) / w1

        # Inter-class variance
        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > best_variance:
            best_variance = variance
            best_t = t

    # Apply threshold
    binary = global_threshold(image, best_t)

    return best_t, binary


def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 11,
    c: float = 2.0,
    method: str = 'mean'
) -> np.ndarray:
    """
    Apply adaptive (local) thresholding.

    Instead of a single global threshold, we compute a different threshold
    for each pixel based on its local neighborhood.

    This handles varying illumination across the image.

    Algorithm:
    1. For each pixel, compute local statistic (mean or Gaussian-weighted mean)
    2. Threshold = local_statistic - C
    3. Pixel is white if value > threshold, else black

    Args:
        image: Grayscale (2D) or color (3D) image
        block_size: Size of local neighborhood (must be odd)
        c: Constant subtracted from local mean
        method: 'mean' for box filter, 'gaussian' for Gaussian-weighted

    Returns:
        Binary image (2D grayscale)

    Note:
        Color images are converted to grayscale using luminosity method.

    Example:
        >>> binary = adaptive_threshold(image, block_size=11, c=2)
    """
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd")

    # Handle color images by converting to grayscale
    if len(image.shape) == 3:
        # Luminosity method: 0.299*R + 0.587*G + 0.114*B
        # Note: OpenCV uses BGR order
        image = 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]

    image = image.astype(np.float64)
    h, w = image.shape
    half = block_size // 2

    # Pad image
    padded = np.pad(image, half, mode='reflect')

    # Compute local means using integral image (summed area table)
    # This is O(1) per pixel instead of O(block_size^2)
    if method == 'mean':
        local_mean = _compute_local_mean_integral(padded, block_size, h, w)
    elif method == 'gaussian':
        try:
            from .convolution import gaussian_blur
        except ImportError:
            from convolution import gaussian_blur
        local_mean = gaussian_blur(image, sigma=block_size/6)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Threshold: pixel > (local_mean - c)
    threshold = local_mean - c
    binary = (image > threshold).astype(np.uint8) * 255

    return binary


def _compute_local_mean_integral(
    padded: np.ndarray,
    block_size: int,
    h: int,
    w: int
) -> np.ndarray:
    """
    Compute local means using integral image for O(1) per-pixel computation.

    An integral image I' is defined as:
        I'(x,y) = sum of all pixels I(i,j) where i<=x and j<=y

    The sum of pixels in any rectangle can be computed in O(1):
        sum(x1,y1,x2,y2) = I'(x2,y2) - I'(x1-1,y2) - I'(x2,y1-1) + I'(x1-1,y1-1)
    """
    # Compute integral image
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    half = block_size // 2
    output = np.zeros((h, w), dtype=np.float64)

    area = block_size * block_size

    for i in range(h):
        for j in range(w):
            # Coordinates in padded/integral image
            # The window is centered at (i+half, j+half) in padded coordinates
            x1 = j
            y1 = i
            x2 = j + block_size - 1
            y2 = i + block_size - 1

            # Sum using integral image
            total = integral[y2, x2]
            if x1 > 0:
                total -= integral[y2, x1 - 1]
            if y1 > 0:
                total -= integral[y1 - 1, x2]
            if x1 > 0 and y1 > 0:
                total += integral[y1 - 1, x1 - 1]

            output[i, j] = total / area

    return output


def adaptive_threshold_vectorized(
    image: np.ndarray,
    block_size: int = 11,
    c: float = 2.0
) -> np.ndarray:
    """
    Vectorized adaptive thresholding using integral image.

    More efficient than the loop-based version.

    Args:
        image: Grayscale (2D) or color (3D) image
        block_size: Size of local neighborhood (must be odd)
        c: Constant subtracted from local mean

    Returns:
        Binary image (2D grayscale)

    Note:
        Color images are converted to grayscale using luminosity method.
    """
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd")

    # Handle color images by converting to grayscale
    if len(image.shape) == 3:
        # Luminosity method: 0.299*R + 0.587*G + 0.114*B
        # Note: OpenCV uses BGR order
        image = 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]

    image = image.astype(np.float64)
    h, w = image.shape
    half = block_size // 2

    # Pad image
    padded = np.pad(image, half, mode='reflect')

    # Compute integral image with extra row/col of zeros for boundary handling
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    # Compute local sums using vectorized integral image lookup
    area = block_size * block_size

    # Create coordinate arrays
    y1 = np.arange(h)
    y2 = y1 + block_size
    x1 = np.arange(w)
    x2 = x1 + block_size

    # Compute sums for all pixels at once
    Y1, X1 = np.meshgrid(y1, x1, indexing='ij')
    Y2, X2 = np.meshgrid(y2, x2, indexing='ij')

    local_sum = (
        integral[Y2, X2]
        - integral[Y1, X2]
        - integral[Y2, X1]
        + integral[Y1, X1]
    )

    local_mean = local_sum / area

    # Threshold
    threshold = local_mean - c
    binary = (image > threshold).astype(np.uint8) * 255

    return binary
