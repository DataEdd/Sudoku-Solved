"""
Convolution operations implemented from scratch using numpy.

This module provides:
- 2D convolution with various padding modes
- Gaussian kernel generation
- Gaussian blur for noise reduction
"""

import numpy as np
from typing import Tuple


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    padding: str = 'zero'
) -> np.ndarray:
    """
    Perform 2D convolution on an image with a kernel.

    The convolution operation slides the kernel over the image and computes
    the sum of element-wise products at each position.

    Mathematical definition:
        (I * K)[i,j] = sum_{m,n} I[i+m, j+n] * K[m, n]

    Args:
        image: 2D grayscale or 3D color image (H, W) or (H, W, C)
        kernel: 2D numpy array (convolution kernel)
        padding: 'zero' for zero-padding, 'reflect' for reflection padding

    Returns:
        Convolved image with same dimensions as input

    Example:
        >>> img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Laplacian
        >>> convolve2d(img, kernel)
    """
    # Ensure inputs are float for computation
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    # Handle color images by convolving each channel separately
    if len(image.shape) == 3:
        result = np.zeros_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            result[:, :, c] = convolve2d(image[:, :, c], kernel, padding)
        return result

    # Get dimensions (now guaranteed 2D)
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    # Kernel must be odd-sized for symmetric padding
    pad_h = k_h // 2
    pad_w = k_w // 2

    # Apply padding
    if padding == 'zero':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    elif padding == 'reflect':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    else:
        raise ValueError(f"Unknown padding mode: {padding}")

    # Initialize output
    output = np.zeros_like(image, dtype=np.float64)

    # Flip kernel for convolution (not correlation)
    # Convolution flips the kernel, correlation does not
    kernel_flipped = np.flip(kernel)

    # Perform convolution
    for i in range(img_h):
        for j in range(img_w):
            # Extract region
            region = padded[i:i+k_h, j:j+k_w]
            # Element-wise multiply and sum
            output[i, j] = np.sum(region * kernel_flipped)

    return output


def convolve2d_vectorized(
    image: np.ndarray,
    kernel: np.ndarray,
    padding: str = 'zero'
) -> np.ndarray:
    """
    Vectorized 2D convolution using numpy stride tricks.

    This is a faster implementation that creates a view of all patches
    and performs matrix multiplication instead of explicit loops.

    Args:
        image: 2D grayscale or 3D color image (H, W) or (H, W, C)
        kernel: 2D numpy array (convolution kernel)
        padding: 'zero' for zero-padding, 'reflect' for reflection padding

    Returns:
        Convolved image with same dimensions as input
    """
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    # Handle color images by convolving each channel separately
    if len(image.shape) == 3:
        result = np.zeros_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            result[:, :, c] = convolve2d_vectorized(image[:, :, c], kernel, padding)
        return result

    # Grayscale processing (guaranteed 2D)
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    # Apply padding
    if padding == 'zero':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    else:
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    # Create sliding window view using stride tricks
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(padded, (k_h, k_w))

    # Flip kernel and compute dot product
    kernel_flipped = np.flip(kernel).flatten()

    # Reshape windows for matrix multiplication
    windows_flat = windows.reshape(img_h, img_w, -1)
    output = np.dot(windows_flat, kernel_flipped)

    return output


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.

    The Gaussian function in 2D:
        G(x,y) = (1 / 2*pi*sigma^2) * exp(-(x^2 + y^2) / (2*sigma^2))

    Args:
        size: Kernel size (must be odd)
        sigma: Standard deviation of the Gaussian

    Returns:
        2D numpy array containing the normalized Gaussian kernel

    Example:
        >>> kernel = gaussian_kernel(5, 1.0)
        >>> kernel.sum()  # Should be approximately 1.0
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    # Create coordinate grids
    half = size // 2
    x = np.arange(-half, half + 1)
    y = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(x, y)

    # Compute Gaussian
    # G(x,y) = exp(-(x^2 + y^2) / (2*sigma^2))
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # Normalize so sum equals 1
    kernel = kernel / kernel.sum()

    return kernel


def gaussian_blur(
    image: np.ndarray,
    sigma: float = 1.0,
    kernel_size: int = None
) -> np.ndarray:
    """
    Apply Gaussian blur to an image.

    Gaussian blur smooths the image by averaging pixels with their neighbors,
    weighted by a Gaussian function. This reduces noise and detail.

    Args:
        image: 2D grayscale or 3D color image (H, W) or (H, W, C)
        sigma: Standard deviation of the Gaussian
        kernel_size: Size of the kernel (auto-computed if None)

    Returns:
        Blurred image (same shape as input)

    Note:
        A common rule is kernel_size = ceil(6*sigma) | 1 (ensure odd)
        For color images, each channel is blurred independently.
    """
    # Auto-compute kernel size if not provided
    if kernel_size is None:
        kernel_size = int(np.ceil(6 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)  # Minimum size of 3

    # Generate Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)

    # Apply convolution
    return convolve2d_vectorized(image, kernel)
