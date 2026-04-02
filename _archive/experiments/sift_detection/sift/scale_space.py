"""Scale-space and Difference of Gaussians (DoG) implementation for SIFT.

This module provides functions for building Gaussian scale-space pyramids
and computing Difference of Gaussians, which are foundational components
of the SIFT (Scale-Invariant Feature Transform) algorithm.

The scale-space representation allows detection of features at multiple
scales, making feature detection invariant to image scale changes.
"""

import numpy as np


def gaussian_kernel_1d(sigma: float, size: int | None = None) -> np.ndarray:
    """Create a 1D Gaussian kernel.

    The Gaussian kernel is used for smoothing images. The kernel size is
    automatically calculated to capture 99.7% of the distribution (6*sigma)
    if not explicitly provided.

    Args:
        sigma: Standard deviation of the Gaussian (controls blur amount).
            Larger values produce more blur.
        size: Kernel size. If None, automatically calculated as ceil(6*sigma),
            adjusted to be odd for symmetry.

    Returns:
        1D numpy array representing the Gaussian kernel, normalized so that
        the sum equals 1 (preserves image brightness).

    Examples:
        >>> kernel = gaussian_kernel_1d(1.0)
        >>> np.isclose(kernel.sum(), 1.0)
        True
        >>> kernel = gaussian_kernel_1d(2.0, size=11)
        >>> len(kernel)
        11
    """
    # Rule of thumb: kernel size should be 6*sigma to capture 99.7% of distribution
    if size is None:
        size = int(np.ceil(6 * sigma))
        if size % 2 == 0:
            size += 1  # Make it odd for symmetric kernel

    # Create coordinate array centered at 0
    x = np.arange(size) - size // 2

    # Apply Gaussian formula: exp(-x^2 / (2 * sigma^2))
    kernel = np.exp(-x**2 / (2 * sigma**2))

    # Normalize so sum = 1 (preserves image brightness)
    kernel = kernel / kernel.sum()

    return kernel


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur using separable convolution.

    Instead of 2D convolution (slow), we apply 1D blur twice:
    1. Horizontally across rows
    2. Vertically across columns

    This is O(n*k) instead of O(n*k^2) where k is kernel size.

    Args:
        image: 2D grayscale image as numpy array (float64, values in 0-1 range).
        sigma: Standard deviation of the Gaussian blur. Larger values produce
            more blur.

    Returns:
        Blurred image with the same shape as input.

    Examples:
        >>> img = np.random.rand(100, 100)
        >>> blurred = gaussian_blur(img, 2.0)
        >>> blurred.shape == img.shape
        True
    """
    kernel = gaussian_kernel_1d(sigma)

    # Pad image to handle borders using reflection
    pad = len(kernel) // 2
    padded = np.pad(image, pad, mode='reflect')

    # Horizontal pass
    temp = np.zeros_like(padded, dtype=np.float64)
    for i in range(len(kernel)):
        temp[:, pad:-pad] += kernel[i] * padded[:, i:i + image.shape[1]]

    # Vertical pass
    result = np.zeros_like(image, dtype=np.float64)
    for i in range(len(kernel)):
        result += kernel[i] * temp[i:i + image.shape[0], pad:-pad]

    return result


def build_gaussian_pyramid(
    image: np.ndarray,
    num_octaves: int = 4,
    num_scales: int = 5,
    sigma_base: float = 1.6
) -> tuple[list[list[np.ndarray]], list[list[float]]]:
    """Build a Gaussian scale-space pyramid.

    The pyramid is organized into octaves (resolution levels) and scales
    (blur levels within each octave). Each octave has half the resolution
    of the previous one.

    The scale progression within an octave follows: sigma_base * k^i
    where k = 2^(1/s) and s = num_scales - 3.

    Args:
        image: Input grayscale image normalized to 0-1 range.
        num_octaves: Number of octaves (resolution levels). Each successive
            octave has half the resolution. Default is 4.
        num_scales: Number of blur levels per octave (s+3 in Lowe's paper).
            Default is 5.
        sigma_base: Base sigma value for the first scale. Default is 1.6.

    Returns:
        A tuple containing:
        - pyramid: List of octaves, where each octave is a list of blurred
          images at different scales.
        - sigmas_per_octave: List of sigma values for each scale in each
          octave, in original image coordinates.

    Examples:
        >>> img = np.random.rand(256, 256)
        >>> pyramid, sigmas = build_gaussian_pyramid(img, num_octaves=3)
        >>> len(pyramid)  # 3 octaves
        3
        >>> len(pyramid[0])  # 5 scales per octave
        5
        >>> pyramid[1][0].shape[0] == pyramid[0][0].shape[0] // 2
        True
    """
    # k = 2^(1/s) where s = num_scales - 3 (we need s+3 images to get s DoG extrema)
    s = num_scales - 3
    k = 2 ** (1 / s)

    pyramid: list[list[np.ndarray]] = []
    sigmas_per_octave: list[list[float]] = []

    # Compute sigma values for one octave
    sigma_values = [sigma_base * (k ** i) for i in range(num_scales)]

    current_image = image.astype(np.float64)

    for octave in range(num_octaves):
        octave_images: list[np.ndarray] = []
        octave_sigmas: list[float] = []

        for scale_idx, sigma in enumerate(sigma_values):
            if scale_idx == 0 and octave == 0:
                # First image: blur original
                blurred = gaussian_blur(current_image, sigma)
            elif scale_idx == 0:
                # Start of new octave: use downsampled version
                blurred = gaussian_blur(current_image, sigma)
            else:
                # Incremental blur: blur from previous scale
                # sigma_diff = sqrt(sigma_new^2 - sigma_prev^2)
                sigma_prev = sigma_values[scale_idx - 1]
                sigma_diff = np.sqrt(sigma**2 - sigma_prev**2)
                blurred = gaussian_blur(octave_images[-1], sigma_diff)

            octave_images.append(blurred)
            # Effective sigma in original image coordinates
            octave_sigmas.append(sigma * (2 ** octave))

        pyramid.append(octave_images)
        sigmas_per_octave.append(octave_sigmas)

        # Downsample for next octave (take every other pixel)
        # Use the image at scale index s (where sigma = 2*sigma_base)
        if octave < num_octaves - 1:
            current_image = octave_images[s][::2, ::2]

    return pyramid, sigmas_per_octave


def build_dog_pyramid(
    gaussian_pyramid: list[list[np.ndarray]]
) -> list[list[np.ndarray]]:
    """Build Difference of Gaussians (DoG) pyramid from Gaussian pyramid.

    The DoG approximates the Laplacian of Gaussian (LoG) and is computed
    by subtracting adjacent Gaussian-blurred images:
        DoG(x, y, sigma) = G(x, y, k*sigma) - G(x, y, sigma)

    DoG highlights features at specific scales - features that are larger
    or smaller than the scale difference are suppressed.

    Args:
        gaussian_pyramid: Gaussian scale-space pyramid as returned by
            build_gaussian_pyramid(). A list of octaves, where each octave
            is a list of blurred images.

    Returns:
        DoG pyramid with the same structure as the input, but with N-1
        images per octave (where N is the number of Gaussian images).

    Examples:
        >>> img = np.random.rand(128, 128)
        >>> gauss_pyr, _ = build_gaussian_pyramid(img, num_octaves=2)
        >>> dog_pyr = build_dog_pyramid(gauss_pyr)
        >>> len(dog_pyr)  # Same number of octaves
        2
        >>> len(dog_pyr[0])  # One less than Gaussian (5-1=4)
        4
    """
    dog_pyramid: list[list[np.ndarray]] = []

    for octave_images in gaussian_pyramid:
        dog_octave: list[np.ndarray] = []

        for i in range(len(octave_images) - 1):
            # Subtract adjacent Gaussian images
            dog = octave_images[i + 1] - octave_images[i]
            dog_octave.append(dog)

        dog_pyramid.append(dog_octave)

    return dog_pyramid
