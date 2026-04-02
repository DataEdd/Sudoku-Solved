"""
SIFT Descriptor Generation.

This module computes 128-dimensional SIFT descriptors for keypoints.
The descriptor captures the local gradient structure around each keypoint
in a rotation and scale invariant manner.

Key concepts:
- 4x4 grid of spatial bins, each with 8 orientation bins = 128 dimensions
- Coordinates rotated to align with keypoint orientation (rotation invariance)
- Gaussian weighting to focus on central region
- Trilinear interpolation for smooth histogram updates
"""

import numpy as np

from .scale_space import gaussian_blur
from .orientation import compute_gradients
from .keypoints import Keypoint


def rotate_point(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    angle_rad: float
) -> tuple[float, float]:
    """
    Rotate a point around a center by angle_rad radians.

    The rotation formula:
        x' = (x - cx) * cos(theta) + (y - cy) * sin(theta)
        y' = -(x - cx) * sin(theta) + (y - cy) * cos(theta)

    Args:
        x: X coordinate of point to rotate
        y: Y coordinate of point to rotate
        center_x: X coordinate of rotation center
        center_y: Y coordinate of rotation center
        angle_rad: Rotation angle in radians

    Returns:
        Tuple of (x', y') rotated coordinates
    """
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Translate to origin
    dx = x - center_x
    dy = y - center_y

    # Rotate
    x_rot = dx * cos_a + dy * sin_a
    y_rot = -dx * sin_a + dy * cos_a

    return x_rot, y_rot


def trilinear_interpolation_weights(
    x_bin: float,
    y_bin: float,
    ori_bin: float,
    num_spatial_bins: int = 4,
    num_ori_bins: int = 8
) -> list[tuple[int, int, float]]:
    """
    Compute soft assignment weights for histogram bins using trilinear interpolation.

    Each sample contributes to up to 8 neighboring bins (2x2x2 cube):
    - 2 spatial bins in X
    - 2 spatial bins in Y
    - 2 orientation bins

    This provides smooth transitions as features move between bins,
    making the descriptor more robust to small changes.

    Args:
        x_bin: Fractional X bin position (0 to num_spatial_bins)
        y_bin: Fractional Y bin position (0 to num_spatial_bins)
        ori_bin: Fractional orientation bin (0 to num_ori_bins)
        num_spatial_bins: Number of spatial bins per dimension (default 4)
        num_ori_bins: Number of orientation bins (default 8)

    Returns:
        List of (spatial_bin_idx, ori_bin_idx, weight) tuples.
        Weights sum to approximately 1.0 for points fully inside the grid.
    """
    contributions = []

    # Determine the bins this sample falls between
    x0 = int(np.floor(x_bin))
    y0 = int(np.floor(y_bin))
    o0 = int(np.floor(ori_bin))

    # Fractional parts (how far into the bin)
    x_frac = x_bin - x0
    y_frac = y_bin - y0
    o_frac = ori_bin - o0

    # Iterate over 2x2x2 neighboring bins
    for dx in [0, 1]:
        for dy in [0, 1]:
            for do in [0, 1]:
                xi = x0 + dx
                yi = y0 + dy
                oi = (o0 + do) % num_ori_bins  # Orientation wraps around

                # Check spatial bounds
                if xi < 0 or xi >= num_spatial_bins:
                    continue
                if yi < 0 or yi >= num_spatial_bins:
                    continue

                # Compute weight (product of 1D linear weights)
                wx = (1 - x_frac) if dx == 0 else x_frac
                wy = (1 - y_frac) if dy == 0 else y_frac
                wo = (1 - o_frac) if do == 0 else o_frac

                weight = wx * wy * wo

                if weight > 0:
                    spatial_idx = yi * num_spatial_bins + xi
                    contributions.append((spatial_idx, oi, weight))

    return contributions


def compute_sift_descriptor(
    image: np.ndarray,
    kp_x: float,
    kp_y: float,
    kp_sigma: float,
    kp_orientation: float,
    num_spatial_bins: int = 4,
    num_ori_bins: int = 8,
    window_width: int = 16,
    magnitude_threshold: float = 0.2
) -> np.ndarray:
    """
    Compute a 128-dimensional SIFT descriptor for a single keypoint.

    The descriptor captures the local gradient structure in a rotation-invariant
    manner by:
    1. Blurring the image at the keypoint's scale
    2. Computing gradients
    3. Sampling in a rotated window aligned with keypoint orientation
    4. Building a 4x4 grid of 8-bin orientation histograms
    5. Normalizing and thresholding for illumination invariance

    Args:
        image: Grayscale image (float, typically 0-1)
        kp_x: Keypoint X coordinate
        kp_y: Keypoint Y coordinate
        kp_sigma: Keypoint scale (sigma)
        kp_orientation: Keypoint orientation in degrees
        num_spatial_bins: Number of spatial bins per dimension (default 4)
        num_ori_bins: Number of orientation bins (default 8)
        window_width: Descriptor window width in pixels (default 16)
        magnitude_threshold: Cap on individual histogram values (default 0.2)

    Returns:
        Normalized 128-dimensional descriptor vector (4*4*8 = 128)
    """
    height, width = image.shape

    # Step 1: Blur image at keypoint scale
    blurred = gaussian_blur(image, kp_sigma)

    # Step 2: Compute gradients
    magnitude, orientation = compute_gradients(blurred)

    # Initialize histogram: 4x4 spatial bins, 8 orientation bins each
    histogram = np.zeros((num_spatial_bins, num_spatial_bins, num_ori_bins))

    # Convert orientation to radians
    theta = np.deg2rad(kp_orientation)

    # Gaussian weight sigma (half of window * 0.5 for smooth falloff)
    sigma_w = window_width / 2 * 0.5

    # Size of each spatial bin
    bin_size = window_width / num_spatial_bins

    # Sample radius (half window, plus some margin for rotation)
    radius = int(np.ceil(window_width * np.sqrt(2) / 2))

    # Step 3: Iterate over sample points in rotated window
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # Sample location in image
            sample_x = int(round(kp_x)) + dx
            sample_y = int(round(kp_y)) + dy

            # Check bounds
            if sample_x < 1 or sample_x >= width - 1:
                continue
            if sample_y < 1 or sample_y >= height - 1:
                continue

            # Rotate sample position relative to keypoint
            x_rot, y_rot = rotate_point(dx, dy, 0, 0, theta)

            # Check if within descriptor window
            if abs(x_rot) > window_width / 2 or abs(y_rot) > window_width / 2:
                continue

            # Compute Gaussian weight based on distance from center
            gauss_weight = np.exp(-(x_rot**2 + y_rot**2) / (2 * sigma_w**2))

            # Get gradient at sample point
            mag = magnitude[sample_y, sample_x]
            ori = orientation[sample_y, sample_x]

            # Rotate gradient orientation relative to keypoint orientation
            ori_relative = (ori - kp_orientation) % 360

            # Convert to bin coordinates
            # Spatial: shift so (0,0) is at corner of descriptor window
            x_bin = (x_rot + window_width / 2) / bin_size - 0.5
            y_bin = (y_rot + window_width / 2) / bin_size - 0.5

            # Orientation bin
            ori_bin = ori_relative / (360 / num_ori_bins)

            # Weighted magnitude
            weighted_mag = mag * gauss_weight

            # Trilinear interpolation - distribute to neighboring bins
            contributions = trilinear_interpolation_weights(
                x_bin, y_bin, ori_bin, num_spatial_bins, num_ori_bins
            )

            for spatial_idx, ori_idx, interp_weight in contributions:
                row = spatial_idx // num_spatial_bins
                col = spatial_idx % num_spatial_bins
                histogram[row, col, ori_idx] += weighted_mag * interp_weight

    # Step 4: Flatten to 128-D vector
    descriptor = histogram.flatten()

    # Normalize to unit length
    norm = np.linalg.norm(descriptor)
    if norm > 1e-6:
        descriptor = descriptor / norm

    # Step 5: Threshold large values to reduce influence of large gradients
    descriptor = np.minimum(descriptor, magnitude_threshold)

    # Step 6: Re-normalize
    norm = np.linalg.norm(descriptor)
    if norm > 1e-6:
        descriptor = descriptor / norm

    return descriptor


def compute_descriptors(
    image: np.ndarray,
    keypoints: list[Keypoint]
) -> np.ndarray:
    """
    Compute SIFT descriptors for all keypoints.

    Args:
        image: Grayscale image (float, typically 0-1)
        keypoints: List of Keypoint objects with x, y, sigma, orientation

    Returns:
        (N, 128) array of descriptors, one row per keypoint
    """
    if len(keypoints) == 0:
        return np.zeros((0, 128), dtype=np.float64)

    descriptors = []

    for kp in keypoints:
        desc = compute_sift_descriptor(
            image,
            kp.x,
            kp.y,
            kp.sigma,
            kp.orientation
        )
        descriptors.append(desc)

    return np.array(descriptors, dtype=np.float64)
