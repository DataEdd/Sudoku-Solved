"""
Orientation Assignment for SIFT Keypoints

This module computes the dominant orientation for each keypoint based on
local image gradients. This enables rotation-invariant feature description.

Key concepts:
- Gradients describe local image structure (magnitude = edge strength, direction = edge angle)
- Orientation histogram captures dominant gradient directions around a keypoint
- Multiple dominant orientations create multiple keypoints for better matching
"""

import numpy as np

from .scale_space import gaussian_blur
from .keypoints import Keypoint


def compute_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient magnitude and orientation for every pixel.

    Uses central differences for derivative approximation:
    - dx = L(x+1, y) - L(x-1, y)
    - dy = L(x, y+1) - L(x, y-1)

    Args:
        image: 2D grayscale image (float values)

    Returns:
        Tuple of (magnitude, orientation):
        - magnitude: 2D array of gradient magnitudes
        - orientation: 2D array of gradient angles in degrees [0, 360)
    """
    # Compute derivatives using central differences
    dx = np.zeros_like(image)
    dx[:, 1:-1] = image[:, 2:] - image[:, :-2]

    dy = np.zeros_like(image)
    dy[1:-1, :] = image[2:, :] - image[:-2, :]

    # Compute magnitude
    magnitude = np.sqrt(dx**2 + dy**2)

    # Compute orientation (in degrees, range [0, 360))
    orientation = np.rad2deg(np.arctan2(dy, dx)) % 360

    return magnitude, orientation


def compute_orientation_histogram(
    magnitude: np.ndarray,
    orientation: np.ndarray,
    kp_x: float,
    kp_y: float,
    sigma: float,
    num_bins: int = 36,
    window_factor: float = 1.5
) -> np.ndarray:
    """
    Compute orientation histogram for a keypoint.

    Args:
        magnitude: Gradient magnitude image
        orientation: Gradient orientation image (degrees)
        kp_x, kp_y: Keypoint location
        sigma: Keypoint scale (determines window size)
        num_bins: Number of orientation bins (default 36 = 10 degrees each)
        window_factor: Gaussian window sigma = window_factor * sigma

    Returns:
        1D histogram array of weighted orientation counts
    """
    height, width = magnitude.shape

    # Window radius: 3 * window_sigma captures most of the Gaussian
    window_sigma = window_factor * sigma
    radius = int(round(3 * window_sigma))

    # Initialize histogram
    histogram = np.zeros(num_bins)
    bin_width = 360.0 / num_bins

    # Integer keypoint location
    kp_x_int = int(round(kp_x))
    kp_y_int = int(round(kp_y))

    # Iterate over window
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y = kp_y_int + dy
            x = kp_x_int + dx

            # Check bounds
            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            # Check if within circular window
            dist_sq = dx**2 + dy**2
            if dist_sq > radius**2:
                continue

            # Gaussian weight (closer to keypoint = higher weight)
            gaussian_weight = np.exp(-dist_sq / (2 * window_sigma**2))

            # Contribution = magnitude * gaussian weight
            weight = magnitude[y, x] * gaussian_weight

            # Determine bin
            angle = orientation[y, x]
            bin_idx = int(angle / bin_width) % num_bins

            # Add to histogram
            histogram[bin_idx] += weight

    return histogram


def find_dominant_orientations(
    histogram: np.ndarray,
    peak_ratio: float = 0.8
) -> list[float]:
    """
    Find dominant orientations from histogram.

    SIFT creates multiple keypoints if there are multiple peaks
    above 80% of the maximum. This improves matching stability.

    Args:
        histogram: Orientation histogram
        peak_ratio: Minimum ratio of max to be considered a peak

    Returns:
        List of refined orientations (in degrees)
    """
    num_bins = len(histogram)
    bin_width = 360.0 / num_bins

    # Smooth histogram to reduce noise
    smoothed = np.zeros_like(histogram)
    for i in range(num_bins):
        smoothed[i] = (histogram[(i-1) % num_bins] +
                       histogram[i] +
                       histogram[(i+1) % num_bins]) / 3

    # Find peaks
    max_val = smoothed.max()
    threshold = peak_ratio * max_val

    orientations = []

    for i in range(num_bins):
        # Check if this bin is a local maximum above threshold
        prev_val = smoothed[(i - 1) % num_bins]
        curr_val = smoothed[i]
        next_val = smoothed[(i + 1) % num_bins]

        if curr_val > prev_val and curr_val > next_val and curr_val >= threshold:
            # Parabolic interpolation for sub-bin accuracy
            denominator = 2 * (prev_val - 2 * curr_val + next_val)

            if abs(denominator) > 1e-6:
                offset = (prev_val - next_val) / denominator
            else:
                offset = 0

            # Refined orientation
            orientation = (i + offset) * bin_width
            orientation = orientation % 360  # Wrap to [0, 360)

            orientations.append(orientation)

    return orientations


def assign_orientations(
    keypoints: list[Keypoint],
    gaussian_pyramid: list[list[np.ndarray]],
    sigmas: list[list[float]],
) -> list[Keypoint]:
    """
    Assign orientations to all keypoints.

    For each keypoint, computes an orientation histogram and finds dominant
    orientations. If a keypoint has multiple dominant orientations (>80% of max),
    creates multiple keypoints at the same location with different orientations.

    Args:
        keypoints: List of Keypoint objects (orientation field will be ignored)
        gaussian_pyramid: List of blurred images per octave from build_gaussian_pyramid()
        sigmas: Sigma values per scale per octave from build_gaussian_pyramid()

    Returns:
        List of Keypoint objects with orientation field set. May have more keypoints
        than input if some have multiple dominant orientations.
    """
    oriented_keypoints = []

    for kp in keypoints:
        octave = kp.octave

        # Find the closest scale index for this keypoint's sigma
        octave_sigmas = sigmas[octave]
        # Convert sigma to local octave coordinates
        scale_factor = 2**octave
        local_sigma = kp.sigma / scale_factor

        # Find closest scale index
        scale_idx = 0
        min_diff = float("inf")
        for idx, s in enumerate(octave_sigmas):
            diff = abs(s / scale_factor - local_sigma)
            if diff < min_diff:
                min_diff = diff
                scale_idx = idx

        # Clamp scale_idx to valid range
        scale_idx = max(0, min(scale_idx, len(gaussian_pyramid[octave]) - 1))

        # Get the Gaussian image at the keypoint's scale
        gaussian_image = gaussian_pyramid[octave][scale_idx]

        # Convert keypoint coordinates to octave coordinates
        local_x = kp.x / scale_factor
        local_y = kp.y / scale_factor

        # Compute gradients
        mag, ori = compute_gradients(gaussian_image)

        # Build orientation histogram
        histogram = compute_orientation_histogram(mag, ori, local_x, local_y, local_sigma)

        # Find dominant orientations
        orientations = find_dominant_orientations(histogram)

        # If no orientations found, use 0 degrees as default
        if not orientations:
            orientations = [0.0]

        # Create keypoint(s) with orientation(s)
        for orientation in orientations:
            new_kp = Keypoint(
                x=kp.x,
                y=kp.y,
                octave=kp.octave,
                scale_idx=kp.scale_idx,
                sigma=kp.sigma,
                response=kp.response,
                is_maximum=kp.is_maximum,
                orientation=orientation,
            )
            oriented_keypoints.append(new_kp)

    return oriented_keypoints
