"""
SIFT Keypoint Detection and Localization

This module implements keypoint detection for SIFT including:
- Finding extrema in the DoG pyramid
- Sub-pixel localization via Taylor expansion
- Low contrast filtering
- Edge response filtering

Reference: Lowe, D.G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints"
"""

from dataclasses import dataclass
import numpy as np

from .scale_space import gaussian_blur, build_gaussian_pyramid, build_dog_pyramid


@dataclass
class Keypoint:
    """
    Represents a SIFT keypoint.

    Attributes:
        x: Image x-coordinate (can be sub-pixel after localization)
        y: Image y-coordinate (can be sub-pixel after localization)
        octave: Octave index in the pyramid
        scale_idx: Scale index within octave (can be fractional after localization)
        sigma: Effective sigma in original image coordinates
        response: DoG response value at this keypoint
        is_maximum: True if local maximum, False if local minimum
        orientation: Dominant orientation in degrees (0-360), assigned later
    """
    x: float
    y: float
    octave: int
    scale_idx: float
    sigma: float
    response: float
    is_maximum: bool
    orientation: float | None = None


def find_dog_extrema(dog_pyramid: list[list[np.ndarray]], threshold: float = 0.03) -> list[dict]:
    """
    Find local extrema in the DoG pyramid.

    A pixel is considered an extremum if it is greater than (or less than)
    all 26 neighbors: 8 in the same scale, 9 in the scale above, and 9 in
    the scale below.

    Args:
        dog_pyramid: List of octaves, each containing DoG images
        threshold: Minimum absolute value to consider (removes weak responses)

    Returns:
        List of candidate keypoints as dicts with keys:
        - octave: Octave index
        - scale: Scale index within octave
        - y, x: Pixel coordinates
        - response: DoG value
        - is_maximum: True if local max, False if local min
    """
    keypoints = []

    for octave_idx, dog_octave in enumerate(dog_pyramid):
        # Need at least 3 DoG images to find extrema
        if len(dog_octave) < 3:
            continue

        # Check scales 1 to n-2 (need neighbors above and below)
        for scale_idx in range(1, len(dog_octave) - 1):
            current = dog_octave[scale_idx]
            below = dog_octave[scale_idx - 1]
            above = dog_octave[scale_idx + 1]

            height, width = current.shape

            # Check each pixel (excluding 1-pixel border)
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    val = current[y, x]

                    # Skip weak responses
                    if abs(val) < threshold:
                        continue

                    # Get all 26 neighbors
                    neighbors = []

                    # 9 from scale below
                    neighbors.extend(below[y-1:y+2, x-1:x+2].flatten())

                    # 8 from current scale (excluding center)
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy != 0 or dx != 0:
                                neighbors.append(current[y + dy, x + dx])

                    # 9 from scale above
                    neighbors.extend(above[y-1:y+2, x-1:x+2].flatten())

                    # Check if extremum
                    is_max = val > max(neighbors)
                    is_min = val < min(neighbors)

                    if is_max or is_min:
                        keypoints.append({
                            'octave': octave_idx,
                            'scale': scale_idx,
                            'y': y,
                            'x': x,
                            'response': val,
                            'is_maximum': is_max
                        })

    return keypoints


def compute_gradient_3d(dog_octave: list[np.ndarray], scale: int, y: int, x: int) -> np.ndarray:
    """
    Compute the 3D gradient of DoG at (scale, y, x) using central differences.

    The gradient is [dD/dx, dD/dy, dD/ds] where D is the DoG function.

    Args:
        dog_octave: List of DoG images for one octave
        scale: Scale index (must have neighbors at scale-1 and scale+1)
        y: Row coordinate (must have neighbors at y-1 and y+1)
        x: Column coordinate (must have neighbors at x-1 and x+1)

    Returns:
        3D gradient vector as numpy array [dx, dy, ds]
    """
    # Central difference: f'(x) = (f(x+1) - f(x-1)) / 2
    dx = (dog_octave[scale][y, x + 1] - dog_octave[scale][y, x - 1]) / 2
    dy = (dog_octave[scale][y + 1, x] - dog_octave[scale][y - 1, x]) / 2
    ds = (dog_octave[scale + 1][y, x] - dog_octave[scale - 1][y, x]) / 2

    return np.array([dx, dy, ds])


def compute_hessian_3d(dog_octave: list[np.ndarray], scale: int, y: int, x: int) -> np.ndarray:
    """
    Compute the 3x3 Hessian matrix of DoG at (scale, y, x).

    The Hessian contains second partial derivatives:
    H = [[dxx, dxy, dxs],
         [dxy, dyy, dys],
         [dxs, dys, dss]]

    Args:
        dog_octave: List of DoG images for one octave
        scale: Scale index
        y: Row coordinate
        x: Column coordinate

    Returns:
        3x3 symmetric Hessian matrix as numpy array
    """
    val = dog_octave[scale][y, x]

    # Second derivatives: f''(x) = f(x+1) - 2*f(x) + f(x-1)
    dxx = dog_octave[scale][y, x + 1] - 2 * val + dog_octave[scale][y, x - 1]
    dyy = dog_octave[scale][y + 1, x] - 2 * val + dog_octave[scale][y - 1, x]
    dss = dog_octave[scale + 1][y, x] - 2 * val + dog_octave[scale - 1][y, x]

    # Cross derivatives: f_xy = (f(x+1,y+1) - f(x+1,y-1) - f(x-1,y+1) + f(x-1,y-1)) / 4
    dxy = (dog_octave[scale][y + 1, x + 1] - dog_octave[scale][y + 1, x - 1] -
           dog_octave[scale][y - 1, x + 1] + dog_octave[scale][y - 1, x - 1]) / 4

    dxs = (dog_octave[scale + 1][y, x + 1] - dog_octave[scale + 1][y, x - 1] -
           dog_octave[scale - 1][y, x + 1] + dog_octave[scale - 1][y, x - 1]) / 4

    dys = (dog_octave[scale + 1][y + 1, x] - dog_octave[scale + 1][y - 1, x] -
           dog_octave[scale - 1][y + 1, x] + dog_octave[scale - 1][y - 1, x]) / 4

    hessian = np.array([
        [dxx, dxy, dxs],
        [dxy, dyy, dys],
        [dxs, dys, dss]
    ])

    return hessian


def localize_keypoint(
    dog_octave: list[np.ndarray],
    scale: int,
    y: int,
    x: int,
    max_iterations: int = 5
) -> dict | None:
    """
    Refine keypoint location to sub-pixel accuracy via Taylor expansion.

    Uses iterative Newton's method to find the true extremum location.
    The offset is computed as: offset = -H^(-1) * gradient

    Args:
        dog_octave: List of DoG images for one octave
        scale: Initial scale index
        y: Initial row coordinate
        x: Initial column coordinate
        max_iterations: Maximum refinement iterations

    Returns:
        Dict with refined keypoint info, or None if rejected:
        - x, y, scale: Refined coordinates (can be fractional)
        - response: DoG value at refined location
        - offset: Final offset from integer location
    """
    height, width = dog_octave[0].shape
    num_scales = len(dog_octave)

    for iteration in range(max_iterations):
        # Compute gradient and Hessian
        gradient = compute_gradient_3d(dog_octave, scale, y, x)
        hessian = compute_hessian_3d(dog_octave, scale, y, x)

        # Solve for offset: offset = -H^(-1) * g
        try:
            offset = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            # Singular matrix - reject this keypoint
            return None

        # If offset is small, we've converged
        if np.all(np.abs(offset) < 0.5):
            break

        # Otherwise, move to new integer location and repeat
        x += int(round(offset[0]))
        y += int(round(offset[1]))
        scale += int(round(offset[2]))

        # Check bounds
        if (x < 1 or x >= width - 1 or
            y < 1 or y >= height - 1 or
            scale < 1 or scale >= num_scales - 1):
            return None

    # Compute response at refined location
    # D(x_hat) = D + 0.5 * gradient^T * offset
    response = dog_octave[scale][y, x] + 0.5 * np.dot(gradient, offset)

    return {
        'x': x + offset[0],
        'y': y + offset[1],
        'scale': scale + offset[2],
        'response': response,
        'offset': offset
    }


def compute_hessian_2d(dog_image: np.ndarray, y: int, x: int) -> np.ndarray:
    """
    Compute the 2x2 spatial Hessian at (y, x) for edge detection.

    The Hessian eigenvalues indicate local curvature:
    - Corner: both eigenvalues large and similar
    - Edge: one eigenvalue large, one small
    - Flat: both eigenvalues small

    Args:
        dog_image: Single DoG image
        y: Row coordinate
        x: Column coordinate

    Returns:
        2x2 symmetric Hessian matrix [[dxx, dxy], [dxy, dyy]]
    """
    val = dog_image[y, x]

    dxx = dog_image[y, x + 1] - 2 * val + dog_image[y, x - 1]
    dyy = dog_image[y + 1, x] - 2 * val + dog_image[y - 1, x]
    dxy = (dog_image[y + 1, x + 1] - dog_image[y + 1, x - 1] -
           dog_image[y - 1, x + 1] + dog_image[y - 1, x - 1]) / 4

    return np.array([[dxx, dxy], [dxy, dyy]])


def filter_low_contrast(keypoint: dict, contrast_threshold: float = 0.03) -> bool:
    """
    Check if keypoint has sufficient contrast.

    Low contrast keypoints are sensitive to noise and not stable
    across different images.

    Args:
        keypoint: Result from localize_keypoint() with 'response' key
        contrast_threshold: Minimum |response| to keep

    Returns:
        True if keypoint should be KEPT (sufficient contrast)
        False if keypoint should be rejected (low contrast)
    """
    return abs(keypoint['response']) >= contrast_threshold


def filter_edge_response(
    dog_image: np.ndarray,
    y: int,
    x: int,
    edge_threshold: float = 10
) -> bool:
    """
    Check if keypoint is on an edge (reject) or corner (keep).

    Uses the ratio of Hessian eigenvalues to distinguish:
    - Corners have similar eigenvalues (ratio close to 1)
    - Edges have very different eigenvalues (high ratio)

    The test uses trace^2/det which avoids explicit eigenvalue computation:
    trace^2/det = (lambda1 + lambda2)^2 / (lambda1 * lambda2)

    Args:
        dog_image: DoG image at the keypoint's scale
        y: Row coordinate
        x: Column coordinate
        edge_threshold: Maximum allowed eigenvalue ratio (r in paper)
                       Default 10 means reject if ratio > 10

    Returns:
        True if keypoint should be KEPT (corner-like)
        False if keypoint should be rejected (edge-like)
    """
    H = compute_hessian_2d(dog_image, y, x)

    trace = H[0, 0] + H[1, 1]
    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]

    # Reject if determinant is non-positive (saddle point or worse)
    if det <= 0:
        return False

    # The ratio test: trace^2/det < (r+1)^2/r
    r = edge_threshold
    threshold = ((r + 1) ** 2) / r

    ratio = (trace ** 2) / det

    return ratio < threshold


def find_keypoints(
    dog_pyramid: list[list[np.ndarray]],
    sigmas: list[list[float]],
    contrast_threshold: float = 0.03,
    edge_threshold: float = 10,
    initial_threshold: float = 0.01
) -> tuple[list[Keypoint], dict]:
    """
    Complete SIFT keypoint detection with localization and filtering.

    Pipeline steps:
    1. Find DoG extrema (candidate keypoints)
    2. Localize to sub-pixel accuracy via Taylor expansion
    3. Filter low-contrast points
    4. Filter edge responses

    Args:
        dog_pyramid: DoG pyramid from build_dog_pyramid()
        sigmas: Sigma values from build_gaussian_pyramid()
        contrast_threshold: Minimum contrast for final filtering
        edge_threshold: Maximum eigenvalue ratio for edge test
        initial_threshold: Minimum DoG value for initial candidates

    Returns:
        Tuple of (keypoints, stats):
        - keypoints: List of Keypoint objects
        - stats: Dict with detection statistics
    """
    keypoints = []
    stats = {
        'candidates': 0,
        'localization_failed': 0,
        'low_contrast': 0,
        'edge_response': 0,
        'accepted': 0
    }

    for octave_idx, dog_octave in enumerate(dog_pyramid):
        if len(dog_octave) < 3:
            continue

        height, width = dog_octave[0].shape

        # Check each scale (excluding first and last)
        for scale_idx in range(1, len(dog_octave) - 1):
            current = dog_octave[scale_idx]
            below = dog_octave[scale_idx - 1]
            above = dog_octave[scale_idx + 1]

            # Check each pixel (excluding border)
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    val = current[y, x]

                    # Initial threshold check
                    if abs(val) < initial_threshold:
                        continue

                    # Check if extremum
                    neighbors = []
                    neighbors.extend(below[y-1:y+2, x-1:x+2].flatten())
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy != 0 or dx != 0:
                                neighbors.append(current[y + dy, x + dx])
                    neighbors.extend(above[y-1:y+2, x-1:x+2].flatten())

                    is_max = val > max(neighbors)
                    is_min = val < min(neighbors)

                    if not (is_max or is_min):
                        continue

                    stats['candidates'] += 1

                    # Step 2: Sub-pixel localization
                    localized = localize_keypoint(dog_octave, scale_idx, y, x)

                    if localized is None:
                        stats['localization_failed'] += 1
                        continue

                    # Step 3: Low contrast filter
                    if not filter_low_contrast(localized, contrast_threshold):
                        stats['low_contrast'] += 1
                        continue

                    # Step 4: Edge response filter
                    # Use integer coordinates for Hessian
                    int_y = int(round(localized['y']))
                    int_x = int(round(localized['x']))
                    int_scale = int(round(localized['scale']))

                    # Bounds check
                    if (int_x < 1 or int_x >= width - 1 or
                        int_y < 1 or int_y >= height - 1 or
                        int_scale < 0 or int_scale >= len(dog_octave)):
                        stats['localization_failed'] += 1
                        continue

                    if not filter_edge_response(dog_octave[int_scale], int_y, int_x, edge_threshold):
                        stats['edge_response'] += 1
                        continue

                    # Keypoint accepted!
                    stats['accepted'] += 1

                    # Convert to image coordinates
                    scale_factor = 2 ** octave_idx

                    keypoints.append(Keypoint(
                        x=localized['x'] * scale_factor,
                        y=localized['y'] * scale_factor,
                        octave=octave_idx,
                        scale_idx=localized['scale'],
                        sigma=sigmas[octave_idx][int_scale],
                        response=localized['response'],
                        is_maximum=is_max
                    ))

    return keypoints, stats
