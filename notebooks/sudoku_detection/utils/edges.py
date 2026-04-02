"""
Edge detection operations implemented from scratch using numpy.

This module provides:
- Sobel gradient computation (Gx, Gy)
- Edge magnitude calculation
- Edge direction/angle calculation
- Laplacian edge detection (second derivative)
- Laplacian of Gaussian (LoG)
"""

import numpy as np
from typing import Tuple

try:
    from .convolution import convolve2d_vectorized
except ImportError:
    from convolution import convolve2d_vectorized


# Sobel kernels for gradient computation
# These approximate the partial derivatives of the image intensity

SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float64)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float64)


# Laplacian kernels for second derivative computation
# These detect edges at zero-crossings of the second derivative

# 4-connected Laplacian (only considers horizontal and vertical neighbors)
LAPLACIAN_4 = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
], dtype=np.float64)

# 8-connected Laplacian (includes diagonal neighbors)
LAPLACIAN_8 = np.array([
    [1,  1, 1],
    [1, -8, 1],
    [1,  1, 1]
], dtype=np.float64)

# Directional Laplacian kernels (second derivative in one direction only)
# These are useful for directional edge analysis and Canny-like experiments

# Second derivative in X direction: d²I/dx²
# Approximates: I[i,j-1] - 2*I[i,j] + I[i,j+1]
LAPLACIAN_X = np.array([
    [0,  0,  0],
    [1, -2,  1],
    [0,  0,  0]
], dtype=np.float64)

# Second derivative in Y direction: d²I/dy²
# Approximates: I[i-1,j] - 2*I[i,j] + I[i+1,j]
LAPLACIAN_Y = np.array([
    [0,  1,  0],
    [0, -2,  0],
    [0,  1,  0]
], dtype=np.float64)

# Second derivative in diagonal directions (45° and 135°)
LAPLACIAN_DIAG_45 = np.array([
    [0,  0,  1],
    [0, -2,  0],
    [1,  0,  0]
], dtype=np.float64)

LAPLACIAN_DIAG_135 = np.array([
    [1,  0,  0],
    [0, -2,  0],
    [0,  0,  1]
], dtype=np.float64)


def sobel_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradients using Sobel operators.

    The Sobel operator combines Gaussian smoothing with differentiation.
    It computes the gradient of the image intensity at each pixel.

    Mathematical basis:
    - Gx approximates dI/dx (horizontal gradient)
    - Gy approximates dI/dy (vertical gradient)

    The Sobel kernels are:
        Gx = [-1 0 1]     Gy = [-1 -2 -1]
             [-2 0 2]          [ 0  0  0]
             [-1 0 1]          [ 1  2  1]

    These can be decomposed as:
        Gx = [1]   [-1 0 1]     (smoothing in y, differentiation in x)
             [2] *
             [1]

        Gy = [-1]  [1 2 1]      (differentiation in y, smoothing in x)
             [ 0] *
             [ 1]

    Args:
        image: 2D grayscale or 3D color image (H, W) or (H, W, C)

    Returns:
        Tuple of (Gx, Gy) gradient images (same shape as input)
        - Gx: Horizontal gradient (positive = bright on right)
        - Gy: Vertical gradient (positive = bright below)

    Note:
        For color images, gradients are computed per-channel.
        For edge detection, grayscale is typically preferred.

    Example:
        >>> gx, gy = sobel_gradients(image)
        >>> magnitude = np.sqrt(gx**2 + gy**2)
    """
    gx = convolve2d_vectorized(image, SOBEL_X)
    gy = convolve2d_vectorized(image, SOBEL_Y)

    return gx, gy


def edge_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Compute edge magnitude from gradient components.

    The magnitude represents the "strength" of edges at each pixel.
    Larger values indicate stronger edges (higher contrast).

    Mathematical formula:
        M = sqrt(Gx^2 + Gy^2)

    An approximation sometimes used for speed:
        M ≈ |Gx| + |Gy|

    Args:
        gx: Horizontal gradient (from sobel_gradients)
        gy: Vertical gradient (from sobel_gradients)

    Returns:
        Edge magnitude image (same shape as inputs)
    """
    return np.sqrt(gx**2 + gy**2)


def edge_direction(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Compute edge direction (angle) from gradient components.

    The direction indicates the orientation of edges at each pixel.
    Edges are perpendicular to the gradient direction.

    Mathematical formula:
        theta = atan2(Gy, Gx)

    Result is in radians, range [-pi, pi]:
    - 0: Horizontal edge (vertical gradient)
    - pi/2: Vertical edge (horizontal gradient)

    Args:
        gx: Horizontal gradient
        gy: Vertical gradient

    Returns:
        Edge direction in radians
    """
    return np.arctan2(gy, gx)


def edge_direction_degrees(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Compute edge direction in degrees.

    Returns:
        Edge direction in degrees [0, 180)
        (Edges are symmetric, so we map to half-circle)
    """
    angle = np.rad2deg(np.arctan2(gy, gx))
    # Map to [0, 180) since edges are symmetric
    angle = angle % 180
    return angle


def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Apply non-maximum suppression to thin edges.

    For each pixel, keep it only if it's a local maximum along
    the gradient direction. This produces 1-pixel wide edges.

    Algorithm:
    1. Quantize direction to 4 orientations (0, 45, 90, 135 degrees)
    2. For each pixel, compare with neighbors along gradient direction
    3. Keep pixel only if it's the maximum among the three

    Args:
        magnitude: Edge magnitude image
        direction: Edge direction in radians

    Returns:
        Thinned edge magnitude image
    """
    h, w = magnitude.shape
    output = np.zeros_like(magnitude)

    # Convert direction to degrees and map to [0, 180)
    angle = np.rad2deg(direction) % 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Current magnitude
            m = magnitude[i, j]
            a = angle[i, j]

            # Get neighbors based on gradient direction
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                # Horizontal edge: compare with left and right
                n1 = magnitude[i, j - 1]
                n2 = magnitude[i, j + 1]
            elif 22.5 <= a < 67.5:
                # Diagonal edge (45 deg): compare with top-right and bottom-left
                n1 = magnitude[i - 1, j + 1]
                n2 = magnitude[i + 1, j - 1]
            elif 67.5 <= a < 112.5:
                # Vertical edge: compare with top and bottom
                n1 = magnitude[i - 1, j]
                n2 = magnitude[i + 1, j]
            else:  # 112.5 <= a < 157.5
                # Diagonal edge (135 deg): compare with top-left and bottom-right
                n1 = magnitude[i - 1, j - 1]
                n2 = magnitude[i + 1, j + 1]

            # Keep only if local maximum
            if m >= n1 and m >= n2:
                output[i, j] = m

    return output


def canny_edge_detection(
    image: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Canny edge detection implemented from scratch.

    The Canny algorithm:
    1. Gaussian blur for noise reduction
    2. Compute gradients using Sobel
    3. Non-maximum suppression for edge thinning
    4. Hysteresis thresholding for edge tracking

    Args:
        image: Grayscale input image
        low_threshold: Lower threshold for hysteresis
        high_threshold: Upper threshold for hysteresis
        sigma: Gaussian blur sigma

    Returns:
        Binary edge image
    """
    try:
        from .convolution import gaussian_blur
    except ImportError:
        from convolution import gaussian_blur

    # Step 1: Gaussian blur
    blurred = gaussian_blur(image, sigma)

    # Step 2: Compute gradients
    gx, gy = sobel_gradients(blurred)
    magnitude = edge_magnitude(gx, gy)
    direction = edge_direction(gx, gy)

    # Step 3: Non-maximum suppression
    nms = non_maximum_suppression(magnitude, direction)

    # Step 4: Hysteresis thresholding
    strong = nms >= high_threshold
    weak = (nms >= low_threshold) & (nms < high_threshold)

    # Edge tracking by hysteresis
    output = np.zeros_like(image, dtype=np.uint8)
    output[strong] = 255

    # Connect weak edges to strong edges
    h, w = image.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if weak[i, j]:
                # Check if any strong neighbor
                if np.any(strong[i-1:i+2, j-1:j+2]):
                    output[i, j] = 255

    return output


def laplacian(image: np.ndarray, kernel_type: str = '4') -> np.ndarray:
    """
    Compute Laplacian (second derivative) of an image.

    The Laplacian detects edges by finding zero-crossings of the
    second derivative. Unlike Sobel (first derivative), it:
    - Detects edges in all directions simultaneously
    - Is more sensitive to noise
    - Produces double edges at transitions

    Mathematical basis:
        ∇²I = ∂²I/∂x² + ∂²I/∂y²

    For discrete images, using finite differences:
        ∂²I/∂x² ≈ I[i,j+1] - 2*I[i,j] + I[i,j-1]
        ∂²I/∂y² ≈ I[i+1,j] - 2*I[i,j] + I[i-1,j]

    Combined into the Laplacian kernel:
        4-connected: [0  1  0]    8-connected: [1  1  1]
                     [1 -4  1]                  [1 -8  1]
                     [0  1  0]                  [1  1  1]

    Args:
        image: 2D grayscale or 3D color image
        kernel_type: '4' for 4-connected, '8' for 8-connected

    Returns:
        Laplacian image (can have negative values!)

    Note:
        The output contains positive and negative values.
        Edges occur at zero-crossings.
        For visualization, use np.abs() or normalize.

    Example:
        >>> lap = laplacian(image, kernel_type='4')
        >>> edges = np.abs(lap)  # For visualization
    """
    if kernel_type == '4':
        kernel = LAPLACIAN_4
    elif kernel_type == '8':
        kernel = LAPLACIAN_8
    else:
        raise ValueError(f"kernel_type must be '4' or '8', got '{kernel_type}'")

    return convolve2d_vectorized(image, kernel)


def laplacian_directional(
    image: np.ndarray,
    direction: str = 'x'
) -> np.ndarray:
    """
    Compute directional Laplacian (second derivative in one direction).

    Unlike the full Laplacian which combines ∂²I/∂x² + ∂²I/∂y²,
    this computes only one component. Useful for:
    - Analyzing edges in specific directions
    - Building custom Canny-like detectors
    - Understanding the relationship between first and second derivatives

    Mathematical basis:
        ∂²I/∂x² ≈ I[i,j-1] - 2*I[i,j] + I[i,j+1]  (horizontal)
        ∂²I/∂y² ≈ I[i-1,j] - 2*I[i,j] + I[i+1,j]  (vertical)

    Relationship to first derivative:
        If Sobel gives dI/dx, then Laplacian_x gives d²I/dx²
        Zero-crossings of second derivative = peaks of first derivative = edges

    Args:
        image: 2D grayscale or 3D color image
        direction: 'x' for horizontal, 'y' for vertical,
                   'diag45' for 45° diagonal, 'diag135' for 135° diagonal

    Returns:
        Directional second derivative (can have negative values)

    Example:
        >>> lap_x = laplacian_directional(image, 'x')  # d²I/dx²
        >>> lap_y = laplacian_directional(image, 'y')  # d²I/dy²
        >>> # Note: lap_x + lap_y ≈ laplacian(image, '4')
    """
    if direction == 'x':
        kernel = LAPLACIAN_X
    elif direction == 'y':
        kernel = LAPLACIAN_Y
    elif direction == 'diag45':
        kernel = LAPLACIAN_DIAG_45
    elif direction == 'diag135':
        kernel = LAPLACIAN_DIAG_135
    else:
        raise ValueError(f"direction must be 'x', 'y', 'diag45', or 'diag135', got '{direction}'")

    return convolve2d_vectorized(image, kernel)


def laplacian_components(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute both directional Laplacian components.

    Returns the X and Y second derivatives separately, which can be:
    - Combined: lap_x + lap_y = full Laplacian (4-connected)
    - Analyzed separately for directional edge information
    - Used with gradient direction for Canny-like edge refinement

    Args:
        image: 2D grayscale or 3D color image

    Returns:
        Tuple of (lap_x, lap_y):
        - lap_x: Second derivative in X (d²I/dx²)
        - lap_y: Second derivative in Y (d²I/dy²)

    Example:
        >>> lap_x, lap_y = laplacian_components(image)
        >>> full_laplacian = lap_x + lap_y  # Same as laplacian(image, '4')
    """
    lap_x = convolve2d_vectorized(image, LAPLACIAN_X)
    lap_y = convolve2d_vectorized(image, LAPLACIAN_Y)
    return lap_x, lap_y


def laplacian_of_gaussian(
    image: np.ndarray,
    sigma: float = 1.0,
    kernel_type: str = '4'
) -> np.ndarray:
    """
    Laplacian of Gaussian (LoG) edge detection.

    The LoG combines:
    1. Gaussian blur (noise reduction)
    2. Laplacian (edge detection)

    This is more robust to noise than plain Laplacian.

    Mathematical basis:
        LoG(x,y) = -1/(π*σ⁴) * [1 - (x²+y²)/(2σ²)] * exp(-(x²+y²)/(2σ²))

    In practice, we compute: Laplacian(Gaussian(image))

    The LoG can also be approximated by Difference of Gaussians (DoG):
        DoG ≈ Gaussian(σ₁) - Gaussian(σ₂)  where σ₂ ≈ 1.6 * σ₁

    Args:
        image: 2D grayscale or 3D color image
        sigma: Gaussian blur sigma (larger = less noise, less detail)
        kernel_type: '4' or '8' for Laplacian kernel

    Returns:
        LoG image (edges at zero-crossings)

    Example:
        >>> log = laplacian_of_gaussian(image, sigma=2.0)
        >>> edges = np.abs(log)
    """
    try:
        from .convolution import gaussian_blur
    except ImportError:
        from convolution import gaussian_blur

    # Step 1: Gaussian blur
    blurred = gaussian_blur(image, sigma)

    # Step 2: Laplacian
    return laplacian(blurred, kernel_type)


def zero_crossing(laplacian_image: np.ndarray, threshold: float = 0) -> np.ndarray:
    """
    Detect zero-crossings in a Laplacian image.

    Zero-crossings occur where the Laplacian changes sign,
    indicating an edge location.

    Algorithm:
    For each pixel, check if any neighboring pair has opposite signs.

    Args:
        laplacian_image: Output from laplacian() or laplacian_of_gaussian()
        threshold: Minimum absolute difference to count as crossing

    Returns:
        Binary image with edges at zero-crossings

    Example:
        >>> lap = laplacian_of_gaussian(image, sigma=2.0)
        >>> edges = zero_crossing(lap, threshold=5)
    """
    h, w = laplacian_image.shape[:2]
    output = np.zeros((h, w), dtype=np.uint8)

    # Handle color images by converting to grayscale
    if len(laplacian_image.shape) == 3:
        # Use mean across channels
        laplacian_image = np.mean(laplacian_image, axis=2)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Check horizontal crossing
            if (laplacian_image[i, j-1] * laplacian_image[i, j+1] < 0 and
                abs(laplacian_image[i, j-1] - laplacian_image[i, j+1]) > threshold):
                output[i, j] = 255
            # Check vertical crossing
            elif (laplacian_image[i-1, j] * laplacian_image[i+1, j] < 0 and
                  abs(laplacian_image[i-1, j] - laplacian_image[i+1, j]) > threshold):
                output[i, j] = 255
            # Check diagonal crossings
            elif (laplacian_image[i-1, j-1] * laplacian_image[i+1, j+1] < 0 and
                  abs(laplacian_image[i-1, j-1] - laplacian_image[i+1, j+1]) > threshold):
                output[i, j] = 255
            elif (laplacian_image[i-1, j+1] * laplacian_image[i+1, j-1] < 0 and
                  abs(laplacian_image[i-1, j+1] - laplacian_image[i+1, j-1]) > threshold):
                output[i, j] = 255

    return output
