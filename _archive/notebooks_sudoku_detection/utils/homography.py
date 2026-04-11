"""
Homography and perspective transform implemented from scratch using numpy.

This module provides:
- Homography matrix computation using DLT algorithm
- Point transformation using homography
- Perspective warping with bilinear interpolation

Mathematical Background:
=======================

A homography (projective transformation) maps points from one plane to another.
In homogeneous coordinates:

    [x']     [h11 h12 h13] [x]
    [y']  =  [h21 h22 h23] [y]
    [w']     [h31 h32 h33] [1]

The actual 2D coordinates are obtained by:
    x'_actual = x' / w'
    y'_actual = y' / w'

The homography matrix H has 8 degrees of freedom (9 elements, 1 for scale).
Given 4 point correspondences, we can solve for H.

DLT Algorithm (Direct Linear Transform):
========================================

For each point correspondence (x,y) -> (x',y'), we have:

    x' = (h11*x + h12*y + h13) / (h31*x + h32*y + h33)
    y' = (h21*x + h22*y + h23) / (h31*x + h32*y + h33)

Rearranging:
    x'*(h31*x + h32*y + h33) = h11*x + h12*y + h13
    y'*(h31*x + h32*y + h33) = h21*x + h22*y + h23

This gives us two linear equations per point:
    -x*h11 - y*h12 - h13 + x'*x*h31 + x'*y*h32 + x'*h33 = 0
    -x*h21 - y*h22 - h23 + y'*x*h31 + y'*y*h32 + y'*h33 = 0

In matrix form: A * h = 0

For 4 points, A is 8x9. We solve using SVD.
"""

import numpy as np
from typing import Tuple, Optional


def compute_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Compute 3x3 homography matrix using the DLT algorithm.

    Given 4 point correspondences, finds the homography H such that:
        dst = H @ src (in homogeneous coordinates)

    Args:
        src_pts: 4x2 array of source points
        dst_pts: 4x2 array of destination points

    Returns:
        3x3 homography matrix

    Example:
        >>> # Map corners of quadrilateral to square
        >>> src = np.array([[10, 20], [100, 25], [95, 110], [15, 105]])
        >>> dst = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        >>> H = compute_homography(src, dst)
    """
    src_pts = np.array(src_pts, dtype=np.float64)
    dst_pts = np.array(dst_pts, dtype=np.float64)

    if src_pts.shape != (4, 2) or dst_pts.shape != (4, 2):
        raise ValueError("Expected 4 points each for source and destination")

    # Build the A matrix (8x9)
    # For each point correspondence, we get 2 rows
    A = np.zeros((8, 9), dtype=np.float64)

    for i in range(4):
        x, y = src_pts[i]
        xp, yp = dst_pts[i]

        # Row for x' equation:
        # [-x, -y, -1, 0, 0, 0, x'*x, x'*y, x']
        A[2*i] = [-x, -y, -1, 0, 0, 0, xp*x, xp*y, xp]

        # Row for y' equation:
        # [0, 0, 0, -x, -y, -1, y'*x, y'*y, y']
        A[2*i + 1] = [0, 0, 0, -x, -y, -1, yp*x, yp*y, yp]

    # Solve A*h = 0 using SVD
    # The solution h is the right singular vector corresponding to
    # the smallest singular value
    U, S, Vt = np.linalg.svd(A)

    # h is the last row of Vt (corresponds to smallest singular value)
    h = Vt[-1]

    # Reshape to 3x3 matrix
    H = h.reshape(3, 3)

    # Normalize so H[2,2] = 1 (if non-zero)
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]

    return H


def apply_homography(H: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Transform a single 2D point using a homography matrix.

    Args:
        H: 3x3 homography matrix
        point: 2D point as (x, y) array

    Returns:
        Transformed 2D point

    Mathematical operation:
        [x']     [h11 h12 h13] [x]
        [y']  =  [h21 h22 h23] [y]
        [w']     [h31 h32 h33] [1]

        result = (x'/w', y'/w')
    """
    point = np.array(point, dtype=np.float64)

    # Convert to homogeneous coordinates
    p_homo = np.array([point[0], point[1], 1.0])

    # Apply transformation
    transformed = H @ p_homo

    # Convert back to 2D (divide by w)
    w = transformed[2]
    if abs(w) < 1e-10:
        # Point at infinity
        return np.array([np.inf, np.inf])

    return transformed[:2] / w


def apply_homography_batch(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Transform multiple 2D points using a homography matrix.

    Args:
        H: 3x3 homography matrix
        points: Nx2 array of points

    Returns:
        Nx2 array of transformed points
    """
    points = np.array(points, dtype=np.float64)
    n = len(points)

    # Convert to homogeneous (Nx3)
    ones = np.ones((n, 1))
    p_homo = np.hstack([points, ones])

    # Apply transformation: (3x3) @ (3xN) = (3xN)
    transformed = (H @ p_homo.T).T

    # Convert back to 2D
    w = transformed[:, 2:3]
    w[np.abs(w) < 1e-10] = 1e-10  # Avoid division by zero

    return transformed[:, :2] / w


def bilinear_interpolate(
    image: np.ndarray,
    x: float,
    y: float,
    border_value: float = 0
) -> float:
    """
    Sample image at non-integer coordinates using bilinear interpolation.

    Bilinear interpolation uses the 4 nearest pixels to estimate the
    value at a non-integer location.

    For point (x, y) where:
        x = i + a  (i = floor(x), 0 <= a < 1)
        y = j + b  (j = floor(y), 0 <= b < 1)

    The interpolated value is:
        f(x,y) = (1-a)(1-b)*f[j,i] + a(1-b)*f[j,i+1]
               + (1-a)*b*f[j+1,i] + a*b*f[j+1,i+1]

    Visual representation:
        Q12 ---- Q22
         |   P    |     P = query point
         |        |     Q11, Q12, Q21, Q22 = neighboring pixels
        Q11 ---- Q21

    Args:
        image: 2D grayscale image
        x: x-coordinate (column)
        y: y-coordinate (row)
        border_value: Value to use for out-of-bounds coordinates

    Returns:
        Interpolated pixel value
    """
    h, w = image.shape

    # Integer and fractional parts
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    # Fractional parts (weights)
    a = x - x0
    b = y - y0

    # Get pixel values (with boundary handling)
    def get_pixel(row, col):
        if 0 <= row < h and 0 <= col < w:
            return float(image[row, col])
        return border_value

    # Four neighboring pixels
    Q11 = get_pixel(y0, x0)
    Q21 = get_pixel(y0, x1)
    Q12 = get_pixel(y1, x0)
    Q22 = get_pixel(y1, x1)

    # Bilinear interpolation formula
    value = (
        (1 - a) * (1 - b) * Q11 +
        a * (1 - b) * Q21 +
        (1 - a) * b * Q12 +
        a * b * Q22
    )

    return value


def warp_perspective(
    image: np.ndarray,
    H: np.ndarray,
    output_size: Tuple[int, int],
    border_value: int = 0
) -> np.ndarray:
    """
    Warp an image using a homography matrix.

    For each pixel in the output image, we:
    1. Apply inverse homography to get source coordinates
    2. Use bilinear interpolation to sample the source image

    This is called "backward mapping" - we iterate over destination pixels
    and find where they came from in the source.

    Args:
        image: Input image (grayscale or color)
        H: 3x3 homography matrix (src -> dst)
        output_size: (width, height) of output image
        border_value: Value for out-of-bounds pixels

    Returns:
        Warped image

    Performance Note:
        This implementation uses explicit loops for clarity.
        A vectorized version is provided as warp_perspective_vectorized().
    """
    out_w, out_h = output_size

    # Handle grayscale vs color
    if len(image.shape) == 2:
        output = np.full((out_h, out_w), border_value, dtype=image.dtype)
        is_color = False
    else:
        output = np.full((out_h, out_w, image.shape[2]), border_value, dtype=image.dtype)
        is_color = True

    # Compute inverse homography (to map dst -> src)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        # Singular matrix
        return output

    # Iterate over output pixels
    for y_dst in range(out_h):
        for x_dst in range(out_w):
            # Apply inverse homography to get source coordinates
            src_point = apply_homography(H_inv, np.array([x_dst, y_dst]))
            x_src, y_src = src_point

            # Skip if source is at infinity
            if np.isinf(x_src) or np.isinf(y_src):
                continue

            # Bilinear interpolation
            if is_color:
                for c in range(image.shape[2]):
                    output[y_dst, x_dst, c] = bilinear_interpolate(
                        image[:, :, c], x_src, y_src, border_value
                    )
            else:
                output[y_dst, x_dst] = bilinear_interpolate(
                    image, x_src, y_src, border_value
                )

    return output.astype(image.dtype)


def warp_perspective_vectorized(
    image: np.ndarray,
    H: np.ndarray,
    output_size: Tuple[int, int],
    border_value: int = 0
) -> np.ndarray:
    """
    Vectorized perspective warp for better performance.

    Uses numpy broadcasting instead of explicit loops.
    """
    out_w, out_h = output_size

    # Create meshgrid of output coordinates
    x_dst = np.arange(out_w)
    y_dst = np.arange(out_h)
    X_dst, Y_dst = np.meshgrid(x_dst, y_dst)

    # Stack into Nx2 array of all destination points
    dst_coords = np.stack([X_dst.ravel(), Y_dst.ravel()], axis=1)

    # Inverse homography
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        if len(image.shape) == 2:
            return np.full((out_h, out_w), border_value, dtype=image.dtype)
        else:
            return np.full((out_h, out_w, image.shape[2]), border_value, dtype=image.dtype)

    # Apply inverse homography to all points at once
    src_coords = apply_homography_batch(H_inv, dst_coords)
    X_src = src_coords[:, 0].reshape(out_h, out_w)
    Y_src = src_coords[:, 1].reshape(out_h, out_w)

    # Vectorized bilinear interpolation
    h, w = image.shape[:2]

    # Integer and fractional parts
    X0 = np.floor(X_src).astype(int)
    Y0 = np.floor(Y_src).astype(int)
    X1 = X0 + 1
    Y1 = Y0 + 1

    # Weights
    Xa = X_src - X0
    Ya = Y_src - Y0

    # Clip coordinates
    X0_clip = np.clip(X0, 0, w - 1)
    X1_clip = np.clip(X1, 0, w - 1)
    Y0_clip = np.clip(Y0, 0, h - 1)
    Y1_clip = np.clip(Y1, 0, h - 1)

    # Mask for valid coordinates
    valid = (X0 >= 0) & (X1 < w) & (Y0 >= 0) & (Y1 < h)

    # Handle grayscale vs color
    if len(image.shape) == 2:
        output = np.full((out_h, out_w), border_value, dtype=np.float64)

        # Get pixel values
        Q11 = image[Y0_clip, X0_clip].astype(np.float64)
        Q21 = image[Y0_clip, X1_clip].astype(np.float64)
        Q12 = image[Y1_clip, X0_clip].astype(np.float64)
        Q22 = image[Y1_clip, X1_clip].astype(np.float64)

        # Bilinear interpolation
        interp = (
            (1 - Xa) * (1 - Ya) * Q11 +
            Xa * (1 - Ya) * Q21 +
            (1 - Xa) * Ya * Q12 +
            Xa * Ya * Q22
        )

        output[valid] = interp[valid]
        return output.astype(image.dtype)

    else:
        output = np.full((out_h, out_w, image.shape[2]), border_value, dtype=np.float64)

        for c in range(image.shape[2]):
            channel = image[:, :, c]

            Q11 = channel[Y0_clip, X0_clip].astype(np.float64)
            Q21 = channel[Y0_clip, X1_clip].astype(np.float64)
            Q12 = channel[Y1_clip, X0_clip].astype(np.float64)
            Q22 = channel[Y1_clip, X1_clip].astype(np.float64)

            interp = (
                (1 - Xa) * (1 - Ya) * Q11 +
                Xa * (1 - Ya) * Q21 +
                (1 - Xa) * Ya * Q12 +
                Xa * Ya * Q22
            )

            output[:, :, c][valid] = interp[valid]

        return output.astype(image.dtype)
