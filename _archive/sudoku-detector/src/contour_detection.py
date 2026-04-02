"""Contour-based detection for sudoku grids."""

from typing import List, Tuple

import cv2
import numpy as np

from .config import DetectionConfig


def find_quadrilateral_contours(
    binary: np.ndarray,
    min_area_ratio: float,
    max_area_ratio: float,
    epsilon_factor: float = 0.02,
) -> List[np.ndarray]:
    """Find quadrilateral contours in a binary image.

    Finds external contours, filters by area, and approximates to polygons.
    Only returns 4-vertex polygons (quadrilaterals).

    Args:
        binary: Binary input image (white objects on black background).
        min_area_ratio: Minimum contour area as ratio of image area.
        max_area_ratio: Maximum contour area as ratio of image area.
        epsilon_factor: Factor for contour approximation (epsilon = factor * perimeter).

    Returns:
        List of quadrilaterals as numpy arrays of shape (4, 2).
        Each array contains the 4 corner points of a quadrilateral.
        Returns empty list if no quadrilaterals found.
    """
    # Handle empty or all-black images
    if binary is None or binary.size == 0 or np.sum(binary) == 0:
        return []

    image_area = binary.shape[0] * binary.shape[1]
    min_area = min_area_ratio * image_area
    max_area = max_area_ratio * image_area

    # Find external contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    quadrilaterals = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by area
        if area < min_area or area > max_area:
            continue

        # Approximate to polygon
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        # Keep only quadrilaterals
        if len(approx) == 4:
            # Reshape from (4, 1, 2) to (4, 2)
            quad = approx.reshape(4, 2)
            quadrilaterals.append(quad)

    return quadrilaterals


def _compute_angle(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
    """Compute the angle at vertex formed by points p1-vertex-p2.

    Args:
        p1: First point.
        vertex: The vertex point where angle is measured.
        p2: Second point.

    Returns:
        Angle in degrees (0-180).
    """
    v1 = p1 - vertex
    v2 = p2 - vertex

    # Compute dot product and magnitudes
    dot = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)

    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0

    # Clamp to [-1, 1] to handle floating point errors
    cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))

    return angle


def validate_quadrilateral(
    quad: np.ndarray,
    min_aspect: float,
    max_aspect: float,
    min_angle: float = 45.0,
    max_angle: float = 135.0,
) -> bool:
    """Validate a quadrilateral based on geometric properties.

    Checks:
    1. Convexity - must be a convex polygon
    2. Aspect ratio - bounding rect aspect ratio within [min_aspect, max_aspect]
    3. Interior angles - all angles must be between min_angle and max_angle

    Args:
        quad: Quadrilateral as numpy array of shape (4, 2).
        min_aspect: Minimum acceptable aspect ratio.
        max_aspect: Maximum acceptable aspect ratio.
        min_angle: Minimum acceptable interior angle in degrees.
        max_angle: Maximum acceptable interior angle in degrees.

    Returns:
        True if quadrilateral passes all validation checks, False otherwise.
    """
    # Ensure correct shape
    if quad.shape != (4, 2):
        return False

    # Check convexity
    # cv2.isContourConvex expects shape (n, 1, 2) or (n, 2)
    quad_for_cv = quad.reshape(4, 1, 2).astype(np.float32)
    if not cv2.isContourConvex(quad_for_cv):
        return False

    # Check aspect ratio using bounding rect
    x, y, w, h = cv2.boundingRect(quad_for_cv)
    if h == 0:
        return False

    aspect_ratio = w / h
    if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
        return False

    # Check interior angles
    # Points are in order, so angles are at each vertex
    for i in range(4):
        p1 = quad[(i - 1) % 4].astype(np.float64)
        vertex = quad[i].astype(np.float64)
        p2 = quad[(i + 1) % 4].astype(np.float64)

        angle = _compute_angle(p1, vertex, p2)

        if angle < min_angle or angle > max_angle:
            return False

    return True


def compute_centeredness(
    quad: np.ndarray,
    image_shape: Tuple[int, int],
) -> float:
    """Compute how centered a quadrilateral is in the image.

    Args:
        quad: Quadrilateral as numpy array of shape (4, 2).
        image_shape: Image shape as (height, width).

    Returns:
        Centeredness score from 0.0 to 1.0.
        1.0 means perfectly centered, 0.0 means at corner.
    """
    height, width = image_shape[:2]

    # Calculate centroid of quadrilateral (mean of 4 corners)
    quad_centroid = np.mean(quad, axis=0)

    # Calculate image center
    image_center = np.array([width / 2, height / 2])

    # Calculate distance from quad centroid to image center
    distance = np.linalg.norm(quad_centroid - image_center)

    # Calculate max possible distance (corner to center)
    max_distance = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)

    # Avoid division by zero
    if max_distance == 0:
        return 1.0

    # Score: 1.0 means perfectly centered, 0.0 means at corner
    score = 1.0 - (distance / max_distance)

    # Clamp to [0, 1] to handle edge cases
    return max(0.0, min(1.0, score))


def detect_contour_path(
    binary: np.ndarray,
    config: DetectionConfig,
) -> List[dict]:
    """Detect sudoku grids using contour-based detection.

    Finds quadrilateral contours, validates them, and scores by centeredness.

    Args:
        binary: Binary input image (white objects on black background).
        config: Detection configuration parameters.

    Returns:
        List of detection results, sorted by centeredness score (descending).
        Each result is a dict with:
        - "corners": np.ndarray of shape (4, 2) with corner coordinates
        - "score": float centeredness score (0.0 to 1.0)
        - "method": "contour"
        Returns empty list if no valid quadrilaterals found.
    """
    # Find quadrilateral contours
    quads = find_quadrilateral_contours(
        binary,
        config.min_area_ratio,
        config.max_area_ratio,
        config.contour_epsilon_factor,
    )

    if not quads:
        return []

    results = []

    for quad in quads:
        # Validate quadrilateral
        if not validate_quadrilateral(
            quad,
            config.min_aspect_ratio,
            config.max_aspect_ratio,
            config.min_interior_angle,
            config.max_interior_angle,
        ):
            continue

        # Compute centeredness score
        score = compute_centeredness(quad, binary.shape)

        results.append({
            "corners": quad,
            "score": score,
            "method": "contour",
        })

    # Sort by score descending (most centered first)
    results.sort(key=lambda x: x["score"], reverse=True)

    return results
