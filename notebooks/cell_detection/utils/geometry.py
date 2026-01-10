"""
Geometry utilities for cell detection.

Provides functions for coordinate transforms, shape calculations,
and geometric operations on points and lines.
"""

from typing import List, Optional, Tuple, Union

import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.

    Uses the sum and difference of coordinates to determine position:
    - Top-left: smallest sum (x + y)
    - Bottom-right: largest sum (x + y)
    - Top-right: smallest difference (y - x)
    - Bottom-left: largest difference (y - x)

    Args:
        pts: Array of 4 points, shape (4, 2) or (4, 1, 2)

    Returns:
        Ordered array of shape (4, 2): [TL, TR, BR, BL]

    Example:
        >>> pts = np.array([[100, 200], [200, 100], [100, 100], [200, 200]])
        >>> ordered = order_corners(pts)
        >>> # ordered[0] is top-left, ordered[2] is bottom-right
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum of coordinates: TL has smallest, BR has largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    # Difference of coordinates: TR has smallest (y-x), BL has largest
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect


def compute_quad_area(corners: np.ndarray) -> float:
    """
    Compute area of a quadrilateral using the Shoelace formula.

    The Shoelace formula calculates the area of a polygon given its vertices:
    Area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|

    Args:
        corners: 4x2 array of corner points

    Returns:
        Area in square pixels

    Example:
        >>> corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        >>> area = compute_quad_area(corners)
        >>> # area == 10000.0
    """
    n = len(corners)
    area = 0.0

    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]

    return abs(area) / 2.0


def is_valid_quadrilateral(
    corners: np.ndarray,
    image_shape: Tuple[int, int],
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.95,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> bool:
    """
    Check if a quadrilateral is a plausible Sudoku grid.

    Validates based on:
    1. Area ratio relative to image (between 5% and 95%)
    2. Aspect ratio (width/height between 0.5 and 2.0)

    Args:
        corners: 4x2 array of corner points [TL, TR, BR, BL]
        image_shape: (height, width) of the image
        min_area_ratio: Minimum quad area / image area
        max_area_ratio: Maximum quad area / image area
        min_aspect_ratio: Minimum width / height
        max_aspect_ratio: Maximum width / height

    Returns:
        True if quadrilateral passes all checks
    """
    h, w = image_shape[:2]
    image_area = h * w
    quad_area = compute_quad_area(corners)

    # Check area ratio
    area_ratio = quad_area / image_area
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False

    # Check aspect ratio using TL-TR distance (width) and TL-BL distance (height)
    width = point_distance(corners[0], corners[1])
    height = point_distance(corners[0], corners[3])

    if height == 0:
        return False

    aspect = width / height
    if aspect < min_aspect_ratio or aspect > max_aspect_ratio:
        return False

    return True


def point_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two points.

    Args:
        p1: First point (x, y)
        p2: Second point (x, y)

    Returns:
        Distance in pixels
    """
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def line_angle(line: np.ndarray) -> float:
    """
    Compute the angle of a line segment in degrees.

    Args:
        line: Line as [x1, y1, x2, y2]

    Returns:
        Angle in degrees (0-180, where 0/180 is horizontal, 90 is vertical)
    """
    x1, y1, x2, y2 = line.flatten()[:4]
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

    # Normalize to 0-180 range
    if angle < 0:
        angle += 180

    return angle


def line_intersection(
    line1: np.ndarray,
    line2: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    Compute the intersection point of two lines.

    Uses the parametric form of line intersection:
    Line 1: P1 + t * (P2 - P1)
    Line 2: P3 + s * (P4 - P3)

    Args:
        line1: First line as [x1, y1, x2, y2]
        line2: Second line as [x3, y3, x4, y4]

    Returns:
        Intersection point (x, y) or None if lines are parallel

    Example:
        >>> line1 = np.array([0, 50, 100, 50])  # Horizontal
        >>> line2 = np.array([50, 0, 50, 100])  # Vertical
        >>> pt = line_intersection(line1, line2)
        >>> # pt == (50.0, 50.0)
    """
    x1, y1, x2, y2 = line1.flatten()[:4]
    x3, y3, x4, y4 = line2.flatten()[:4]

    # Direction vectors
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3

    # Cross product of direction vectors
    cross = dx1 * dy2 - dy1 * dx2

    if abs(cross) < 1e-10:
        # Lines are parallel
        return None

    # Solve for parameters
    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / cross

    # Compute intersection point
    x = x1 + t * dx1
    y = y1 + t * dy1

    return (float(x), float(y))


def extend_line(
    line: np.ndarray,
    length: float,
) -> np.ndarray:
    """
    Extend a line segment in both directions.

    Args:
        line: Line as [x1, y1, x2, y2]
        length: Total length to extend (split between both ends)

    Returns:
        Extended line as [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = line.flatten()[:4]

    # Direction vector
    dx, dy = x2 - x1, y2 - y1
    line_len = np.sqrt(dx * dx + dy * dy)

    if line_len < 1e-10:
        return line.copy()

    # Normalize and extend
    dx, dy = dx / line_len, dy / line_len
    extend = length / 2

    new_x1 = x1 - extend * dx
    new_y1 = y1 - extend * dy
    new_x2 = x2 + extend * dx
    new_y2 = y2 + extend * dy

    return np.array([new_x1, new_y1, new_x2, new_y2])


def classify_line_orientation(
    line: np.ndarray,
    angle_threshold: float = 15.0,
) -> str:
    """
    Classify a line as horizontal, vertical, or diagonal.

    Args:
        line: Line as [x1, y1, x2, y2]
        angle_threshold: Degrees from horizontal/vertical to classify

    Returns:
        "horizontal", "vertical", or "diagonal"
    """
    angle = line_angle(line)

    # Horizontal: angle near 0 or 180
    if angle < angle_threshold or angle > (180 - angle_threshold):
        return "horizontal"

    # Vertical: angle near 90
    if (90 - angle_threshold) < angle < (90 + angle_threshold):
        return "vertical"

    return "diagonal"


def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Compute the midpoint between two points.

    Args:
        p1: First point (x, y)
        p2: Second point (x, y)

    Returns:
        Midpoint as numpy array
    """
    return (np.array(p1) + np.array(p2)) / 2
