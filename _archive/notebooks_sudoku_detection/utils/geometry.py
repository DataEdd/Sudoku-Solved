"""
Geometry utilities implemented from scratch using numpy.

This module provides:
- Corner ordering for quadrilaterals
- Area computation
- Distance calculations
- Point transformations
"""

import numpy as np
from typing import Tuple, Optional


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 corner points as [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

    This uses the sum and difference of coordinates:
    - Top-Left: minimum (x + y)
    - Bottom-Right: maximum (x + y)
    - Top-Right: minimum (y - x), i.e., maximum (x - y)
    - Bottom-Left: maximum (y - x)

    Args:
        pts: 4x2 array of (row, col) or (x, y) coordinates

    Returns:
        4x2 array ordered as [TL, TR, BR, BL]

    Example:
        >>> corners = np.array([[100, 50], [50, 50], [50, 100], [100, 100]])
        >>> ordered = order_corners(corners)
        >>> # ordered[0] is top-left, ordered[2] is bottom-right
    """
    pts = np.array(pts, dtype=np.float32)

    if pts.shape[0] != 4:
        raise ValueError("Expected exactly 4 points")

    # Assuming pts are (x, y) or (col, row)
    # Sum: x + y (TL has smallest, BR has largest)
    s = pts.sum(axis=1)

    # Difference: y - x (TR has smallest/most negative, BL has largest)
    d = pts[:, 1] - pts[:, 0]

    # Order: TL, TR, BR, BL
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # Top-Left
    ordered[2] = pts[np.argmax(s)]  # Bottom-Right
    ordered[1] = pts[np.argmin(d)]  # Top-Right
    ordered[3] = pts[np.argmax(d)]  # Bottom-Left

    return ordered


def compute_quad_area(pts: np.ndarray) -> float:
    """
    Compute the area of a quadrilateral using the Shoelace formula.

    Args:
        pts: 4x2 array of corner points

    Returns:
        Area of the quadrilateral
    """
    pts = np.array(pts, dtype=np.float64)
    n = len(pts)

    # Shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i, 0] * pts[j, 1]
        area -= pts[j, 0] * pts[i, 1]

    return abs(area) / 2.0


def point_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two points.

    Args:
        p1, p2: Points as numpy arrays or tuples

    Returns:
        Distance between points
    """
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    return np.sqrt(np.sum((p2 - p1) ** 2))


def line_length(line: np.ndarray) -> float:
    """
    Compute length of a line segment.

    Args:
        line: Array [x1, y1, x2, y2]

    Returns:
        Length of the line
    """
    return point_distance(line[:2], line[2:])


def quad_side_lengths(pts: np.ndarray) -> np.ndarray:
    """
    Compute the 4 side lengths of a quadrilateral.

    Args:
        pts: 4x2 array of ordered corner points [TL, TR, BR, BL]

    Returns:
        Array of 4 side lengths [top, right, bottom, left]
    """
    lengths = np.zeros(4)
    for i in range(4):
        j = (i + 1) % 4
        lengths[i] = point_distance(pts[i], pts[j])

    return lengths


def quad_diagonal_lengths(pts: np.ndarray) -> Tuple[float, float]:
    """
    Compute the diagonal lengths of a quadrilateral.

    Args:
        pts: 4x2 array of ordered corner points [TL, TR, BR, BL]

    Returns:
        Tuple of (TL-BR diagonal, TR-BL diagonal)
    """
    d1 = point_distance(pts[0], pts[2])  # TL to BR
    d2 = point_distance(pts[1], pts[3])  # TR to BL
    return d1, d2


def is_valid_quadrilateral(
    pts: np.ndarray,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.95,
    image_area: float = None,
    min_aspect: float = 0.3,
    max_aspect: float = 3.0
) -> bool:
    """
    Check if a quadrilateral is valid for Sudoku detection.

    Criteria:
    - Area within expected range (5-95% of image)
    - Aspect ratio reasonable (not too elongated)
    - All sides have positive length

    Args:
        pts: 4x2 array of corners
        min_area_ratio: Minimum area as fraction of image
        max_area_ratio: Maximum area as fraction of image
        image_area: Total image area (for ratio calculation)
        min_aspect: Minimum width/height ratio
        max_aspect: Maximum width/height ratio

    Returns:
        True if valid, False otherwise
    """
    if len(pts) != 4:
        return False

    # Compute area
    area = compute_quad_area(pts)

    if image_area is not None:
        area_ratio = area / image_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            return False

    # Check side lengths
    sides = quad_side_lengths(pts)
    if np.any(sides < 1):
        return False

    # Check aspect ratio (width/height approximation)
    width = (sides[0] + sides[2]) / 2  # Average of top and bottom
    height = (sides[1] + sides[3]) / 2  # Average of left and right

    if height > 0:
        aspect = width / height
        if aspect < min_aspect or aspect > max_aspect:
            return False

    return True


def centroid(pts: np.ndarray) -> np.ndarray:
    """
    Compute the centroid (center of mass) of a polygon.

    Args:
        pts: Nx2 array of points

    Returns:
        Centroid as (x, y) array
    """
    return pts.mean(axis=0)


def bounding_box(pts: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute axis-aligned bounding box of points.

    Args:
        pts: Nx2 array of points

    Returns:
        Tuple (x, y, width, height)
    """
    min_x = pts[:, 0].min()
    min_y = pts[:, 1].min()
    max_x = pts[:, 0].max()
    max_y = pts[:, 1].max()

    return int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Test if a point is inside a polygon using ray casting.

    Shoots a ray from the point to infinity and counts intersections.
    Odd = inside, Even = outside.

    Args:
        point: (x, y) point
        polygon: Nx2 array of polygon vertices

    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]

        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside

        p1x, p1y = p2x, p2y

    return inside
