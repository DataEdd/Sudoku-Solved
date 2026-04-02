"""
Contour detection and manipulation implemented from scratch using numpy.

This module provides:
- Moore-Neighbor contour tracing algorithm
- Douglas-Peucker polygon simplification
- Contour utilities (area, perimeter, etc.)
"""

import numpy as np
from typing import List, Tuple, Optional


# Moore neighborhood: 8 neighbors in clockwise order starting from right
# Direction indices: 0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE
MOORE_DIRECTIONS = np.array([
    [0, 1],    # 0: East (right)
    [1, 1],    # 1: Southeast
    [1, 0],    # 2: South (down)
    [1, -1],   # 3: Southwest
    [0, -1],   # 4: West (left)
    [-1, -1],  # 5: Northwest
    [-1, 0],   # 6: North (up)
    [-1, 1],   # 7: Northeast
])


def find_contours(binary: np.ndarray) -> List[np.ndarray]:
    """
    Find all contours in a binary image using Moore-Neighbor tracing.

    The Moore-Neighbor algorithm traces the boundary of connected regions
    by walking around the edge pixels in a consistent direction.

    Algorithm:
    1. Scan image to find an unvisited boundary pixel (foreground with background neighbor)
    2. Trace the contour by following boundary pixels clockwise
    3. Stop when we return to the starting pixel
    4. Mark traced pixels and repeat for remaining regions

    Args:
        binary: Binary image (0 for background, >0 for foreground)

    Returns:
        List of contours, each as Nx2 numpy array of (row, col) coordinates

    Note:
        This finds only external contours (not holes).
    """
    # Ensure binary
    binary = (binary > 0).astype(np.uint8)
    h, w = binary.shape

    # Pad image to avoid boundary checks
    padded = np.pad(binary, 1, mode='constant', constant_values=0)

    # Track which contour pixels have been used
    used = np.zeros_like(padded, dtype=bool)

    contours = []

    # Scan for starting points
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            # Check if this is a boundary pixel:
            # - Must be foreground
            # - Must have a background neighbor to the left (entry condition)
            # - Must not be already used
            if padded[i, j] == 1 and padded[i, j-1] == 0 and not used[i, j]:
                contour = _trace_contour(padded, i, j, used)
                if len(contour) >= 3:  # Valid contour needs at least 3 points
                    # Convert from padded coords to original coords
                    contour = contour - 1
                    contours.append(contour)

    return contours


def _trace_contour(
    padded: np.ndarray,
    start_row: int,
    start_col: int,
    used: np.ndarray
) -> np.ndarray:
    """
    Trace a single contour using Moore-Neighbor algorithm.

    Args:
        padded: Padded binary image
        start_row, start_col: Starting pixel coordinates
        used: Array tracking used boundary pixels

    Returns:
        Nx2 array of contour points
    """
    contour = []
    row, col = start_row, start_col

    # Starting direction: we came from the left (West), so start checking from East
    direction = 0  # East

    # First point
    contour.append([row, col])
    used[row, col] = True

    while True:
        # Find next boundary pixel by checking neighbors in clockwise order
        # Start from (direction + 5) % 8 to backtrack and sweep clockwise
        start_dir = (direction + 5) % 8

        found = False
        for i in range(8):
            check_dir = (start_dir + i) % 8
            dr, dc = MOORE_DIRECTIONS[check_dir]
            nr, nc = row + dr, col + dc

            if padded[nr, nc] == 1:
                # Found next boundary pixel
                row, col = nr, nc
                direction = check_dir
                found = True
                break

        if not found:
            # Isolated pixel
            break

        # Check if we've returned to start
        if row == start_row and col == start_col:
            break

        contour.append([row, col])
        used[row, col] = True

        # Safety: prevent infinite loops
        if len(contour) > padded.size:
            break

    return np.array(contour, dtype=np.int32)


def approximate_polygon(
    contour: np.ndarray,
    epsilon: float
) -> np.ndarray:
    """
    Simplify a polygon using the Douglas-Peucker algorithm.

    This algorithm recursively simplifies a curve by removing points
    that are within epsilon distance of the line between endpoints.

    Algorithm:
    1. Connect first and last point with a line
    2. Find point with maximum perpendicular distance to this line
    3. If max distance > epsilon:
       - Recursively simplify [start, max_point] and [max_point, end]
    4. Else: discard all intermediate points

    Args:
        contour: Nx2 array of points
        epsilon: Maximum distance threshold

    Returns:
        Simplified polygon as Mx2 array (M <= N)

    Example:
        >>> simplified = approximate_polygon(contour, epsilon=0.02 * perimeter)
    """
    if len(contour) < 3:
        return contour.copy()

    # Find point with maximum distance from line(first, last)
    start = contour[0]
    end = contour[-1]

    # Perpendicular distances
    distances = _perpendicular_distances(contour, start, end)

    max_idx = np.argmax(distances)
    max_dist = distances[max_idx]

    if max_dist > epsilon:
        # Recursively simplify both halves
        left = approximate_polygon(contour[:max_idx + 1], epsilon)
        right = approximate_polygon(contour[max_idx:], epsilon)

        # Combine (remove duplicate point at junction)
        return np.vstack([left[:-1], right])
    else:
        # All points are within epsilon, keep only endpoints
        return np.array([start, end])


def _perpendicular_distances(
    points: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray
) -> np.ndarray:
    """
    Compute perpendicular distance from each point to a line segment.

    The distance from point P to line AB is:
        d = |cross(AP, AB)| / |AB|

    For 2D:
        cross(AP, AB) = (Px - Ax)(By - Ay) - (Py - Ay)(Bx - Ax)
    """
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-10:
        # Degenerate line: return distance to start point
        return np.linalg.norm(points - line_start, axis=1)

    # Vector from line start to each point
    point_vec = points - line_start

    # Cross product magnitude (for 2D, this is the z-component of 3D cross)
    cross = np.abs(point_vec[:, 0] * line_vec[1] - point_vec[:, 1] * line_vec[0])

    return cross / line_len


def contour_area(contour: np.ndarray) -> float:
    """
    Compute the area enclosed by a contour using the Shoelace formula.

    The Shoelace formula (also called Surveyor's formula):
        A = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|

    Args:
        contour: Nx2 array of points

    Returns:
        Area (always positive)
    """
    n = len(contour)
    if n < 3:
        return 0.0

    # Close the polygon
    x = contour[:, 1]  # col = x
    y = contour[:, 0]  # row = y

    # Shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= x[j] * y[i]

    return abs(area) / 2.0


def contour_perimeter(contour: np.ndarray, closed: bool = True) -> float:
    """
    Compute the perimeter (arc length) of a contour.

    Args:
        contour: Nx2 array of points
        closed: Whether to include distance from last to first point

    Returns:
        Perimeter length
    """
    if len(contour) < 2:
        return 0.0

    # Compute distances between consecutive points
    diffs = np.diff(contour, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)

    perimeter = lengths.sum()

    if closed:
        # Add distance from last to first
        perimeter += np.linalg.norm(contour[-1] - contour[0])

    return perimeter


def is_contour_convex(contour: np.ndarray) -> bool:
    """
    Check if a contour is convex.

    A polygon is convex if all cross products of consecutive edges
    have the same sign.

    Args:
        contour: Nx2 array of points

    Returns:
        True if convex, False otherwise
    """
    n = len(contour)
    if n < 3:
        return True

    sign = 0
    for i in range(n):
        p1 = contour[i]
        p2 = contour[(i + 1) % n]
        p3 = contour[(i + 2) % n]

        # Cross product of edge vectors
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        if cross != 0:
            if sign == 0:
                sign = 1 if cross > 0 else -1
            elif (cross > 0) != (sign > 0):
                return False

    return True
