"""Hough line-based detection for sudoku grids."""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import DetectionConfig
from .contour_detection import compute_centeredness


def detect_lines(
    edges: np.ndarray,
    config: DetectionConfig,
) -> Optional[np.ndarray]:
    """Detect lines in an edge image using probabilistic Hough transform.

    Args:
        edges: Binary edge image (output from Canny).
        config: Detection configuration with Hough parameters.

    Returns:
        Array of lines with shape (N, 1, 4) where each line is (x1, y1, x2, y2).
        Returns None if no lines detected.
    """
    if edges is None or edges.size == 0 or np.sum(edges) == 0:
        return None

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config.hough_threshold,
        minLineLength=config.hough_min_line_length,
        maxLineGap=config.hough_max_line_gap,
    )

    return lines


def classify_lines(
    lines: np.ndarray,
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """Classify lines as horizontal or vertical based on their angle.

    Args:
        lines: Array of lines with shape (N, 1, 4).

    Returns:
        Tuple of (horizontal_lines, vertical_lines).
        Each line is stored as tuple (x1, y1, x2, y2).
        Horizontal: angle < 45° or angle > 135°
        Vertical: 45° <= angle <= 135°
    """
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Compute angle in radians and convert to degrees
        angle = math.atan2(y2 - y1, x2 - x1)
        angle_deg = math.degrees(angle)

        # Normalize to 0-180 range
        if angle_deg < 0:
            angle_deg += 180

        # Classify based on angle
        if angle_deg < 45 or angle_deg > 135:
            horizontal_lines.append((x1, y1, x2, y2))
        else:
            vertical_lines.append((x1, y1, x2, y2))

    return horizontal_lines, vertical_lines


def compute_line_intercept(
    line: Tuple[int, int, int, int],
    is_horizontal: bool,
) -> float:
    """Compute the intercept value for a line.

    For horizontal lines, returns the average y-coordinate (y-intercept).
    For vertical lines, returns the average x-coordinate (x-intercept).

    Args:
        line: Line as tuple (x1, y1, x2, y2).
        is_horizontal: True if line is horizontal, False if vertical.

    Returns:
        Intercept value (average y for horizontal, average x for vertical).
    """
    x1, y1, x2, y2 = line

    if is_horizontal:
        return (y1 + y2) / 2
    else:
        return (x1 + x2) / 2


def cluster_lines(
    lines: List[Tuple[int, int, int, int]],
    threshold: float,
    is_horizontal: bool,
) -> List[Tuple[int, int, int, int]]:
    """Cluster nearby lines using gap-based clustering.

    Groups lines where the gap between adjacent lines (sorted by intercept)
    exceeds the threshold. For each cluster, selects the line closest to the
    median intercept as the representative.

    This approach prevents chain-merging that occurs with simple greedy
    clustering where lines A-B-C-D could all merge if each pair is within
    threshold, even if A and D are far apart.

    Args:
        lines: List of lines as tuples (x1, y1, x2, y2).
        threshold: Minimum gap between clusters (lines closer than this merge).
        is_horizontal: True if lines are horizontal, False if vertical.

    Returns:
        List of representative lines after clustering.
    """
    if not lines:
        return []

    # Extract intercepts and pair with lines
    if is_horizontal:
        intercepts = [(line[1] + line[3]) / 2 for line in lines]
    else:
        intercepts = [(line[0] + line[2]) / 2 for line in lines]

    # Sort by intercept
    sorted_pairs = sorted(zip(intercepts, lines), key=lambda x: x[0])

    # Group into clusters where gap > threshold
    clusters = []
    current_cluster = [sorted_pairs[0]]

    for i in range(1, len(sorted_pairs)):
        current_intercept = sorted_pairs[i][0]
        prev_intercept = sorted_pairs[i - 1][0]

        if current_intercept - prev_intercept > threshold:
            # Gap found - finalize current cluster and start new one
            clusters.append(current_cluster)
            current_cluster = [sorted_pairs[i]]
        else:
            current_cluster.append(sorted_pairs[i])

    # Don't forget last cluster
    clusters.append(current_cluster)

    # For each cluster, take the line closest to median intercept
    result = []
    for cluster in clusters:
        intercepts_in_cluster = [p[0] for p in cluster]
        median_idx = len(intercepts_in_cluster) // 2
        median_intercept = sorted(intercepts_in_cluster)[median_idx]

        # Find line closest to median
        best_pair = min(cluster, key=lambda p: abs(p[0] - median_intercept))
        result.append(best_pair[1])

    return result


def _merge_lines(
    lines: List[Tuple[int, int, int, int]],
    is_horizontal: bool,
) -> Tuple[int, int, int, int]:
    """Merge multiple lines into a single representative line.

    Args:
        lines: List of lines to merge.
        is_horizontal: True if lines are horizontal, False if vertical.

    Returns:
        Single merged line as tuple (x1, y1, x2, y2).
    """
    if len(lines) == 1:
        return lines[0]

    # Compute average endpoints
    x1_avg = int(sum(l[0] for l in lines) / len(lines))
    y1_avg = int(sum(l[1] for l in lines) / len(lines))
    x2_avg = int(sum(l[2] for l in lines) / len(lines))
    y2_avg = int(sum(l[3] for l in lines) / len(lines))

    # For horizontal lines, extend x range; for vertical, extend y range
    if is_horizontal:
        x1_min = min(min(l[0], l[2]) for l in lines)
        x2_max = max(max(l[0], l[2]) for l in lines)
        y_avg = (y1_avg + y2_avg) // 2
        return (x1_min, y_avg, x2_max, y_avg)
    else:
        y1_min = min(min(l[1], l[3]) for l in lines)
        y2_max = max(max(l[1], l[3]) for l in lines)
        x_avg = (x1_avg + x2_avg) // 2
        return (x_avg, y1_min, x_avg, y2_max)


def validate_grid_lines(
    h_lines: List[Tuple[int, int, int, int]],
    v_lines: List[Tuple[int, int, int, int]],
    min_count: int,
    max_count: int,
) -> bool:
    """Validate that line counts are consistent with a sudoku grid.

    A sudoku grid has 10 horizontal and 10 vertical lines (including borders).
    We allow some tolerance for detection errors.

    Args:
        h_lines: List of horizontal lines.
        v_lines: List of vertical lines.
        min_count: Minimum acceptable line count.
        max_count: Maximum acceptable line count.

    Returns:
        True if both horizontal and vertical line counts are within range.
    """
    h_count = len(h_lines)
    v_count = len(v_lines)

    return (min_count <= h_count <= max_count and
            min_count <= v_count <= max_count)


def line_intersection(
    line1: Tuple[int, int, int, int],
    line2: Tuple[int, int, int, int],
) -> Optional[Tuple[float, float]]:
    """Compute the intersection point of two lines.

    Uses linear algebra to solve the system of equations.

    Args:
        line1: First line as tuple (x1, y1, x2, y2).
        line2: Second line as tuple (x1, y1, x2, y2).

    Returns:
        Intersection point as (x, y) tuple, or None if lines are parallel.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Line 1: (x2-x1, y2-y1) direction
    # Line 2: (x4-x3, y4-y3) direction
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        # Lines are parallel
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    # Compute intersection point
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)

    return (px, py)


def compute_all_intersections(
    h_lines: List[Tuple[int, int, int, int]],
    v_lines: List[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Compute all intersection points between horizontal and vertical lines.

    Args:
        h_lines: List of horizontal lines.
        v_lines: List of vertical lines.
        image_shape: Image shape as (height, width).

    Returns:
        Array of intersection points with shape (N, 2).
        Points outside image bounds are filtered out.
    """
    height, width = image_shape[:2]
    intersections = []

    for h_line in h_lines:
        for v_line in v_lines:
            point = line_intersection(h_line, v_line)

            if point is not None:
                x, y = point

                # Filter out points outside image bounds
                if 0 <= x < width and 0 <= y < height:
                    intersections.append([x, y])

    if not intersections:
        return np.array([]).reshape(0, 2)

    return np.array(intersections, dtype=np.float32)


def extract_outer_corners(
    intersections: np.ndarray,
) -> Optional[np.ndarray]:
    """Extract the 4 outer corners from intersection points.

    Uses sum and difference method to find extreme points:
    - Top-left: minimum (x + y)
    - Bottom-right: maximum (x + y)
    - Top-right: maximum (x - y)
    - Bottom-left: minimum (x - y)

    Args:
        intersections: Array of intersection points with shape (N, 2).

    Returns:
        Array of shape (4, 2) with ordered corners [TL, TR, BR, BL],
        or None if fewer than 4 intersections.
    """
    if intersections is None or len(intersections) < 4:
        return None

    # Compute sum and difference for each point
    sums = intersections[:, 0] + intersections[:, 1]
    diffs = intersections[:, 0] - intersections[:, 1]

    # Find extreme points
    top_left_idx = np.argmin(sums)
    bottom_right_idx = np.argmax(sums)
    top_right_idx = np.argmax(diffs)
    bottom_left_idx = np.argmin(diffs)

    # Extract corners in order: TL, TR, BR, BL
    corners = np.array([
        intersections[top_left_idx],
        intersections[top_right_idx],
        intersections[bottom_right_idx],
        intersections[bottom_left_idx],
    ], dtype=np.float32)

    return corners


def detect_hough_path(
    edges: np.ndarray,
    config: DetectionConfig,
    image_shape: Tuple[int, int],
) -> List[dict]:
    """Detect sudoku grids using Hough line detection.

    Runs full Hough detection pipeline:
    1. Detect lines
    2. Classify into horizontal/vertical
    3. Cluster lines
    4. Validate line counts
    5. Compute intersections
    6. Extract outer corners
    7. Compute centeredness score

    Args:
        edges: Binary edge image.
        config: Detection configuration parameters.
        image_shape: Original image shape as (height, width).

    Returns:
        List of detection results, sorted by centeredness score (descending).
        Each result is a dict with:
        - "corners": np.ndarray of shape (4, 2) with corner coordinates
        - "score": float centeredness score (0.0 to 1.0)
        - "method": "hough"
        Returns empty list if detection fails.
    """
    height, width = image_shape[:2]

    # Step 1: Detect lines
    lines = detect_lines(edges, config)
    if lines is None or len(lines) == 0:
        return []

    # Step 2: Classify lines
    h_lines, v_lines = classify_lines(lines)

    if not h_lines or not v_lines:
        return []

    # Step 3: Cluster lines
    # Threshold based on expected cell size:
    # - Sudoku has 9 cells, so ~10 lines
    # - Cell size ≈ image_dimension / 9
    # - Threshold = cell_size / 4 to separate distinct lines
    # - Default divisor is 36 (configurable via config.line_cluster_divisor)
    h_threshold = max(1, height / config.line_cluster_divisor)
    v_threshold = max(1, width / config.line_cluster_divisor)

    h_clustered = cluster_lines(h_lines, h_threshold, is_horizontal=True)
    v_clustered = cluster_lines(v_lines, v_threshold, is_horizontal=False)

    # Step 4: Validate line counts
    if not validate_grid_lines(
        h_clustered, v_clustered,
        config.min_line_count,
        config.max_line_count
    ):
        return []

    # Step 5: Compute intersections
    intersections = compute_all_intersections(
        h_clustered, v_clustered, image_shape
    )

    if len(intersections) < 4:
        return []

    # Step 6: Extract outer corners
    corners = extract_outer_corners(intersections)

    if corners is None:
        return []

    # Step 7: Compute centeredness score
    score = compute_centeredness(corners, image_shape)

    result = {
        "corners": corners,
        "score": score,
        "method": "hough",
    }

    return [result]
