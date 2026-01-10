"""
Standard Hough Transform for Sudoku grid line detection.

Mathematical foundation:
- Every line can be represented as: ρ = x·cos(θ) + y·sin(θ)
- ρ = perpendicular distance from origin to line
- θ = angle of the perpendicular

Algorithm:
1. For each edge pixel, vote for all possible (ρ, θ) pairs
2. Find peaks in accumulator space = detected lines
3. Classify lines as horizontal or vertical by angle
4. Cluster similar lines to get exactly 10 in each direction
5. Compute intersection points for cell extraction
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.config import get_settings


@dataclass
class LineDetectionResult:
    """Result of line detection with visualization data."""

    # Detected lines
    all_lines: np.ndarray          # All lines from Hough: [[x1,y1,x2,y2], ...]
    horizontal_lines: np.ndarray   # Classified horizontal lines
    vertical_lines: np.ndarray     # Classified vertical lines

    # Clustered positions (10 each for Sudoku)
    h_positions: List[float]       # Y-coordinates of 10 horizontal lines
    v_positions: List[float]       # X-coordinates of 10 vertical lines

    # Grid intersections
    intersections: np.ndarray      # 10x10x2 array of (x,y) coordinates

    # Confidence score
    confidence: float

    # Visualization images
    annotated_image: Optional[np.ndarray] = None


def detect_lines(
    thresh: np.ndarray,
    rho: float = 1,
    theta: float = np.pi / 180,
    threshold: int = 100,
    min_line_length: int = 50,
    max_line_gap: int = 10
) -> np.ndarray:
    """
    Detect lines using Probabilistic Hough Transform.

    Hough Transform basics:
    - Each edge pixel (x, y) could lie on infinite lines
    - In polar form: ρ = x·cos(θ) + y·sin(θ)
    - For each pixel, we vote for all (ρ, θ) that pass through it
    - Lines = peaks in the (ρ, θ) accumulator space

    Probabilistic Hough (HoughLinesP) is faster:
    - Randomly samples edge pixels
    - Returns line segments [x1, y1, x2, y2] directly

    Args:
        thresh: Binary image with edges as white pixels
        rho: Distance resolution in pixels (1 = 1 pixel)
        theta: Angle resolution in radians (π/180 = 1 degree)
        threshold: Min votes to consider a line
        min_line_length: Min length of line segment to detect
        max_line_gap: Max gap between segments to join

    Returns:
        Array of lines: [[x1, y1, x2, y2], ...]
    """
    lines = cv2.HoughLinesP(
        thresh,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lines is None:
        return np.array([])

    return lines.reshape(-1, 4)


def compute_line_angle(line: np.ndarray) -> float:
    """
    Compute angle of a line segment in degrees.

    θ = arctan2(Δy, Δx) × 180/π

    Angle is normalized to [-90, 90] range:
    - 0° = horizontal line
    - ±90° = vertical line

    Args:
        line: [x1, y1, x2, y2]

    Returns:
        Angle in degrees, range [-90, 90]
    """
    x1, y1, x2, y2 = line
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

    # Normalize to [-90, 90]
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return angle


def classify_lines(
    lines: np.ndarray,
    angle_threshold: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify lines as horizontal or vertical based on angle.

    Classification rules:
    - Horizontal: |angle| < threshold (near 0°)
    - Vertical: ||angle| - 90°| < threshold (near ±90°)
    - Other: discarded (diagonal lines)

    Args:
        lines: Array of lines [[x1,y1,x2,y2], ...]
        angle_threshold: Max deviation from 0° or 90° in degrees

    Returns:
        Tuple of (horizontal_lines, vertical_lines)
    """
    if len(lines) == 0:
        return np.array([]), np.array([])

    horizontal = []
    vertical = []

    for line in lines:
        angle = compute_line_angle(line)

        if abs(angle) < angle_threshold:
            horizontal.append(line)
        elif abs(abs(angle) - 90) < angle_threshold:
            vertical.append(line)
        # else: diagonal line, discard

    return np.array(horizontal), np.array(vertical)


def get_line_position(line: np.ndarray, is_horizontal: bool) -> float:
    """
    Extract the relevant position coordinate from a line.

    For horizontal lines: average Y coordinate
    For vertical lines: average X coordinate

    Args:
        line: [x1, y1, x2, y2]
        is_horizontal: True for horizontal lines

    Returns:
        Position value (Y for horizontal, X for vertical)
    """
    x1, y1, x2, y2 = line
    if is_horizontal:
        return (y1 + y2) / 2
    else:
        return (x1 + x2) / 2


def cluster_lines(
    lines: np.ndarray,
    is_horizontal: bool,
    distance_threshold: float = 20.0
) -> List[float]:
    """
    Cluster nearby lines and return representative positions.

    Multiple lines may be detected for a single grid line due to:
    - Thick lines detected as multiple parallel lines
    - Noise causing duplicate detections

    Algorithm:
    1. Extract position from each line (Y for H, X for V)
    2. Sort positions
    3. Merge positions within distance_threshold
    4. Return mean of each cluster

    Args:
        lines: Array of lines (all horizontal or all vertical)
        is_horizontal: True if lines are horizontal
        distance_threshold: Max distance to merge lines

    Returns:
        List of clustered positions
    """
    if len(lines) == 0:
        return []

    # Extract positions
    positions = [get_line_position(line, is_horizontal) for line in lines]
    positions = sorted(positions)

    # Cluster nearby positions
    clusters = []
    current_cluster = [positions[0]]

    for pos in positions[1:]:
        if pos - current_cluster[-1] < distance_threshold:
            current_cluster.append(pos)
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [pos]

    clusters.append(np.mean(current_cluster))

    return clusters


def interpolate_grid_lines(
    positions: List[float],
    target_count: int = 10
) -> List[float]:
    """
    Interpolate or extrapolate to get exactly target_count lines.

    Sudoku grid has exactly 10 horizontal and 10 vertical lines
    (outer border + 8 internal divisions).

    If we detect fewer, we interpolate based on median spacing.
    If we detect more, we take the most evenly spaced subset.

    Args:
        positions: Detected line positions
        target_count: Number of lines needed (10 for Sudoku)

    Returns:
        List of exactly target_count evenly spaced positions
    """
    if len(positions) < 2:
        raise ValueError("Need at least 2 lines to interpolate")

    # Calculate spacing statistics
    spacings = np.diff(positions)
    median_spacing = np.median(spacings)

    # Use detected boundaries
    start = positions[0]
    end = positions[-1]

    # Check if we should use detected range or estimate from spacing
    detected_span = end - start
    expected_span = (target_count - 1) * median_spacing

    # If detected span is reasonable, use it
    if 0.8 < detected_span / expected_span < 1.2:
        return np.linspace(start, end, target_count).tolist()

    # Otherwise, generate from start with median spacing
    return [start + i * median_spacing for i in range(target_count)]


def find_grid_lines(
    h_positions: List[float],
    v_positions: List[float],
    target_count: int = 10
) -> Tuple[List[float], List[float]]:
    """
    Ensure we have exactly target_count lines in each direction.

    Args:
        h_positions: Clustered horizontal line Y-positions
        v_positions: Clustered vertical line X-positions
        target_count: Number of lines needed (10)

    Returns:
        Tuple of (h_final, v_final) with exactly target_count each
    """
    # Interpolate if needed
    if len(h_positions) != target_count:
        if len(h_positions) >= 2:
            h_positions = interpolate_grid_lines(h_positions, target_count)
        else:
            raise ValueError(f"Need at least 2 horizontal lines, got {len(h_positions)}")

    if len(v_positions) != target_count:
        if len(v_positions) >= 2:
            v_positions = interpolate_grid_lines(v_positions, target_count)
        else:
            raise ValueError(f"Need at least 2 vertical lines, got {len(v_positions)}")

    return h_positions, v_positions


def compute_intersections(
    h_positions: List[float],
    v_positions: List[float]
) -> np.ndarray:
    """
    Compute intersection points of horizontal and vertical lines.

    For 10 H lines and 10 V lines, we get 10×10 = 100 intersections.
    These define the corners of all 81 cells.

    Args:
        h_positions: Y-coordinates of horizontal lines
        v_positions: X-coordinates of vertical lines

    Returns:
        Array of shape (10, 10, 2) with (x, y) at each intersection
    """
    intersections = np.zeros((len(h_positions), len(v_positions), 2))

    for i, h in enumerate(h_positions):
        for j, v in enumerate(v_positions):
            # intersection point (x, y)
            intersections[i, j] = [v, h]

    return intersections


def compute_confidence(
    h_positions: List[float],
    v_positions: List[float]
) -> float:
    """
    Compute confidence score based on grid line regularity.

    A well-detected grid should have:
    - Equal spacing between lines
    - Same number of H and V lines

    Confidence = 1 - (relative std of spacings)

    Args:
        h_positions: Y-coordinates of horizontal lines
        v_positions: X-coordinates of vertical lines

    Returns:
        Confidence score in [0, 1]
    """
    if len(h_positions) < 2 or len(v_positions) < 2:
        return 0.0

    h_spacings = np.diff(h_positions)
    v_spacings = np.diff(v_positions)

    # Relative standard deviation (coefficient of variation)
    h_cv = np.std(h_spacings) / (np.mean(h_spacings) + 1e-6)
    v_cv = np.std(v_spacings) / (np.mean(v_spacings) + 1e-6)

    # Confidence: 1 means perfectly regular, 0 means irregular
    confidence = 1 - (h_cv + v_cv) / 2

    return max(0.0, min(1.0, confidence))


def draw_lines_on_image(
    image: np.ndarray,
    lines: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw line segments on an image.

    Args:
        image: BGR image to draw on
        lines: Array of lines [[x1,y1,x2,y2], ...]
        color: BGR color tuple
        thickness: Line thickness in pixels

    Returns:
        Image with lines drawn
    """
    result = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line.astype(int)
        cv2.line(result, (x1, y1), (x2, y2), color, thickness)
    return result


def draw_grid_overlay(
    image: np.ndarray,
    h_positions: List[float],
    v_positions: List[float],
    h_color: Tuple[int, int, int] = (255, 0, 0),
    v_color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw detected grid lines on image.

    Args:
        image: BGR image
        h_positions: Y-coordinates of horizontal lines
        v_positions: X-coordinates of vertical lines
        h_color: Color for horizontal lines (default: blue)
        v_color: Color for vertical lines (default: green)
        thickness: Line thickness

    Returns:
        Image with grid overlay
    """
    result = image.copy()
    h, w = image.shape[:2]

    # Draw horizontal lines
    for y in h_positions:
        y = int(y)
        cv2.line(result, (0, y), (w, y), h_color, thickness)

    # Draw vertical lines
    for x in v_positions:
        x = int(x)
        cv2.line(result, (x, 0), (x, h), v_color, thickness)

    return result


def draw_intersections(
    image: np.ndarray,
    intersections: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 255),
    radius: int = 3
) -> np.ndarray:
    """
    Draw intersection points on image.

    Args:
        image: BGR image
        intersections: 10x10x2 array of (x, y) coordinates
        color: Point color (default: yellow)
        radius: Circle radius

    Returns:
        Image with intersection points marked
    """
    result = image.copy()

    for i in range(intersections.shape[0]):
        for j in range(intersections.shape[1]):
            x, y = intersections[i, j].astype(int)
            cv2.circle(result, (x, y), radius, color, -1)

    return result


def detect_grid_standard(
    image: np.ndarray,
    return_visualization: bool = True
) -> LineDetectionResult:
    """
    Full Standard Hough Transform pipeline for grid detection.

    Pipeline:
    1. Preprocess image (grayscale, blur, threshold)
    2. Detect lines with HoughLinesP
    3. Classify lines as horizontal/vertical
    4. Cluster similar lines
    5. Interpolate to get 10 lines each direction
    6. Compute intersection points

    Args:
        image: BGR input image
        return_visualization: If True, include annotated image

    Returns:
        LineDetectionResult with all detection data
    """
    from app.core.preprocessing import preprocess_for_hough

    settings = get_settings()

    # Preprocess
    thresh = preprocess_for_hough(image)

    # Detect lines
    all_lines = detect_lines(
        thresh,
        threshold=settings.hough_threshold,
        min_line_length=settings.hough_min_line_length,
        max_line_gap=settings.hough_max_line_gap
    )

    if len(all_lines) < 10:
        # Not enough lines detected
        return LineDetectionResult(
            all_lines=all_lines,
            horizontal_lines=np.array([]),
            vertical_lines=np.array([]),
            h_positions=[],
            v_positions=[],
            intersections=np.array([]),
            confidence=0.0
        )

    # Classify
    h_lines, v_lines = classify_lines(all_lines, settings.line_angle_threshold)

    # Cluster
    h_clustered = cluster_lines(h_lines, is_horizontal=True,
                                 distance_threshold=settings.line_cluster_distance)
    v_clustered = cluster_lines(v_lines, is_horizontal=False,
                                 distance_threshold=settings.line_cluster_distance)

    # Try to get exactly 10 lines each
    try:
        h_final, v_final = find_grid_lines(h_clustered, v_clustered, target_count=10)
    except ValueError:
        return LineDetectionResult(
            all_lines=all_lines,
            horizontal_lines=h_lines,
            vertical_lines=v_lines,
            h_positions=h_clustered,
            v_positions=v_clustered,
            intersections=np.array([]),
            confidence=0.0
        )

    # Compute intersections
    intersections = compute_intersections(h_final, v_final)

    # Compute confidence
    confidence = compute_confidence(h_final, v_final)

    # Create visualization
    annotated = None
    if return_visualization:
        annotated = draw_grid_overlay(image, h_final, v_final)
        annotated = draw_intersections(annotated, intersections)

    return LineDetectionResult(
        all_lines=all_lines,
        horizontal_lines=h_lines,
        vertical_lines=v_lines,
        h_positions=h_final,
        v_positions=v_final,
        intersections=intersections,
        confidence=confidence,
        annotated_image=annotated
    )


# =============================================================================
# POLAR HOUGH TRANSFORM (cv2.HoughLines)
# =============================================================================
# Uses (rho, theta) parameterization instead of (x1, y1, x2, y2) segments.
# Better for detecting infinite lines and handling rotated grids.
# =============================================================================


@dataclass
class PolarLineDetectionResult:
    """Result of polar Hough line detection."""

    all_lines: np.ndarray          # Raw detected lines: [[rho, theta], ...]
    filtered_lines: List[Tuple[float, float]]  # After similarity filtering
    confidence: float
    annotated_image: Optional[np.ndarray] = None


def detect_lines_polar(
    edges: np.ndarray,
    threshold: int = 150
) -> np.ndarray:
    """
    Detect lines using Standard Hough Transform (polar form).

    Returns lines in polar coordinates (rho, theta):
    - rho: perpendicular distance from origin to line
    - theta: angle of the perpendicular from x-axis (in radians)

    The line equation is: rho = x*cos(theta) + y*sin(theta)

    Args:
        edges: Binary edge image (e.g., from Canny)
        threshold: Minimum votes to detect a line (higher = fewer lines)

    Returns:
        Array of lines: [[rho, theta], ...] or None if no lines found
    """
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=threshold)
    return lines


def filter_similar_lines(
    lines: np.ndarray,
    rho_threshold: float = 15,
    theta_threshold: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Filter out similar/duplicate lines based on (rho, theta) proximity.

    Algorithm:
    1. For each line, count how many other lines are "similar"
    2. Sort lines by similarity count (ascending = most unique first)
    3. Iterate through sorted list, keeping a line only if no
       already-kept line is similar to it

    Two lines are similar if:
    - |rho_i - rho_j| < rho_threshold (distance difference)
    - |theta_i - theta_j| < theta_threshold (angle difference)

    Args:
        lines: Array from cv2.HoughLines: shape (N, 1, 2)
        rho_threshold: Max rho difference to consider similar (pixels)
        theta_threshold: Max theta difference to consider similar (radians)

    Returns:
        List of filtered (rho, theta) tuples
    """
    if lines is None or len(lines) == 0:
        return []

    n = len(lines)

    # Build similarity graph: for each line, list similar lines
    similar_lines = {i: [] for i in range(n)}
    for i in range(n):
        rho_i, theta_i = lines[i][0]
        for j in range(n):
            if i == j:
                continue
            rho_j, theta_j = lines[j][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                similar_lines[i].append(j)

    # Sort indices by number of similar lines (least similar first)
    # This prioritizes unique lines over lines in clusters
    indices = sorted(range(n), key=lambda x: len(similar_lines[x]))

    # Greedy selection: keep line if no already-kept line is similar
    line_flags = [True] * n
    for i in range(n - 1):
        if not line_flags[indices[i]]:
            continue
        for j in range(i + 1, n):
            if not line_flags[indices[j]]:
                continue
            rho_i, theta_i = lines[indices[i]][0]
            rho_j, theta_j = lines[indices[j]][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                line_flags[indices[j]] = False

    return [tuple(lines[i][0]) for i in range(n) if line_flags[i]]


def polar_to_cartesian(
    rho: float,
    theta: float,
    line_length: int = 2000
) -> Tuple[int, int, int, int]:
    """
    Convert polar line (rho, theta) to Cartesian endpoints (x1, y1, x2, y2).

    The line extends `line_length` pixels in each direction from the
    closest point to origin.

    Math:
    - (x0, y0) = point on line closest to origin
    - Direction vector perpendicular to (cos(theta), sin(theta))
    - Extend in both directions

    Args:
        rho: Distance from origin to line
        theta: Angle of perpendicular from x-axis
        line_length: How far to extend line in each direction

    Returns:
        Tuple (x1, y1, x2, y2) for line endpoints
    """
    a = np.cos(theta)
    b = np.sin(theta)

    # Point on line closest to origin
    x0 = a * rho
    y0 = b * rho

    # Extend line in both directions
    # Direction perpendicular to (a, b) is (-b, a)
    x1 = int(x0 + line_length * (-b))
    y1 = int(y0 + line_length * a)
    x2 = int(x0 - line_length * (-b))
    y2 = int(y0 - line_length * a)

    return (x1, y1, x2, y2)


def draw_polar_line(
    image: np.ndarray,
    rho: float,
    theta: float,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a polar line on an image.

    Args:
        image: BGR image to draw on
        rho: Distance from origin
        theta: Angle of perpendicular
        color: BGR color tuple
        thickness: Line thickness

    Returns:
        Image with line drawn
    """
    x1, y1, x2, y2 = polar_to_cartesian(rho, theta)
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def draw_polar_lines(
    image: np.ndarray,
    lines: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw multiple polar lines on an image.

    Args:
        image: BGR image to draw on
        lines: List of (rho, theta) tuples
        color: BGR color tuple
        thickness: Line thickness

    Returns:
        Image with lines drawn
    """
    result = image.copy()
    for rho, theta in lines:
        draw_polar_line(result, rho, theta, color, thickness)
    return result


def classify_lines_by_theta(
    lines: List[Tuple[float, float]],
    angle_tolerance: float = 0.2  # ~11 degrees
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Classify polar lines as horizontal or vertical based on theta.

    In polar form:
    - Horizontal lines have theta near 90° (π/2 radians)
    - Vertical lines have theta near 0° or 180° (0 or π radians)

    Args:
        lines: List of (rho, theta) tuples
        angle_tolerance: Max deviation from ideal angle (radians)

    Returns:
        Tuple of (horizontal_lines, vertical_lines)
    """
    horizontal = []
    vertical = []

    for rho, theta in lines:
        # Horizontal: theta near π/2 (90°)
        if abs(theta - np.pi/2) < angle_tolerance:
            horizontal.append((rho, theta))
        # Vertical: theta near 0 or π
        elif theta < angle_tolerance or abs(theta - np.pi) < angle_tolerance:
            vertical.append((rho, theta))

    return horizontal, vertical


def find_dominant_angles(
    lines: np.ndarray,
    n_bins: int = 36
) -> Tuple[float, float]:
    """
    Find the two dominant line angles using histogram analysis.

    For a grid, the two dominant angles should be ~90° apart.

    Args:
        lines: Array from cv2.HoughLines: shape (N, 1, 2)
        n_bins: Number of histogram bins for angle

    Returns:
        Tuple of (angle1, angle2) in radians
    """
    if lines is None or len(lines) == 0:
        return (0.0, np.pi / 2)

    # Get all theta values
    thetas = [line[0][1] for line in lines]

    # Create histogram
    bins = np.linspace(0, np.pi, n_bins + 1)
    hist, edges = np.histogram(thetas, bins=bins)

    # Find top 2 peaks
    top_indices = np.argsort(hist)[-2:]
    peak1_theta = (edges[top_indices[0]] + edges[top_indices[0] + 1]) / 2
    peak2_theta = (edges[top_indices[1]] + edges[top_indices[1] + 1]) / 2

    return (peak1_theta, peak2_theta)


def filter_by_dominant_angles(
    lines: np.ndarray,
    angle_tolerance: float = 0.2
) -> List[Tuple[float, float]]:
    """
    Filter lines to keep only those near the two dominant angles.

    This removes diagonal lines and noise while preserving grid lines.

    Args:
        lines: Array from cv2.HoughLines: shape (N, 1, 2)
        angle_tolerance: Max deviation from dominant angle (radians, ~11°)

    Returns:
        List of (rho, theta) tuples near dominant angles
    """
    if lines is None or len(lines) == 0:
        return []

    # Find dominant angles
    peak1_theta, peak2_theta = find_dominant_angles(lines)

    # Filter to lines near these angles
    filtered = []
    for line in lines:
        rho, theta = line[0]
        if (abs(theta - peak1_theta) < angle_tolerance or
            abs(theta - peak2_theta) < angle_tolerance):
            filtered.append((rho, theta))

    return filtered


def detect_grid_polar(
    image: np.ndarray,
    return_visualization: bool = True,
    enable_filtering: bool = True,
    use_angle_filtering: bool = True
) -> PolarLineDetectionResult:
    """
    Full polar Hough Transform pipeline for grid detection.

    Pipeline:
    1. Preprocess image (Canny + dilate + erode)
    2. Detect lines with cv2.HoughLines (polar form)
    3. Filter by dominant angles (removes diagonal lines)
    4. Filter similar lines by (rho, theta) proximity
    5. Optionally visualize results

    This approach works better than HoughLinesP for:
    - Rotated grids (not axis-aligned)
    - Detecting infinite lines (not segments)
    - Images where grid lines have gaps

    Args:
        image: BGR input image
        return_visualization: If True, include annotated image
        enable_filtering: If True, filter similar lines
        use_angle_filtering: If True, filter to dominant angles first

    Returns:
        PolarLineDetectionResult with detection data
    """
    from app.core.preprocessing import preprocess_for_hough_polar

    settings = get_settings()

    # Preprocess with morphological operations
    edges = preprocess_for_hough_polar(
        image,
        canny_low=settings.canny_low,
        canny_high=settings.canny_high,
        dilate_kernel_size=settings.dilate_kernel_size,
        erode_kernel_size=settings.erode_kernel_size
    )

    # Detect lines in polar form
    all_lines = detect_lines_polar(edges, threshold=settings.hough_polar_threshold)

    if all_lines is None or len(all_lines) == 0:
        return PolarLineDetectionResult(
            all_lines=np.array([]),
            filtered_lines=[],
            confidence=0.0
        )

    # First filter by dominant angles to remove noise
    if use_angle_filtering and enable_filtering:
        angle_filtered = filter_by_dominant_angles(all_lines, angle_tolerance=0.2)
        # Convert back to numpy array for similarity filtering
        lines_for_similarity = np.array([[[r, t]] for r, t in angle_filtered])
    else:
        lines_for_similarity = all_lines

    # Filter similar lines
    if enable_filtering and len(lines_for_similarity) > 0:
        filtered_lines = filter_similar_lines(
            lines_for_similarity,
            rho_threshold=settings.rho_threshold,
            theta_threshold=settings.theta_threshold
        )
    else:
        filtered_lines = [tuple(line[0]) for line in lines_for_similarity]

    # Compute confidence based on line count
    # A good Sudoku grid has ~20 lines (10 H + 10 V)
    expected_lines = 20
    confidence = min(1.0, len(filtered_lines) / expected_lines)

    # Create visualization
    annotated = None
    if return_visualization:
        annotated = draw_polar_lines(image, filtered_lines, color=(0, 0, 255), thickness=2)

    return PolarLineDetectionResult(
        all_lines=all_lines,
        filtered_lines=filtered_lines,
        confidence=confidence,
        annotated_image=annotated
    )
