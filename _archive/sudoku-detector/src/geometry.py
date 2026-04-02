"""Geometric utility functions for sudoku grid detection."""

from typing import List, Tuple

import cv2
import numpy as np

try:
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order 4 corner points as [top-left, top-right, bottom-right, bottom-left].

    Uses sum and difference method:
    - Top-left: smallest (x + y)
    - Bottom-right: largest (x + y)
    - Top-right: largest (x - y)
    - Bottom-left: smallest (x - y)

    Args:
        corners: Unordered corner points with shape (4, 2).

    Returns:
        Ordered corners array with shape (4, 2) and dtype float32.
    """
    corners = np.array(corners, dtype=np.float32)

    if corners.shape != (4, 2):
        raise ValueError(f"Expected shape (4, 2), got {corners.shape}")

    # Compute sum and difference for each point
    sums = corners[:, 0] + corners[:, 1]
    diffs = corners[:, 0] - corners[:, 1]

    # Find indices of extreme points
    top_left_idx = np.argmin(sums)
    bottom_right_idx = np.argmax(sums)
    top_right_idx = np.argmax(diffs)
    bottom_left_idx = np.argmin(diffs)

    # Order corners: TL, TR, BR, BL
    ordered = np.array([
        corners[top_left_idx],
        corners[top_right_idx],
        corners[bottom_right_idx],
        corners[bottom_left_idx],
    ], dtype=np.float32)

    return ordered


def refine_corners(
    gray: np.ndarray,
    corners: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Refine corner positions to sub-pixel accuracy.

    Uses cv2.cornerSubPix to find precise corner locations.

    Args:
        gray: Grayscale image.
        corners: Initial corner positions with shape (4, 2).
        window_size: Half of the search window side length.

    Returns:
        Refined corners array with shape (4, 2) and dtype float32.
    """
    corners = np.array(corners, dtype=np.float32)

    # cornerSubPix expects shape (N, 1, 2)
    corners_reshaped = corners.reshape(-1, 1, 2)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    refined = cv2.cornerSubPix(
        gray,
        corners_reshaped,
        winSize=(window_size, window_size),
        zeroZone=(-1, -1),
        criteria=criteria,
    )

    return refined.reshape(4, 2).astype(np.float32)


def compute_homography(
    src_corners: np.ndarray,
    output_size: int,
) -> np.ndarray:
    """Compute homography matrix for perspective transformation.

    Args:
        src_corners: 4 source corners ordered [TL, TR, BR, BL] with shape (4, 2).
        output_size: Size of the output square image.

    Returns:
        3x3 homography matrix.
    """
    src_corners = np.array(src_corners, dtype=np.float32)

    # Destination corners for square output
    dst_corners = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_corners, dst_corners)
    return H


def warp_perspective(
    image: np.ndarray,
    H: np.ndarray,
    output_size: int,
) -> np.ndarray:
    """Apply perspective warp to extract rectified sudoku grid.

    Args:
        image: Input image (color or grayscale).
        H: 3x3 homography matrix.
        output_size: Size of the output square image.

    Returns:
        Warped image with shape (output_size, output_size) or
        (output_size, output_size, 3).
    """
    warped = cv2.warpPerspective(
        image,
        H,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def scale_corners_to_original(
    corners: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """Map corners from resized image coordinates to original image coordinates.

    Args:
        corners: Corner positions in resized image coordinates.
        scale_factor: Scale factor used to resize the image (resized/original).

    Returns:
        Corners in original image coordinates.
    """
    corners = np.array(corners, dtype=np.float32)

    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive")

    # corners_original = corners / scale_factor
    return corners / scale_factor


def compute_quadrilateral_area(corners: np.ndarray) -> float:
    """Compute area of quadrilateral using the Shoelace formula.

    Args:
        corners: Ordered corner points with shape (4, 2).

    Returns:
        Absolute area of the quadrilateral.
    """
    corners = np.array(corners, dtype=np.float32)

    if corners.shape != (4, 2):
        raise ValueError(f"Expected shape (4, 2), got {corners.shape}")

    # Shoelace formula: 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    n = len(corners)
    area = 0.0

    for i in range(n):
        j = (i + 1) % n
        area += corners[i, 0] * corners[j, 1]
        area -= corners[j, 0] * corners[i, 1]

    return abs(area) / 2.0


def _compute_angle_at_vertex(
    p_prev: np.ndarray,
    vertex: np.ndarray,
    p_next: np.ndarray,
) -> float:
    """Compute interior angle at a vertex.

    Args:
        p_prev: Previous point.
        vertex: The vertex point.
        p_next: Next point.

    Returns:
        Angle in degrees.
    """
    v1 = p_prev - vertex
    v2 = p_next - vertex

    dot = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)

    # Handle degenerate case (zero-length edge)
    if mag1 < 1e-10 or mag2 < 1e-10:
        return 0.0

    # Clamp to avoid numerical errors with arccos
    cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)

    return np.degrees(angle_rad)


def compute_interior_angles(corners: np.ndarray) -> List[float]:
    """Compute interior angles at each of the 4 corners.

    Args:
        corners: Ordered corner points with shape (4, 2).

    Returns:
        List of 4 angles in degrees, one for each corner.
    """
    corners = np.array(corners, dtype=np.float64)

    if corners.shape != (4, 2):
        raise ValueError(f"Expected shape (4, 2), got {corners.shape}")

    angles = []
    n = len(corners)

    for i in range(n):
        p_prev = corners[(i - 1) % n]
        vertex = corners[i]
        p_next = corners[(i + 1) % n]

        angle = _compute_angle_at_vertex(p_prev, vertex, p_next)
        angles.append(angle)

    return angles


def is_valid_quadrilateral(
    corners: np.ndarray,
    min_angle: float = 45.0,
    max_angle: float = 135.0,
) -> bool:
    """Check if quadrilateral meets validity criteria.

    Criteria:
    - Must be convex
    - All interior angles must be between min_angle and max_angle

    Args:
        corners: Ordered corner points with shape (4, 2).
        min_angle: Minimum acceptable interior angle in degrees.
        max_angle: Maximum acceptable interior angle in degrees.

    Returns:
        True if quadrilateral is valid, False otherwise.
    """
    corners = np.array(corners, dtype=np.float32)

    if corners.shape != (4, 2):
        return False

    # Check convexity
    corners_for_cv = corners.reshape(4, 1, 2)
    if not cv2.isContourConvex(corners_for_cv):
        return False

    # Check angles
    angles = compute_interior_angles(corners)

    for angle in angles:
        if angle < min_angle or angle > max_angle:
            return False

    return True


def compute_aspect_ratio(corners: np.ndarray) -> float:
    """Compute approximate aspect ratio of quadrilateral.

    Aspect ratio = average width / average height

    Width = (dist(TL, TR) + dist(BL, BR)) / 2
    Height = (dist(TL, BL) + dist(TR, BR)) / 2

    Args:
        corners: Ordered corners [TL, TR, BR, BL] with shape (4, 2).

    Returns:
        Aspect ratio (width / height). Returns 0.0 if height is zero.
    """
    corners = np.array(corners, dtype=np.float32)

    if corners.shape != (4, 2):
        raise ValueError(f"Expected shape (4, 2), got {corners.shape}")

    tl, tr, br, bl = corners

    # Compute edge lengths
    top_width = np.linalg.norm(tr - tl)
    bottom_width = np.linalg.norm(br - bl)
    left_height = np.linalg.norm(bl - tl)
    right_height = np.linalg.norm(br - tr)

    # Average width and height
    avg_width = (top_width + bottom_width) / 2
    avg_height = (left_height + right_height) / 2

    # Handle degenerate case
    if avg_height < 1e-10:
        return 0.0

    return avg_width / avg_height


def compute_iou(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """Compute Intersection over Union of two quadrilaterals.

    Uses shapely library for polygon intersection/union calculation.
    Falls back to a bounding-box based approximation if shapely is unavailable.

    Args:
        corners1: (4, 2) array of first quad corners [TL, TR, BR, BL].
        corners2: (4, 2) array of second quad corners.

    Returns:
        IoU value between 0.0 and 1.0.
    """
    corners1 = np.array(corners1, dtype=np.float64)
    corners2 = np.array(corners2, dtype=np.float64)

    if corners1.shape != (4, 2) or corners2.shape != (4, 2):
        raise ValueError(
            f"Expected shapes (4, 2), got {corners1.shape} and {corners2.shape}"
        )

    if SHAPELY_AVAILABLE:
        # Use shapely for accurate polygon IoU
        poly1 = Polygon(corners1)
        poly2 = Polygon(corners2)

        if not poly1.is_valid or not poly2.is_valid:
            return 0.0

        try:
            intersection_area = poly1.intersection(poly2).area
            union_area = poly1.union(poly2).area

            if union_area <= 0:
                return 0.0

            return intersection_area / union_area
        except Exception:
            return 0.0
    else:
        # Fallback: use OpenCV contour intersection
        # This is less accurate but works without shapely
        corners1_int = corners1.astype(np.int32)
        corners2_int = corners2.astype(np.int32)

        # Create masks for both polygons
        # Find bounding box to minimize mask size
        all_corners = np.vstack([corners1_int, corners2_int])
        min_x, min_y = all_corners.min(axis=0) - 1
        max_x, max_y = all_corners.max(axis=0) + 1
        min_x, min_y = max(0, min_x), max(0, min_y)

        h, w = max_y - min_y + 1, max_x - min_x + 1

        # Offset corners
        c1_offset = corners1_int - [min_x, min_y]
        c2_offset = corners2_int - [min_x, min_y]

        # Create masks
        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask2 = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask1, [c1_offset.reshape(-1, 1, 2)], 255)
        cv2.fillPoly(mask2, [c2_offset.reshape(-1, 1, 2)], 255)

        # Compute intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return float(intersection) / float(union)


def corners_from_bbox(
    x: float, y: float, width: float, height: float
) -> np.ndarray:
    """Create corner array from bounding box parameters.

    Useful for creating test quadrilaterals.

    Args:
        x: Top-left x coordinate.
        y: Top-left y coordinate.
        width: Box width.
        height: Box height.

    Returns:
        (4, 2) array of corners [TL, TR, BR, BL].
    """
    return np.array([
        [x, y],                      # TL
        [x + width, y],              # TR
        [x + width, y + height],     # BR
        [x, y + height],             # BL
    ], dtype=np.float32)
