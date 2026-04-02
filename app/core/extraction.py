"""
Grid extraction pipeline: detect grid -> perspective transform -> OCR.

Uses hybrid contour detection (95% detection, 5.4px warp accuracy)
as the primary method. OCR is decoupled via the DigitRecognizer protocol
in app.core.ocr — swap between Tesseract (legacy) and CNN.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.ocr import CNNRecognizer, DigitRecognizer, TesseractRecognizer


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert image to binary for grid detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh


def score_quad(
    quad: np.ndarray,
    contour_area: float,
    max_contour_area: float,
    img_center: np.ndarray,
    max_dist: float,
) -> float:
    """Score a quadrilateral candidate for being a Sudoku grid.

    score = (area_norm * 0.4) + (squareness * 0.3) + (centeredness * 0.3)

    area_norm: contour_area / max_contour_area — favors larger quads without dominating
    squareness: min(w,h) / max(w,h) from bounding rect — Sudoku grids are ~square
    centeredness: 1 - (dist_from_center / max_dist) — grids tend to be near center
    """
    # Area normalized to largest candidate
    area_norm = contour_area / max_contour_area if max_contour_area > 0 else 0.0

    # Squareness from bounding rect
    x, y, w, h = cv2.boundingRect(quad.reshape(-1, 1, 2).astype(np.int32))
    squareness = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0

    # Centeredness
    centroid = np.mean(quad.reshape(-1, 2), axis=0)
    dist = np.linalg.norm(centroid - img_center)
    centeredness = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0

    return (area_norm * 0.4) + (squareness * 0.3) + (centeredness * 0.3)


def find_grid_contour(
    thresh: np.ndarray,
    epsilon: float = 0.02,
) -> Optional[np.ndarray]:
    """Find the best quadrilateral contour (the Sudoku grid).

    Scores all convex quad candidates by area, squareness, and centeredness.
    """
    h, w = thresh.shape[:2]
    image_area = h * w
    img_center = np.array([w / 2.0, h / 2.0])
    max_dist = np.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Collect all valid quad candidates
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.01 * image_area or area > 0.99 * image_area:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon * peri, True)
        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue

        candidates.append((approx, area))

    if not candidates:
        return None

    max_area = max(c[1] for c in candidates)

    best_contour = None
    best_score = -1.0
    for approx, area in candidates:
        quad = approx.reshape(4, 2).astype(np.float32)
        s = score_quad(quad, area, max_area, img_center, max_dist)
        if s > best_score:
            best_score = s
            best_contour = approx

    return best_contour


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_transform(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Apply perspective transform to get a top-down view of the grid."""
    pts = order_points(contour)
    width = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
    height = max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))
    size = max(int(width), int(height))
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, matrix, (size, size))


def extract_cells(grid_image: np.ndarray) -> List[np.ndarray]:
    """Split the grid image into 81 cells."""
    cells = []
    height, width = grid_image.shape[:2]
    cell_h = height // 9
    cell_w = width // 9
    for i in range(9):
        for j in range(9):
            cell = grid_image[
                i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
            ]
            cells.append(cell)
    return cells


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two points."""
    return a + (b - a) * t


def compute_warp_deviation(
    outer_corners: np.ndarray,
    center_corners: np.ndarray,
    size: int = 450,
) -> float:
    """Compute how far center-box corners deviate from ideal positions after 4-corner warp.

    Returns max deviation in pixels. Low = grid is regular, high = interior is warped.
    """
    outer = np.array(outer_corners, dtype=np.float32).reshape(4, 2)
    center = np.array(center_corners, dtype=np.float32).reshape(4, 2)

    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(outer, dst)

    # Transform center corners through the same homography
    center_h = np.hstack([center, np.ones((4, 1))]).T  # 3x4
    transformed = M @ center_h  # 3x4
    transformed = (transformed[:2] / transformed[2:]).T  # 4x2

    # Expected positions: (size/3, size/3), (2s/3, s/3), (2s/3, 2s/3), (s/3, 2s/3)
    s3 = size / 3
    s6 = size * 2 / 3
    expected = np.array([[s3, s3], [s6, s3], [s6, s6], [s3, s6]], dtype=np.float32)

    deviations = np.linalg.norm(transformed - expected, axis=1)
    return float(np.max(deviations))


def extract_cells_piecewise(
    image: np.ndarray,
    outer_corners: np.ndarray,
    center_corners: np.ndarray,
    size: int = 450,
) -> List[np.ndarray]:
    """Extract 81 cells using piecewise perspective transforms.

    Uses 8 annotated corners (4 outer + 4 center-box) to define 9
    box-regions, each with its own local homography. This handles
    interior grid distortion from warped/crumpled paper by ensuring
    the center box maps to a perfect square and all grid lines are
    straight within each box.

    Args:
        image: Original (unwarped) image.
        outer_corners: 4 outer corners [TL, TR, BR, BL].
        center_corners: 4 center-box corners [CTL, CTR, CBR, CBL].
        size: Output grid size in pixels.

    Returns:
        81 cell images in row-major order.
    """
    outer = np.array(outer_corners, dtype=np.float32).reshape(4, 2)
    center = np.array(center_corners, dtype=np.float32).reshape(4, 2)

    TL, TR, BR, BL = outer[0], outer[1], outer[2], outer[3]
    CTL, CTR, CBR, CBL = center[0], center[1], center[2], center[3]

    # Interpolate boundary midpoints
    T3 = _lerp(TL, TR, 1 / 3)
    T6 = _lerp(TL, TR, 2 / 3)
    B3 = _lerp(BL, BR, 1 / 3)
    B6 = _lerp(BL, BR, 2 / 3)
    L3 = _lerp(TL, BL, 1 / 3)
    L6 = _lerp(TL, BL, 2 / 3)
    R3 = _lerp(TR, BR, 1 / 3)
    R6 = _lerp(TR, BR, 2 / 3)

    s3 = size / 3
    s6 = size * 2 / 3
    sz = float(size)

    # 9 source quads [TL, TR, BR, BL] for each box
    src_quads = [
        [TL, T3, CTL, L3],
        [T3, T6, CTR, CTL],
        [T6, TR, R3, CTR],
        [L3, CTL, CBL, L6],
        [CTL, CTR, CBR, CBL],
        [CTR, R3, R6, CBR],
        [L6, CBL, B3, BL],
        [CBL, CBR, B6, B3],
        [CBR, R6, BR, B6],
    ]

    # 9 destination quads (regular grid)
    dst_quads = [
        [[0, 0], [s3, 0], [s3, s3], [0, s3]],
        [[s3, 0], [s6, 0], [s6, s3], [s3, s3]],
        [[s6, 0], [sz, 0], [sz, s3], [s6, s3]],
        [[0, s3], [s3, s3], [s3, s6], [0, s6]],
        [[s3, s3], [s6, s3], [s6, s6], [s3, s6]],
        [[s6, s3], [sz, s3], [sz, s6], [s6, s6]],
        [[0, s6], [s3, s6], [s3, sz], [0, sz]],
        [[s3, s6], [s6, s6], [s6, sz], [s3, sz]],
        [[s6, s6], [sz, s6], [sz, sz], [s6, sz]],
    ]

    # Warp each box region and composite
    output = np.zeros((size, size, 3), dtype=np.uint8)

    for src_q, dst_q in zip(src_quads, dst_quads):
        src_pts = np.array(src_q, dtype=np.float32)
        dst_pts = np.array(dst_q, dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (size, size))

        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
        mask_3ch = cv2.merge([mask, mask, mask])
        output = np.where(mask_3ch > 0, warped, output)

    # Split into 81 cells
    cell_h = size // 9
    cell_w = size // 9
    cells = []
    for i in range(9):
        for j in range(9):
            cell = output[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
            cells.append(cell)
    return cells


def _find_best_quad(
    thresh: np.ndarray,
    image_area: int,
    img_center: np.ndarray,
    max_dist: float,
) -> Optional[Tuple[np.ndarray, float, float]]:
    """Find the best quadrilateral from a binary image.

    Uses the unified scoring: 0.4*area_norm + 0.3*squareness + 0.3*centeredness.
    Returns (quad_4x2, area, score) or None.
    """
    contour = find_grid_contour(thresh)
    if contour is None:
        return None

    quad = contour.reshape(4, 2).astype(np.float32)
    area = cv2.contourArea(contour)
    # Score with self as max (this path is called per-threshold, so max_area=area)
    s = score_quad(quad, area, area, img_center, max_dist)
    return (quad, area, s)


def detect_grid(image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Hybrid grid detection combining best of contour + sudoku_detector approaches.

    Pipeline:
    1. CLAHE + adaptive threshold (handles variable lighting)
    2. Find contours with epsilon=0.02 (preserves quad shapes)
    3. Filter by area (1-95% of image), approximate to 4 vertices
    4. Light validation: convexity check only
    5. Score: primary=area, secondary=centeredness as tiebreaker
    6. Sub-pixel corner refinement via cv2.cornerSubPix

    Falls back to simple preprocessing (no CLAHE) if CLAHE pipeline finds no quads.

    Args:
        image: Input BGR image.

    Returns:
        Tuple of (corners_4x2, confidence). corners is None if detection fails.
        Corners are ordered [TL, TR, BR, BL].
    """
    h, w = image.shape[:2]
    image_area = h * w
    img_center = np.array([w / 2, h / 2])
    max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Primary path: CLAHE + adaptive threshold
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    result = _find_best_quad(thresh, image_area, img_center, max_dist)

    # Fallback: simple preprocessing without CLAHE
    if result is None:
        blurred_simple = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh_simple = cv2.adaptiveThreshold(
            blurred_simple,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
        result = _find_best_quad(thresh_simple, image_area, img_center, max_dist)

    if result is None:
        return None, 0.0

    quad, area, _ = result

    # Order corners: TL, TR, BR, BL
    corners = order_points(quad.reshape(4, 1, 2))

    # Sub-pixel refinement
    corners_for_subpix = corners.reshape(-1, 1, 2).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(
        gray, corners_for_subpix, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria
    )
    corners = refined.reshape(4, 2).astype(np.float32)

    # Confidence based on area ratio
    area_ratio = area / image_area
    confidence = min(1.0, area_ratio / 0.5) if area_ratio > 0.05 else 0.0

    return corners, confidence


# Default recognizer instance (used when caller doesn't specify one)
_default_recognizer: Optional[DigitRecognizer] = None


def get_recognizer() -> DigitRecognizer:
    """Get the default digit recognizer (lazy-initialized).

    Uses CNN if checkpoint exists, falls back to Tesseract.
    """
    global _default_recognizer
    if _default_recognizer is None:
        try:
            _default_recognizer = CNNRecognizer()
        except Exception:
            _default_recognizer = TesseractRecognizer()
    return _default_recognizer


def set_recognizer(recognizer: DigitRecognizer) -> None:
    """Set the default digit recognizer (e.g. swap in CNN)."""
    global _default_recognizer
    _default_recognizer = recognizer


def recognize_cells(
    cells: List[np.ndarray],
    recognizer: Optional[DigitRecognizer] = None,
) -> Tuple[List[List[int]], List[List[float]]]:
    """Recognize digits in 81 cell images.

    Args:
        cells: 81 cell images in row-major order.
        recognizer: DigitRecognizer to use. Falls back to default if None.

    Returns:
        (grid, confidence_map) — both 9x9 lists.
    """
    rec = recognizer or get_recognizer()
    results = rec.predict_batch(cells)

    grid = []
    confidence_map = []
    for i in range(9):
        row = []
        conf_row = []
        for j in range(9):
            digit, conf = results[i * 9 + j]
            row.append(digit)
            conf_row.append(conf)
        grid.append(row)
        confidence_map.append(conf_row)
    return grid, confidence_map


def extract_grid_from_image(
    image: np.ndarray,
    recognizer: Optional[DigitRecognizer] = None,
) -> Optional[List[List[int]]]:
    """Extract Sudoku grid from image: detect -> warp -> OCR.

    Args:
        image: Input BGR image.
        recognizer: DigitRecognizer to use. Falls back to default if None.

    Returns:
        9x9 grid of ints (0 = empty), or None if detection fails.
    """
    thresh = preprocess_image(image)
    contour = find_grid_contour(thresh)

    if contour is None:
        return None

    warped = perspective_transform(image, contour)
    cells = extract_cells(warped)

    grid, _ = recognize_cells(cells, recognizer)
    return grid


# Backward-compatible re-exports for evaluation/benchmark.py
from app.core.ocr import preprocess_cell as preprocess_cell  # noqa: E402, F401
from app.core.ocr import TesseractRecognizer as _TR  # noqa: E402

_compat_recognizer = _TR()


def recognize_digit(cell: np.ndarray) -> int:
    """Legacy wrapper — delegates to the default recognizer."""
    digit, _ = _compat_recognizer.predict(cell)
    return digit
