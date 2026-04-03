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


# ---------------------------------------------------------------------------
# Structure-aware scoring for detect_grid_v2
# ---------------------------------------------------------------------------

_WARP_SIZE = 200


def score_grid_structure(
    image: np.ndarray,
    quad: np.ndarray,
    size: int = _WARP_SIZE,
) -> Tuple[float, int, int, float, float]:
    """Score how well a quad's interior matches a 9x9 grid line pattern.

    Warps the quad to a fixed-size square, extracts horizontal and vertical
    lines via morphological filtering, then checks line count, spacing
    regularity, and coverage.

    All pixel measurements are relative to the warped image (size x size),
    so they are independent of the original image resolution.

    Returns:
        (score, n_h_lines, n_v_lines, h_coverage, v_coverage)
    """
    src = order_points(quad.reshape(4, 1, 2))
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3
        else image
    )
    warped = cv2.warpPerspective(gray, M, (size, size))

    thresh = cv2.adaptiveThreshold(
        warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 2,
    )

    # Extract lines using morphological opening (keeps long structures only)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size // 4, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size // 4))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

    # Project to 1D profiles
    h_proj = np.sum(h_lines, axis=1).astype(float) / (size * 255)
    v_proj = np.sum(v_lines, axis=0).astype(float) / (size * 255)

    expected_gap = size / 9  # ~22px for size=200

    def _analyze(prof: np.ndarray) -> Tuple[float, int, float]:
        if np.max(prof) == 0:
            return 0.0, 0, 0.0
        peak_thresh = np.max(prof) * 0.3
        min_dist = size // 15
        peaks = []
        for i in range(1, len(prof) - 1):
            if (
                prof[i] > peak_thresh
                and prof[i] >= prof[i - 1]
                and prof[i] >= prof[i + 1]
            ):
                if not peaks or (i - peaks[-1]) >= min_dist:
                    peaks.append(i)
        n = len(peaks)
        if n < 2:
            return 0.0, n, 0.0
        count_acc = max(0.0, 1.0 - abs(n - 10) / 10)
        gaps = np.diff(peaks)
        regularity = max(0.0, 1.0 - float(np.std(gaps)) / expected_gap)
        coverage = (peaks[-1] - peaks[0]) / size
        return count_acc * regularity * coverage, n, coverage

    h_score, n_h, h_cov = _analyze(h_proj)
    v_score, n_v, v_cov = _analyze(v_proj)
    return (h_score + v_score) / 2, n_h, n_v, h_cov, v_cov


def score_cell_count(
    image: np.ndarray,
    quad: np.ndarray,
    size: int = _WARP_SIZE,
) -> Tuple[float, int, float]:
    """Score whether a quad's interior contains ~81 cell-sized regions.

    Warps the quad, inverts the threshold (white cells on dark lines),
    counts connected components of the expected cell size.

    Returns:
        (score, cell_count, median_cell_area)
    """
    src = order_points(quad.reshape(4, 1, 2))
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3
        else image
    )
    warped = cv2.warpPerspective(gray, M, (size, size))

    thresh = cv2.adaptiveThreshold(
        warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 2,
    )
    cells_img = cv2.bitwise_not(thresh)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(cells_img)

    expected_area = (size / 9) ** 2  # ~494px² for size=200
    min_a = expected_area * 0.2
    max_a = expected_area * 3.0

    cell_areas = []
    for i in range(1, n_labels):  # skip background
        a = stats[i, cv2.CC_STAT_AREA]
        if min_a < a < max_a:
            cell_areas.append(float(a))

    count = len(cell_areas)
    median_area = float(np.median(cell_areas)) if cell_areas else 0.0

    closeness = max(0.0, 1.0 - abs(count - 81) / 81)
    if len(cell_areas) > 5:
        cv_val = float(np.std(cell_areas) / np.mean(cell_areas))
        consistency = max(0.0, 1.0 - cv_val)
    else:
        consistency = 0.0

    return closeness * consistency, count, median_area


def _find_best_quad_structured(
    image: np.ndarray,
    thresh: np.ndarray,
    image_area: int,
    img_center: np.ndarray,
    max_dist: float,
) -> Optional[Tuple[np.ndarray, float, float]]:
    """Find the best quad using RETR_TREE + structure-aware scoring.

    Scoring: area*0.2 + squareness*0.2 + centeredness*0.1
             + grid_structure*0.3 + cell_count*0.2
    """
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.01 * image_area or area > 0.99 * image_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
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
        area_norm = area / max_area if max_area > 0 else 0.0
        x, y, w, h = cv2.boundingRect(approx)
        squareness = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
        centroid = np.mean(quad, axis=0)
        dist = np.linalg.norm(centroid - img_center)
        centeredness = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0

        struct_score = score_grid_structure(image, quad)[0]
        cell_score = score_cell_count(image, quad)[0]

        s = (
            area_norm * 0.2
            + squareness * 0.2
            + centeredness * 0.1
            + struct_score * 0.3
            + cell_score * 0.2
        )
        if s > best_score:
            best_score = s
            best_contour = approx

    if best_contour is None:
        return None

    quad = best_contour.reshape(4, 2).astype(np.float32)
    area = cv2.contourArea(best_contour)
    return (quad, area, best_score)


def _preprocess(
    gray: np.ndarray,
    clahe_clip: float = 2.0,
    block_size: int = 11,
    thresh_c: int = 2,
    morph_dilate: int = 0,
    morph_erode: int = 0,
) -> np.ndarray:
    """Shared preprocessing: CLAHE -> blur -> threshold -> optional morph."""
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, thresh_c,
    )
    if morph_dilate > 0:
        thresh = cv2.dilate(
            thresh, np.ones((morph_dilate, morph_dilate), np.uint8), iterations=1,
        )
    if morph_erode > 0:
        thresh = cv2.erode(
            thresh, np.ones((morph_erode, morph_erode), np.uint8), iterations=1,
        )
    return thresh


def _find_quad_standard(
    thresh: np.ndarray,
    image_area: int,
    img_center: np.ndarray,
    max_dist: float,
) -> Optional[Tuple[np.ndarray, float, float]]:
    """Find best quad with RETR_EXTERNAL + standard scoring (0.4/0.3/0.3)."""
    contour = find_grid_contour(thresh)
    if contour is None:
        return None
    quad = contour.reshape(4, 2).astype(np.float32)
    area = cv2.contourArea(contour)
    s = score_quad(quad, area, area, img_center, max_dist)
    return (quad, area, s)


def _refine_corners(
    gray: np.ndarray,
    quad: np.ndarray,
) -> np.ndarray:
    """Order corners and apply sub-pixel refinement."""
    corners = order_points(quad.reshape(4, 1, 2))
    h, w = gray.shape[:2]
    corners_sp = corners.reshape(-1, 1, 2).astype(np.float32)
    corners_sp[:, :, 0] = np.clip(corners_sp[:, :, 0], 5, w - 6)
    corners_sp[:, :, 1] = np.clip(corners_sp[:, :, 1], 5, h - 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(
        gray, corners_sp, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria,
    )
    return refined.reshape(4, 2).astype(np.float32)


def detect_grid_v2(image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Detect Sudoku grid using a 4-step deterministic fallback chain.

    Step 1: RETR_TREE + structure-aware scoring (29/38 on GT benchmark).
            Uses grid_structure and cell_count scoring to prefer quads
            whose interiors look like a 9x9 grid.
    Step 2: Morph dilate=3/erode=5 fallback — closes broken contours.
    Step 3: Aggressive CLAHE (clip=6, C=7) — enhances faint grids.
    Step 4: Morph dilate=3/erode=3 fallback — closes fragmented lines.

    Each step is tried in order; the first detection is accepted.

    Args:
        image: Input BGR image.

    Returns:
        Tuple of (corners_4x2, confidence). corners is None if all steps fail.
        Corners are ordered [TL, TR, BR, BL].
    """
    h, w = image.shape[:2]
    image_area = h * w
    img_center = np.array([w / 2.0, h / 2.0])
    max_dist = np.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: TREE + structure scoring (primary)
    thresh1 = _preprocess(gray)
    result = _find_best_quad_structured(
        image, thresh1, image_area, img_center, max_dist,
    )

    # Step 2: Morph dilate=3, erode=5
    if result is None:
        thresh2 = _preprocess(gray, morph_dilate=3, morph_erode=5)
        result = _find_quad_standard(thresh2, image_area, img_center, max_dist)

    # Step 3: Aggressive CLAHE (clip=6, C=7)
    if result is None:
        thresh3 = _preprocess(gray, clahe_clip=6.0, thresh_c=7)
        result = _find_quad_standard(thresh3, image_area, img_center, max_dist)

    # Step 4: Morph dilate=3, erode=3
    if result is None:
        thresh4 = _preprocess(gray, morph_dilate=3, morph_erode=3)
        result = _find_quad_standard(thresh4, image_area, img_center, max_dist)

    if result is None:
        return None, 0.0

    quad, area, _ = result
    corners = _refine_corners(gray, quad)

    area_ratio = area / image_area
    confidence = min(1.0, area_ratio / 0.5) if area_ratio > 0.05 else 0.0

    return corners, confidence


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
