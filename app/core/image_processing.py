"""
Image processing module for Sudoku grid extraction.

This module provides multiple detection methods:
1. Standard Hough Transform - Line detection + clustering
2. Generalized Hough Transform - Shape template matching
3. Contour-based detection - Fallback method

Use detect_grid() with method parameter to select the detection approach.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pytesseract

from app.config import get_settings


@dataclass
class DetectionResult:
    """Unified result from any detection method."""

    success: bool
    corners: Optional[np.ndarray] = None
    h_positions: Optional[List[float]] = None
    v_positions: Optional[List[float]] = None
    intersections: Optional[np.ndarray] = None
    confidence: float = 0.0
    method: str = ""
    annotated_image: Optional[np.ndarray] = None


def detect_grid(
    image: np.ndarray,
    method: str = "standard",
    return_visualization: bool = False
) -> DetectionResult:
    """
    Detect Sudoku grid using the specified method.

    Args:
        image: BGR input image
        method: Detection method - "standard", "generalized", or "auto"
        return_visualization: If True, include annotated image

    Returns:
        DetectionResult with detection data
    """
    if method == "standard":
        from app.core.hough_standard import detect_grid_standard
        result = detect_grid_standard(image, return_visualization=return_visualization)

        return DetectionResult(
            success=len(result.h_positions) == 10 and len(result.v_positions) == 10,
            corners=np.array([
                [result.v_positions[0], result.h_positions[0]],
                [result.v_positions[-1], result.h_positions[0]],
                [result.v_positions[-1], result.h_positions[-1]],
                [result.v_positions[0], result.h_positions[-1]],
            ]) if result.h_positions and result.v_positions else None,
            h_positions=result.h_positions,
            v_positions=result.v_positions,
            intersections=result.intersections,
            confidence=result.confidence,
            method="standard",
            annotated_image=result.annotated_image
        )

    elif method == "generalized":
        from app.core.hough_generalized import detect_grid_ght
        result = detect_grid_ght(image, return_accumulator=return_visualization)

        return DetectionResult(
            success=result.confidence > 0.1,
            corners=result.corners if len(result.corners) > 0 else None,
            confidence=result.confidence,
            method="generalized",
            annotated_image=result.annotated_image
        )

    elif method == "auto":
        # Try standard first, fall back to GHT
        result = detect_grid(image, method="standard", return_visualization=return_visualization)
        if result.success and result.confidence > 0.5:
            return result

        # Try GHT
        result_ght = detect_grid(image, method="generalized", return_visualization=return_visualization)
        if result_ght.confidence > result.confidence:
            return result_ght

        return result

    else:
        raise ValueError(f"Unknown detection method: {method}")


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert image to binary for grid detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh


def detect_lines(
    thresh: np.ndarray,
    threshold: int = 100,
    min_line_length: int = 50,
    max_line_gap: int = 10,
) -> np.ndarray:
    """
    Detect lines using Probabilistic Hough Transform.

    Args:
        thresh: Binary threshold image
        threshold: Accumulator threshold for HoughLinesP
        min_line_length: Minimum line length to detect
        max_line_gap: Maximum gap between line segments to join

    Returns:
        Array of lines in format [[x1, y1, x2, y2], ...]
    """
    lines = cv2.HoughLinesP(
        thresh,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return np.array([])
    return lines.reshape(-1, 4)


def classify_lines(
    lines: np.ndarray, angle_threshold: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify lines into horizontal and vertical based on angle.

    Args:
        lines: Array of lines [[x1, y1, x2, y2], ...]
        angle_threshold: Maximum deviation from horizontal/vertical in degrees

    Returns:
        Tuple of (horizontal_lines, vertical_lines)
    """
    if len(lines) == 0:
        return np.array([]), np.array([])

    horizontal = []
    vertical = []

    for line in lines:
        x1, y1, x2, y2 = line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Normalize angle to -90 to 90
        angle = angle % 180
        if angle > 90:
            angle -= 180

        if abs(angle) < angle_threshold:
            horizontal.append(line)
        elif abs(abs(angle) - 90) < angle_threshold:
            vertical.append(line)

    return np.array(horizontal), np.array(vertical)


def cluster_lines(
    lines: np.ndarray,
    is_horizontal: bool,
    num_clusters: int = 10,
    distance_threshold: float = 20.0,
) -> List[float]:
    """
    Cluster parallel lines and return representative positions.

    For horizontal lines: cluster by y-coordinate
    For vertical lines: cluster by x-coordinate

    Args:
        lines: Array of parallel lines
        is_horizontal: True for horizontal lines, False for vertical
        num_clusters: Expected number of grid lines (10 for Sudoku)
        distance_threshold: Maximum distance to merge lines

    Returns:
        List of positions (y for horizontal, x for vertical)
    """
    if len(lines) == 0:
        return []

    # Extract the relevant coordinate
    if is_horizontal:
        # Average y-coordinate for each line
        positions = [(line[1] + line[3]) / 2 for line in lines]
    else:
        # Average x-coordinate for each line
        positions = [(line[0] + line[2]) / 2 for line in lines]

    positions = np.array(sorted(positions))

    # Merge nearby lines
    merged = []
    current_cluster = [positions[0]]

    for pos in positions[1:]:
        if pos - current_cluster[-1] < distance_threshold:
            current_cluster.append(pos)
        else:
            merged.append(np.mean(current_cluster))
            current_cluster = [pos]
    merged.append(np.mean(current_cluster))

    # If we don't have exactly 10 lines, try to interpolate
    if len(merged) >= 2 and len(merged) != num_clusters:
        merged = interpolate_grid_lines(merged, num_clusters)

    return merged


def interpolate_grid_lines(positions: List[float], target_count: int = 10) -> List[float]:
    """
    Interpolate or extrapolate grid lines to get exactly target_count lines.

    Assumes equally spaced grid lines.

    Args:
        positions: Detected line positions
        target_count: Number of lines to return (10 for Sudoku)

    Returns:
        List of evenly spaced line positions
    """
    if len(positions) < 2:
        raise ValueError("Need at least 2 lines to interpolate")

    # Calculate median spacing
    spacings = np.diff(positions)
    avg_spacing = np.median(spacings)

    # Use detected start and end, adjust if needed
    start = positions[0]
    end = positions[-1]

    # Expected span for target_count lines
    expected_span = (target_count - 1) * avg_spacing

    # If detected span is close to expected, use detected bounds
    actual_span = end - start
    if abs(actual_span - expected_span) < avg_spacing:
        # Use detected boundaries
        return np.linspace(start, end, target_count).tolist()
    else:
        # Generate evenly spaced lines from start
        return np.linspace(start, start + expected_span, target_count).tolist()


def find_intersections(
    h_positions: List[float], v_positions: List[float]
) -> np.ndarray:
    """
    Calculate intersection points of horizontal and vertical lines.

    Args:
        h_positions: Y-coordinates of horizontal lines
        v_positions: X-coordinates of vertical lines

    Returns:
        Array of shape (len(h), len(v), 2) with (x, y) coordinates
    """
    intersections = np.zeros((len(h_positions), len(v_positions), 2))

    for i, h in enumerate(h_positions):
        for j, v in enumerate(v_positions):
            intersections[i, j] = [v, h]  # (x, y)

    return intersections


def extract_cells_from_intersections(
    image: np.ndarray, intersections: np.ndarray, margin_ratio: float = 0.1
) -> List[np.ndarray]:
    """
    Extract 81 cell images using intersection coordinates.

    Args:
        image: The warped grid image
        intersections: 10x10 array of (x, y) coordinates
        margin_ratio: Percentage of cell to crop from edges

    Returns:
        List of 81 cell images (row-major order)
    """
    cells = []
    height, width = image.shape[:2]

    for row in range(9):
        for col in range(9):
            # Get the four corners of this cell
            top_left = intersections[row, col]
            bottom_right = intersections[row + 1, col + 1]

            x1, y1 = int(top_left[0]), int(top_left[1])
            x2, y2 = int(bottom_right[0]), int(bottom_right[1])

            # Clamp to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # Apply margin to avoid grid lines
            cell_width = x2 - x1
            cell_height = y2 - y1
            margin_x = int(cell_width * margin_ratio)
            margin_y = int(cell_height * margin_ratio)

            cell = image[y1 + margin_y : y2 - margin_y, x1 + margin_x : x2 - margin_x]
            cells.append(cell)

    return cells


def detect_grid_hough(image: np.ndarray) -> Optional[Dict]:
    """
    Detect Sudoku grid using Hough transform.

    Args:
        image: Input BGR image

    Returns:
        Dictionary with:
        - 'corners': 4 corner points for perspective transform
        - 'h_lines': 10 horizontal line positions
        - 'v_lines': 10 vertical line positions
        - 'confidence': Detection confidence score

        Or None if detection fails
    """
    settings = get_settings()

    # Preprocess
    thresh = preprocess_image(image)

    # Detect lines
    lines = detect_lines(
        thresh,
        threshold=settings.hough_threshold,
        min_line_length=settings.hough_min_line_length,
        max_line_gap=settings.hough_max_line_gap,
    )

    if len(lines) < 20:  # Need at least some lines
        return None

    # Classify lines
    h_lines, v_lines = classify_lines(lines, settings.line_angle_threshold)

    if len(h_lines) < 4 or len(v_lines) < 4:
        return None

    # Cluster to get 10 lines each
    try:
        h_positions = cluster_lines(
            h_lines,
            is_horizontal=True,
            distance_threshold=settings.line_cluster_distance,
        )
        v_positions = cluster_lines(
            v_lines,
            is_horizontal=False,
            distance_threshold=settings.line_cluster_distance,
        )
    except ValueError:
        return None

    if len(h_positions) != 10 or len(v_positions) != 10:
        return None

    # Calculate corners from extreme lines
    corners = np.array(
        [
            [v_positions[0], h_positions[0]],  # top-left
            [v_positions[-1], h_positions[0]],  # top-right
            [v_positions[-1], h_positions[-1]],  # bottom-right
            [v_positions[0], h_positions[-1]],  # bottom-left
        ],
        dtype=np.float32,
    )

    # Calculate confidence based on line regularity
    h_spacings = np.diff(h_positions)
    v_spacings = np.diff(v_positions)
    h_regularity = 1 - np.std(h_spacings) / (np.mean(h_spacings) + 1e-6)
    v_regularity = 1 - np.std(v_spacings) / (np.mean(v_spacings) + 1e-6)
    confidence = max(0.0, min(1.0, (h_regularity + v_regularity) / 2))

    return {
        "corners": corners,
        "h_lines": h_positions,
        "v_lines": v_positions,
        "confidence": confidence,
    }


# ============== Fallback: Contour-based detection ==============


def find_grid_contour(thresh: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest square contour (the Sudoku grid) - fallback method."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Sort by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Looking for a quadrilateral
        if len(approx) == 4:
            return approx

    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    pts = pts.reshape(4, 2)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def detect_grid_contour_fallback(image: np.ndarray) -> Optional[Dict]:
    """Fallback detection using contour method."""
    thresh = preprocess_image(image)
    contour = find_grid_contour(thresh)

    if contour is None:
        return None

    corners = order_points(contour)
    return {"corners": corners, "confidence": 0.5}


# ============== Perspective Transform ==============


def perspective_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Apply perspective transform to get a top-down view of the grid."""
    pts = corners if corners.shape == (4, 2) else order_points(corners)

    # Compute dimensions of the new image
    width = max(
        np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])
    )
    height = max(
        np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])
    )

    # Make it square
    size = max(int(width), int(height))

    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, matrix, (size, size))

    return warped


def extract_cells(grid_image: np.ndarray) -> List[np.ndarray]:
    """Split the grid image into 81 cells (fallback: simple division)."""
    cells = []
    height, width = grid_image.shape[:2]
    cell_h = height // 9
    cell_w = width // 9

    for i in range(9):
        for j in range(9):
            y1 = i * cell_h
            y2 = (i + 1) * cell_h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w

            cell = grid_image[y1:y2, x1:x2]
            cells.append(cell)

    return cells


# ============== OCR ==============


def preprocess_cell(cell: np.ndarray) -> np.ndarray:
    """Preprocess a cell for OCR."""
    if cell.size == 0:
        return np.zeros((50, 50), dtype=np.uint8)

    # Convert to grayscale if needed
    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # Resize for better OCR
    cell = cv2.resize(cell, (50, 50))

    # Crop center to remove grid lines
    margin = 5
    cell = cell[margin:-margin, margin:-margin]

    # Threshold
    _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Add border
    cell = cv2.copyMakeBorder(cell, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

    return cell


def recognize_digit(cell: np.ndarray) -> int:
    """Recognize a digit in a cell using OCR."""
    processed = preprocess_cell(cell)

    # Check if cell is mostly empty
    white_pixels = np.sum(processed == 255)
    total_pixels = processed.size
    if white_pixels / total_pixels < 0.03:
        return 0

    # OCR configuration for single digit
    settings = get_settings()

    try:
        text = pytesseract.image_to_string(processed, config=settings.tesseract_config).strip()
        if text and text.isdigit() and 1 <= int(text) <= 9:
            return int(text)
    except Exception:
        pass

    return 0


# ============== Main Extraction Function ==============


def extract_grid_from_image_sync(image: np.ndarray) -> Tuple[Optional[List[List[int]]], float]:
    """
    Extract Sudoku grid from image (synchronous version).

    Returns:
        Tuple of (grid, confidence) where grid is 9x9 list or None
    """
    # Try Hough transform detection first
    grid_info = detect_grid_hough(image)

    # Fallback to contour detection
    if grid_info is None:
        grid_info = detect_grid_contour_fallback(image)
        if grid_info is None:
            return None, 0.0

    confidence = grid_info.get("confidence", 0.5)
    corners = grid_info["corners"]

    # Apply perspective transform
    warped = perspective_transform(image, corners)

    # Extract cells
    if "h_lines" in grid_info and "v_lines" in grid_info:
        # Use Hough-detected grid lines for precise cell extraction
        size = warped.shape[0]
        h_warped = np.linspace(0, size, 10)
        v_warped = np.linspace(0, size, 10)
        intersections = find_intersections(h_warped.tolist(), v_warped.tolist())
        cells = extract_cells_from_intersections(warped, intersections)
    else:
        # Fallback to simple division
        cells = extract_cells(warped)

    # OCR each cell
    grid = []
    for i in range(9):
        row = []
        for j in range(9):
            digit = recognize_digit(cells[i * 9 + j])
            row.append(digit)
        grid.append(row)

    return grid, confidence


async def extract_grid_from_image(image: np.ndarray) -> Tuple[Optional[List[List[int]]], float]:
    """
    Extract Sudoku grid from image (async version).

    Runs blocking operations in a thread executor.

    Returns:
        Tuple of (grid, confidence) where grid is 9x9 list or None
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, extract_grid_from_image_sync, image)
