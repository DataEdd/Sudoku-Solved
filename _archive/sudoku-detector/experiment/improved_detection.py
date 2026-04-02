#!/usr/bin/env python3
"""
Improved corner detection that extracts OUTER boundary lines.

Key insight: After line detection, we need the OUTERMOST lines (min/max intercepts)
not just clustered representative lines.
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path.cwd().parent))

from src.preprocessing import resize_image, to_grayscale
from src.hough_detection import classify_lines
from src.geometry import order_corners, refine_corners, compute_iou

# Configuration
RANDOM_SEED = 42
TARGET_SIZE = 1000
GROUND_TRUTH_FILE = Path('../data/ground_truth_corners.json')
EXAMPLES_DIR = Path('../examples')


def load_ground_truth() -> Dict:
    with open(GROUND_TRUTH_FILE, 'r') as f:
        data = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in data.items() if '.' in k}


def get_sample_images() -> List[Path]:
    random.seed(RANDOM_SEED)
    all_images = list(EXAMPLES_DIR.glob('*.jpeg'))
    return random.sample(all_images, 6)


def line_intercept(line: Tuple[int, int, int, int], is_horizontal: bool) -> float:
    """Get the intercept of a line (y for horizontal, x for vertical)."""
    x1, y1, x2, y2 = line
    if is_horizontal:
        return (y1 + y2) / 2
    else:
        return (x1 + x2) / 2


def line_to_coefficients(line: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
    """Convert line segment to coefficients (a, b, c) where ax + by + c = 0."""
    x1, y1, x2, y2 = line
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, c


def line_intersection(line1: Tuple, line2: Tuple) -> Optional[Tuple[float, float]]:
    """Find intersection point of two lines."""
    a1, b1, c1 = line_to_coefficients(line1)
    a2, b2, c2 = line_to_coefficients(line2)

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None

    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return x, y


def extend_line(line: Tuple[int, int, int, int], img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Extend a line segment to span the image."""
    x1, y1, x2, y2 = line
    h, w = img_shape

    if x2 == x1:  # Vertical line
        return (x1, 0, x1, h)

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    # Find intersections with image borders
    points = []

    # Left border (x=0)
    y_at_0 = intercept
    if 0 <= y_at_0 <= h:
        points.append((0, int(y_at_0)))

    # Right border (x=w)
    y_at_w = slope * w + intercept
    if 0 <= y_at_w <= h:
        points.append((w, int(y_at_w)))

    # Top border (y=0)
    if abs(slope) > 1e-10:
        x_at_0 = -intercept / slope
        if 0 <= x_at_0 <= w:
            points.append((int(x_at_0), 0))

    # Bottom border (y=h)
    if abs(slope) > 1e-10:
        x_at_h = (h - intercept) / slope
        if 0 <= x_at_h <= w:
            points.append((int(x_at_h), h))

    if len(points) >= 2:
        return (points[0][0], points[0][1], points[1][0], points[1][1])
    return line


def filter_lines_by_angle(lines: List, is_horizontal: bool, angle_threshold: float = 30) -> List:
    """Filter lines that are clearly horizontal or vertical."""
    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

        if is_horizontal and angle < angle_threshold:
            filtered.append(line)
        elif not is_horizontal and angle > (90 - angle_threshold):
            filtered.append(line)

    return filtered


def extract_boundary_lines(
    h_lines: List[Tuple],
    v_lines: List[Tuple],
    img_shape: Tuple[int, int],
    margin_ratio: float = 0.1
) -> Tuple[Optional[Tuple], Optional[Tuple], Optional[Tuple], Optional[Tuple]]:
    """
    Extract the 4 boundary lines (top, bottom, left, right).

    Strategy:
    1. For horizontal lines, find those near the top (min y) and bottom (max y)
    2. For vertical lines, find those near the left (min x) and right (max x)
    3. Use median line from candidates in boundary region
    """
    h, w = img_shape
    margin_h = h * margin_ratio
    margin_w = w * margin_ratio

    # Sort horizontal lines by y-intercept
    h_sorted = sorted(h_lines, key=lambda l: line_intercept(l, True))

    # Sort vertical lines by x-intercept
    v_sorted = sorted(v_lines, key=lambda l: line_intercept(l, False))

    # Find top boundary (smallest y intercepts)
    top_candidates = [l for l in h_sorted if line_intercept(l, True) < h * 0.35]

    # Find bottom boundary (largest y intercepts)
    bottom_candidates = [l for l in h_sorted if line_intercept(l, True) > h * 0.65]

    # Find left boundary (smallest x intercepts)
    left_candidates = [l for l in v_sorted if line_intercept(l, False) < w * 0.35]

    # Find right boundary (largest x intercepts)
    right_candidates = [l for l in v_sorted if line_intercept(l, False) > w * 0.65]

    def get_extreme_line(candidates: List, is_min: bool, is_horizontal: bool) -> Optional[Tuple]:
        if not candidates:
            return None

        # Get the most extreme line
        if is_min:
            return min(candidates, key=lambda l: line_intercept(l, is_horizontal))
        else:
            return max(candidates, key=lambda l: line_intercept(l, is_horizontal))

    top_line = get_extreme_line(top_candidates, is_min=True, is_horizontal=True)
    bottom_line = get_extreme_line(bottom_candidates, is_min=False, is_horizontal=True)
    left_line = get_extreme_line(left_candidates, is_min=True, is_horizontal=False)
    right_line = get_extreme_line(right_candidates, is_min=False, is_horizontal=False)

    return top_line, bottom_line, left_line, right_line


def compute_corners_from_boundaries(
    top_line: Tuple,
    bottom_line: Tuple,
    left_line: Tuple,
    right_line: Tuple,
    img_shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    """Compute 4 corners from boundary line intersections."""

    # Extend lines to ensure they intersect
    h, w = img_shape
    top_ext = extend_line(top_line, img_shape)
    bottom_ext = extend_line(bottom_line, img_shape)
    left_ext = extend_line(left_line, img_shape)
    right_ext = extend_line(right_line, img_shape)

    # Compute intersections
    tl = line_intersection(top_ext, left_ext)
    tr = line_intersection(top_ext, right_ext)
    br = line_intersection(bottom_ext, right_ext)
    bl = line_intersection(bottom_ext, left_ext)

    if None in [tl, tr, br, bl]:
        return None

    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    # Validate corners are within reasonable bounds
    margin = 50
    for x, y in corners:
        if x < -margin or x > w + margin or y < -margin or y > h + margin:
            return None

    return corners


def improved_detect_corners(
    image: np.ndarray,
    params: dict
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Improved corner detection using boundary line extraction.

    Returns:
        corners: (4, 2) array of corners in original image coordinates, or None
        debug: dict with intermediate results for debugging
    """
    debug = {}

    # Preprocess
    resized, scale = resize_image(image, TARGET_SIZE)
    gray = to_grayscale(resized)
    debug['scale'] = scale
    debug['resized'] = resized

    # Bilateral filter
    filtered = cv2.bilateralFilter(
        gray,
        d=params.get('bilateral_d', 9),
        sigmaColor=params.get('bilateral_sigma_color', 75),
        sigmaSpace=params.get('bilateral_sigma_space', 75)
    )

    # Canny edges
    edges = cv2.Canny(
        filtered,
        params.get('canny_low', 50),
        params.get('canny_high', 150)
    )
    debug['edges'] = edges

    # Hough lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=params.get('hough_threshold', 50),
        minLineLength=params.get('hough_min_length', 50),
        maxLineGap=params.get('hough_max_gap', 10)
    )

    if lines is None or len(lines) == 0:
        debug['error'] = 'No lines detected'
        return None, debug

    debug['num_lines'] = len(lines)

    # Classify lines
    h_lines, v_lines = classify_lines(lines)
    debug['h_lines'] = h_lines
    debug['v_lines'] = v_lines

    if len(h_lines) < 2 or len(v_lines) < 2:
        debug['error'] = f'Not enough lines: H={len(h_lines)}, V={len(v_lines)}'
        return None, debug

    # Extract boundary lines
    top, bottom, left, right = extract_boundary_lines(
        h_lines, v_lines, gray.shape[:2]
    )
    debug['boundary_lines'] = {'top': top, 'bottom': bottom, 'left': left, 'right': right}

    if None in [top, bottom, left, right]:
        debug['error'] = f'Missing boundary lines: T={top is not None}, B={bottom is not None}, L={left is not None}, R={right is not None}'
        return None, debug

    # Compute corners
    corners = compute_corners_from_boundaries(top, bottom, left, right, gray.shape[:2])

    if corners is None:
        debug['error'] = 'Corner computation failed'
        return None, debug

    # Refine corners
    try:
        corners = refine_corners(gray, corners)
    except:
        pass  # Use unrefined if refinement fails

    # Scale back to original image
    corners_orig = corners / scale

    debug['detected_corners'] = corners_orig
    return corners_orig, debug


def test_on_all_images():
    """Test improved detection on all sample images."""
    gt = load_ground_truth()
    images = get_sample_images()

    print("Testing Improved Detection Algorithm")
    print("=" * 60)

    # Parameter combinations to try
    param_sets = [
        {'name': 'default', 'hough_threshold': 50, 'canny_low': 50, 'canny_high': 150},
        {'name': 'sensitive', 'hough_threshold': 30, 'canny_low': 30, 'canny_high': 100},
        {'name': 'strict', 'hough_threshold': 70, 'canny_low': 50, 'canny_high': 200},
        {'name': 'very_sensitive', 'hough_threshold': 20, 'canny_low': 20, 'canny_high': 80},
    ]

    results = {}

    for img_path in images:
        name = img_path.name
        if name not in gt:
            continue

        image = cv2.imread(str(img_path))
        gt_corners = gt[name]

        print(f"\n{name}:")

        best_iou = 0
        best_params = None
        best_corners = None

        for params in param_sets:
            corners, debug = improved_detect_corners(image, params)

            if corners is not None:
                iou = compute_iou(corners, gt_corners)
                print(f"  {params['name']}: IoU = {iou:.4f}")

                if iou > best_iou:
                    best_iou = iou
                    best_params = params
                    best_corners = corners
            else:
                print(f"  {params['name']}: FAILED - {debug.get('error', 'unknown')}")

        results[name] = {
            'best_iou': best_iou,
            'best_params': best_params,
            'corners': best_corners,
            'status': 'PASS' if best_iou >= 0.95 else 'FAIL'
        }

        print(f"  -> Best: {best_iou:.4f} ({best_params['name'] if best_params else 'N/A'}) [{results[name]['status']}]")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r['status'] == 'PASS')
    print(f"\nPassed: {passed}/{len(results)}")

    for name, r in results.items():
        print(f"  {name}: {r['best_iou']:.4f} [{r['status']}]")

    return results


if __name__ == '__main__':
    test_on_all_images()
