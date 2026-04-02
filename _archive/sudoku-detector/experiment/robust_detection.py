#!/usr/bin/env python3
"""
Robust sudoku corner detection combining multiple strategies.

Strategy:
1. Try contour-based detection with multiple preprocessing methods
2. Score each result by geometric properties (squareness, size, etc.)
3. If contour fails, use line-based detection
4. Validate and refine the best result
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
from src.hough_detection import classify_lines, compute_all_intersections, extract_outer_corners
from src.geometry import order_corners, refine_corners, compute_iou

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


def score_quadrilateral(quad: np.ndarray, img_shape: Tuple[int, int]) -> float:
    """
    Score a quadrilateral based on geometric properties.
    Higher score = better candidate for sudoku grid.

    Factors:
    - Size relative to image (larger is better, up to a point)
    - Squareness (aspect ratio close to 1)
    - Angle regularity (corners close to 90 degrees)
    - Convexity
    """
    h, w = img_shape
    img_area = h * w

    # Area score (prefer 20-80% of image)
    quad_area = cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.int32))
    area_ratio = quad_area / img_area
    if area_ratio < 0.1:
        area_score = 0.1
    elif area_ratio > 0.9:
        area_score = 0.5
    else:
        area_score = min(1.0, area_ratio * 2)  # Max at 50%

    # Aspect ratio score (prefer square-ish)
    ordered = order_corners(quad)
    widths = [np.linalg.norm(ordered[1] - ordered[0]),
              np.linalg.norm(ordered[2] - ordered[3])]
    heights = [np.linalg.norm(ordered[3] - ordered[0]),
               np.linalg.norm(ordered[2] - ordered[1])]

    avg_width = np.mean(widths)
    avg_height = np.mean(heights)

    if avg_height > 0:
        aspect = avg_width / avg_height
        aspect_score = 1.0 - abs(1.0 - aspect) * 0.5
        aspect_score = max(0.1, aspect_score)
    else:
        aspect_score = 0.1

    # Angle score (prefer near-90-degree corners)
    def angle_at_corner(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    angles = []
    for i in range(4):
        p1 = ordered[(i - 1) % 4]
        p2 = ordered[i]
        p3 = ordered[(i + 1) % 4]
        angles.append(angle_at_corner(p1, p2, p3))

    angle_deviations = [abs(90 - a) for a in angles]
    avg_deviation = np.mean(angle_deviations)
    angle_score = max(0.1, 1.0 - avg_deviation / 45)

    # Combined score
    score = area_score * 0.3 + aspect_score * 0.35 + angle_score * 0.35

    return score


def find_quadrilaterals_contour(
    gray: np.ndarray,
    preprocess_methods: List[str] = None
) -> List[Tuple[np.ndarray, float, str]]:
    """
    Find quadrilateral candidates using contour detection.

    Returns list of (quad, score, method) tuples.
    """
    if preprocess_methods is None:
        preprocess_methods = ['adaptive', 'morph', 'canny', 'otsu']

    h, w = gray.shape
    img_area = h * w
    candidates = []

    for method in preprocess_methods:
        try:
            # Preprocessing
            if method == 'adaptive':
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                binary = cv2.adaptiveThreshold(
                    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
            elif method == 'morph':
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
                _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif method == 'canny':
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                binary = cv2.Canny(blur, 50, 150)
            elif method == 'otsu':
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                continue

            # Close gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.dilate(binary, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < img_area * 0.1 or area > img_area * 0.95:
                    continue

                # Try to approximate to quadrilateral
                peri = cv2.arcLength(contour, True)
                for eps in [0.01, 0.02, 0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(contour, eps * peri, True)
                    if len(approx) == 4:
                        quad = approx.reshape(4, 2).astype(np.float32)
                        if cv2.isContourConvex(approx):
                            score = score_quadrilateral(quad, gray.shape)
                            candidates.append((quad, score, f'contour_{method}'))
                        break

        except Exception as e:
            continue

    return candidates


def find_quadrilaterals_lines(
    gray: np.ndarray,
    edges: np.ndarray = None
) -> List[Tuple[np.ndarray, float, str]]:
    """
    Find quadrilateral candidates using line detection.
    """
    candidates = []

    if edges is None:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

    # Try multiple Hough parameters
    param_sets = [
        {'threshold': 50, 'minLength': 50, 'maxGap': 10},
        {'threshold': 40, 'minLength': 40, 'maxGap': 15},
        {'threshold': 60, 'minLength': 60, 'maxGap': 10},
    ]

    for params in param_sets:
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=params['threshold'],
            minLineLength=params['minLength'],
            maxLineGap=params['maxGap']
        )

        if lines is None or len(lines) < 4:
            continue

        h_lines, v_lines = classify_lines(lines)

        if len(h_lines) < 2 or len(v_lines) < 2:
            continue

        # Get boundary lines (outermost)
        h_sorted = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
        v_sorted = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)

        # Take outermost lines
        top_line = h_sorted[0]
        bottom_line = h_sorted[-1]
        left_line = v_sorted[0]
        right_line = v_sorted[-1]

        # Compute corners
        def line_intersection(l1, l2):
            x1, y1, x2, y2 = l1
            x3, y3, x4, y4 = l2

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                return None

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)

        tl = line_intersection(top_line, left_line)
        tr = line_intersection(top_line, right_line)
        br = line_intersection(bottom_line, right_line)
        bl = line_intersection(bottom_line, left_line)

        if None in [tl, tr, br, bl]:
            continue

        quad = np.array([tl, tr, br, bl], dtype=np.float32)

        # Validate bounds
        h, w = gray.shape
        valid = True
        for x, y in quad:
            if x < -50 or x > w + 50 or y < -50 or y > h + 50:
                valid = False
                break

        if valid:
            score = score_quadrilateral(quad, gray.shape)
            candidates.append((quad, score, 'lines'))

    return candidates


def detect_corners_robust(
    image: np.ndarray
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Robust corner detection combining multiple methods.
    """
    debug = {}

    # Preprocess
    resized, scale = resize_image(image, TARGET_SIZE)
    gray = to_grayscale(resized)
    debug['scale'] = scale

    # Collect candidates from different methods
    candidates = []

    # Method 1: Contour-based
    contour_candidates = find_quadrilaterals_contour(gray)
    candidates.extend(contour_candidates)

    # Method 2: Line-based
    line_candidates = find_quadrilaterals_lines(gray)
    candidates.extend(line_candidates)

    debug['num_candidates'] = len(candidates)

    if not candidates:
        debug['error'] = 'No candidates found'
        return None, debug

    # Select best candidate by score
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_quad, best_score, best_method = candidates[0]

    debug['method'] = best_method
    debug['score'] = best_score

    # Order corners
    corners = order_corners(best_quad)

    # Refine
    try:
        corners = refine_corners(gray, corners)
    except:
        pass

    # Scale back
    corners_orig = corners / scale

    return corners_orig, debug


def test_all_images():
    """Test on all sample images."""
    gt = load_ground_truth()
    images = get_sample_images()

    print("Testing Robust Detection")
    print("=" * 60)

    results = {}

    for img_path in images:
        name = img_path.name
        if name not in gt:
            continue

        image = cv2.imread(str(img_path))
        gt_corners = gt[name]

        corners, debug = detect_corners_robust(image)

        if corners is not None:
            iou = compute_iou(corners, gt_corners)
            status = 'PASS' if iou >= 0.95 else 'FAIL'
            print(f"{name}: IoU={iou:.4f} [{status}] via {debug['method']} (score={debug['score']:.3f})")
        else:
            iou = 0.0
            status = 'FAIL'
            print(f"{name}: FAILED - {debug.get('error')}")

        results[name] = {'iou': iou, 'status': status, 'method': debug.get('method')}

    print("\n" + "=" * 60)
    passed = sum(1 for r in results.values() if r['status'] == 'PASS')
    print(f"Passed: {passed}/{len(results)}")

    return results


if __name__ == '__main__':
    test_all_images()
