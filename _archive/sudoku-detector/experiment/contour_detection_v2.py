#!/usr/bin/env python3
"""
Improved contour-based detection with multiple strategies.
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path.cwd().parent))

from src.preprocessing import resize_image, to_grayscale
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


def find_quadrilateral_from_contour(
    contour: np.ndarray,
    img_shape: Tuple[int, int],
    epsilon_range: List[float] = [0.01, 0.02, 0.03, 0.04, 0.05]
) -> Optional[np.ndarray]:
    """
    Try to approximate contour to a quadrilateral with various tolerances.
    """
    peri = cv2.arcLength(contour, True)

    for eps in epsilon_range:
        approx = cv2.approxPolyDP(contour, eps * peri, True)

        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            if cv2.isContourConvex(quad.reshape(-1, 1, 2).astype(np.int32)):
                return quad

    return None


def find_quadrilateral_from_hull(
    contour: np.ndarray,
    img_shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    """
    Find quadrilateral by fitting to convex hull.
    Uses the 4 points that maximize the quadrilateral area.
    """
    hull = cv2.convexHull(contour)

    if len(hull) < 4:
        return None

    hull_points = hull.reshape(-1, 2)

    # If hull already has 4 points, use them
    if len(hull_points) == 4:
        return hull_points.astype(np.float32)

    # Find 4 corners from hull using distance from centroid
    centroid = hull_points.mean(axis=0)
    angles = np.arctan2(hull_points[:, 1] - centroid[1],
                        hull_points[:, 0] - centroid[0])

    # Sort by angle
    sorted_indices = np.argsort(angles)
    sorted_points = hull_points[sorted_indices]

    # Divide into 4 quadrants and pick furthest point from centroid in each
    n = len(sorted_points)
    quadrant_size = n // 4

    corners = []
    for i in range(4):
        start = i * quadrant_size
        end = start + quadrant_size if i < 3 else n

        quadrant_points = sorted_points[start:end]
        distances = np.linalg.norm(quadrant_points - centroid, axis=1)
        furthest_idx = np.argmax(distances)
        corners.append(quadrant_points[furthest_idx])

    return np.array(corners, dtype=np.float32)


def find_best_quadrilateral(
    contours: List[np.ndarray],
    img_shape: Tuple[int, int],
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.98
) -> Tuple[Optional[np.ndarray], str]:
    """
    Find the best quadrilateral from contours.

    Returns:
        quad: 4 corner points, or None
        method: how the quad was found
    """
    h, w = img_shape
    img_area = h * w
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        # Try polygon approximation first
        quad = find_quadrilateral_from_contour(contour, img_shape)
        if quad is not None:
            candidates.append((quad, area, 'approx'))
            continue

        # Try convex hull
        quad = find_quadrilateral_from_hull(contour, img_shape)
        if quad is not None:
            quad_area = cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.int32))
            if quad_area >= min_area:
                candidates.append((quad, quad_area, 'hull'))

    if not candidates:
        return None, 'none'

    # Return largest
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0], candidates[0][2]


def preprocess_for_contours(
    gray: np.ndarray,
    method: str,
    params: dict
) -> np.ndarray:
    """Apply preprocessing to get binary image for contour detection."""

    if method == 'adaptive':
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            params.get('block_size', 11),
            params.get('c', 2)
        )

    elif method == 'canny':
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.Canny(
            blur,
            params.get('canny_low', 50),
            params.get('canny_high', 150)
        )

    elif method == 'bilateral_canny':
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        binary = cv2.Canny(
            filtered,
            params.get('canny_low', 50),
            params.get('canny_high', 150)
        )

    elif method == 'morph_gradient':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == 'otsu':
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    else:
        raise ValueError(f"Unknown method: {method}")

    return binary


def close_edges(binary: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Close gaps in edges using morphological operations."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Dilate to connect nearby edges
    closed = cv2.dilate(binary, kernel, iterations=iterations)

    # Optional: erode slightly to maintain edge positions
    # closed = cv2.erode(closed, kernel, iterations=1)

    return closed


def detect_corners(
    image: np.ndarray,
    params: dict = None
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Detect sudoku corners using robust contour detection.
    """
    if params is None:
        params = {}

    debug = {}

    # Preprocess
    resized, scale = resize_image(image, TARGET_SIZE)
    gray = to_grayscale(resized)
    debug['scale'] = scale

    # Try multiple preprocessing methods
    methods = [
        ('adaptive', {'block_size': 11, 'c': 2}),
        ('adaptive', {'block_size': 15, 'c': 3}),
        ('adaptive', {'block_size': 21, 'c': 4}),
        ('morph_gradient', {}),
        ('canny', {'canny_low': 30, 'canny_high': 100}),
        ('canny', {'canny_low': 50, 'canny_high': 150}),
        ('bilateral_canny', {'canny_low': 50, 'canny_high': 150}),
        ('otsu', {}),
    ]

    best_quad = None
    best_area = 0
    best_method = None

    for method_name, method_params in methods:
        try:
            binary = preprocess_for_contours(gray, method_name, method_params)

            # Try different levels of edge closing
            for close_iter in [1, 2, 3]:
                closed = close_edges(binary, close_iter)

                # Find contours
                contours, _ = cv2.findContours(
                    closed,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                quad, find_method = find_best_quadrilateral(contours, gray.shape[:2])

                if quad is not None:
                    area = cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.int32))
                    if area > best_area:
                        best_area = area
                        best_quad = quad
                        best_method = f"{method_name}_{find_method}_close{close_iter}"

        except Exception as e:
            continue

    if best_quad is None:
        debug['error'] = 'No quadrilateral found'
        return None, debug

    debug['method'] = best_method

    # Order corners: TL, TR, BR, BL
    corners = order_corners(best_quad)

    # Refine corners
    try:
        corners = refine_corners(gray, corners)
    except:
        pass

    # Scale back
    corners_orig = corners / scale

    return corners_orig, debug


def test_on_all_images():
    """Test on all sample images."""
    gt = load_ground_truth()
    images = get_sample_images()

    print("Testing Improved Contour Detection")
    print("=" * 60)

    results = {}

    for img_path in images:
        name = img_path.name
        if name not in gt:
            continue

        image = cv2.imread(str(img_path))
        gt_corners = gt[name]

        corners, debug = detect_corners(image)

        if corners is not None:
            iou = compute_iou(corners, gt_corners)
            status = 'PASS' if iou >= 0.95 else 'FAIL'
            print(f"{name}: IoU = {iou:.4f} [{status}] via {debug.get('method', '?')}")
        else:
            iou = 0.0
            status = 'FAIL'
            print(f"{name}: FAILED - {debug.get('error', 'unknown')}")

        results[name] = {
            'iou': iou,
            'status': status,
            'method': debug.get('method'),
            'corners': corners
        }

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for r in results.values() if r['status'] == 'PASS')
    print(f"Passed: {passed}/{len(results)}")

    return results


if __name__ == '__main__':
    test_on_all_images()
