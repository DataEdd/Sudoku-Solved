#!/usr/bin/env python3
"""
Contour-based sudoku grid detection.

Strategy:
1. Find contours in the edge image
2. Approximate contours to polygons
3. Find the largest quadrilateral (4-sided polygon)
4. This should be the sudoku grid boundary
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


def find_largest_quadrilateral(
    contours: List[np.ndarray],
    img_shape: Tuple[int, int],
    min_area_ratio: float = 0.1,
    max_area_ratio: float = 0.95
) -> Optional[np.ndarray]:
    """
    Find the largest 4-sided contour.

    Args:
        contours: List of contours from cv2.findContours
        img_shape: (height, width) of image
        min_area_ratio: Minimum area as ratio of image area
        max_area_ratio: Maximum area as ratio of image area

    Returns:
        4 corner points as (4, 2) array, or None
    """
    h, w = img_shape
    img_area = h * w
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio

    best_quad = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by area
        if area < min_area or area > max_area:
            continue

        # Approximate to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Check if quadrilateral
        if len(approx) == 4:
            # Check if convex
            if cv2.isContourConvex(approx):
                if area > best_area:
                    best_area = area
                    best_quad = approx.reshape(4, 2)

    return best_quad


def detect_by_contour(
    image: np.ndarray,
    params: dict
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Detect sudoku corners using contour detection.

    Returns:
        corners: (4, 2) array in original image coordinates, or None
        debug: dict with intermediate results
    """
    debug = {}

    # Preprocess
    resized, scale = resize_image(image, TARGET_SIZE)
    gray = to_grayscale(resized)
    debug['scale'] = scale
    debug['resized'] = resized

    # Try multiple preprocessing approaches
    preprocessed_images = []

    # Approach 1: Bilateral + Canny
    filtered = cv2.bilateralFilter(
        gray,
        d=params.get('bilateral_d', 9),
        sigmaColor=params.get('bilateral_sigma_color', 75),
        sigmaSpace=params.get('bilateral_sigma_space', 75)
    )
    edges1 = cv2.Canny(
        filtered,
        params.get('canny_low', 50),
        params.get('canny_high', 150)
    )
    preprocessed_images.append(('bilateral_canny', edges1))

    # Approach 2: Adaptive threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        params.get('adaptive_block', 11),
        params.get('adaptive_c', 2)
    )
    preprocessed_images.append(('adaptive_thresh', thresh))

    # Approach 3: Morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, grad_thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(('morph_gradient', grad_thresh))

    # Approach 4: Canny with different params
    edges2 = cv2.Canny(blur, 30, 100)
    preprocessed_images.append(('canny_sensitive', edges2))

    # Try to find quadrilateral in each
    best_quad = None
    best_area = 0
    best_method = None

    for method_name, binary_img in preprocessed_images:
        # Dilate to connect edges
        dilated = cv2.dilate(binary_img, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        quad = find_largest_quadrilateral(contours, gray.shape[:2])

        if quad is not None:
            area = cv2.contourArea(quad)
            if area > best_area:
                best_area = area
                best_quad = quad
                best_method = method_name

    debug['method'] = best_method

    if best_quad is None:
        debug['error'] = 'No quadrilateral found'
        return None, debug

    # Order corners
    corners = order_corners(best_quad.astype(np.float32))

    # Refine
    try:
        corners = refine_corners(gray, corners)
    except:
        pass

    # Scale back
    corners_orig = corners / scale

    debug['detected_corners'] = corners_orig
    return corners_orig, debug


def detect_hybrid(
    image: np.ndarray,
    params: dict
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Hybrid detection combining contour and line-based approaches.

    1. Try contour detection first
    2. If IoU is low, try line-based refinement
    """
    # Try contour detection
    corners, debug = detect_by_contour(image, params)

    if corners is not None:
        debug['method_used'] = 'contour'
        return corners, debug

    debug['error'] = 'All methods failed'
    return None, debug


def test_on_all_images():
    """Test contour-based detection on all sample images."""
    gt = load_ground_truth()
    images = get_sample_images()

    print("Testing Contour-Based Detection")
    print("=" * 60)

    param_sets = [
        {'name': 'default'},
        {'name': 'sensitive', 'canny_low': 30, 'canny_high': 100, 'adaptive_block': 15},
        {'name': 'strict', 'canny_low': 70, 'canny_high': 200, 'adaptive_block': 7},
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
        best_method = None

        for params in param_sets:
            corners, debug = detect_by_contour(image, params)

            if corners is not None:
                iou = compute_iou(corners, gt_corners)
                print(f"  {params['name']} ({debug.get('method', '?')}): IoU = {iou:.4f}")

                if iou > best_iou:
                    best_iou = iou
                    best_params = params
                    best_corners = corners
                    best_method = debug.get('method')
            else:
                print(f"  {params['name']}: FAILED - {debug.get('error', 'unknown')}")

        results[name] = {
            'best_iou': best_iou,
            'best_params': best_params,
            'best_method': best_method,
            'corners': best_corners,
            'status': 'PASS' if best_iou >= 0.95 else 'FAIL'
        }

        print(f"  -> Best: {best_iou:.4f} [{results[name]['status']}]")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r['status'] == 'PASS')
    print(f"\nPassed: {passed}/{len(results)}")

    for name, r in results.items():
        print(f"  {name}: {r['best_iou']:.4f} [{r['status']}] via {r['best_method']}")

    return results


if __name__ == '__main__':
    test_on_all_images()
