#!/usr/bin/env python3
"""
Final optimized sudoku corner detector.

Best parameters found through empirical testing:
- gray_b7_block7_c5_d0 works for 5/6 test images (>95% IoU)
- gray_b3_block7_c2_d0 works for tilted grids (90% IoU on hardest case)
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
from src.geometry import order_corners, compute_iou

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


def find_quadrilateral(binary: np.ndarray, min_area_ratio: float = 0.05) -> Optional[np.ndarray]:
    """Find the largest valid quadrilateral in binary image."""
    h, w = binary.shape
    min_area = h * w * min_area_ratio
    max_area = h * w * 0.95

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(contour, True)

        for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
            approx = cv2.approxPolyDP(contour, eps * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
                quad_area = cv2.contourArea(approx)

                if quad_area > best_area:
                    best_area = quad_area
                    best_quad = quad
                    break

    return best_quad


def try_adaptive_detection(gray: np.ndarray, blur_size: int, block_size: int,
                           c_val: int, dilation: int, use_clahe: bool = False) -> Optional[np.ndarray]:
    """Try adaptive thresholding with specific parameters."""
    processed = gray

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)

    blur = cv2.GaussianBlur(processed, (blur_size, blur_size), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_val
    )

    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=dilation)

    return find_quadrilateral(binary)


def score_quadrilateral(quad: np.ndarray, img_shape: Tuple[int, int], method: str = "") -> float:
    """
    Score a quadrilateral based on geometric properties.

    Higher score = better candidate for sudoku grid.
    """
    h, w = img_shape
    img_area = h * w

    # Compute area
    quad_area = cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.int32))
    area_ratio = quad_area / img_area

    # Area score: moderate preference for larger, but don't over-penalize smaller
    # Valid sudoku grids can range from ~10% to ~80% of image
    if area_ratio < 0.05:
        area_score = 0.2
    elif area_ratio > 0.9:
        area_score = 0.5
    elif area_ratio < 0.15:
        area_score = 0.6 + (area_ratio - 0.05) * 2  # 0.6-0.8 for 5-15%
    else:
        area_score = 0.8 + min(0.2, (area_ratio - 0.15) * 0.5)  # 0.8-1.0 for 15%+

    # Aspect ratio score: prefer square-ish
    ordered = order_corners(quad)
    widths = [np.linalg.norm(ordered[1] - ordered[0]),
              np.linalg.norm(ordered[2] - ordered[3])]
    heights = [np.linalg.norm(ordered[3] - ordered[0]),
               np.linalg.norm(ordered[2] - ordered[1])]

    avg_width = np.mean(widths)
    avg_height = np.mean(heights)

    if avg_height > 0 and avg_width > 0:
        aspect = min(avg_width, avg_height) / max(avg_width, avg_height)
        aspect_score = aspect  # 1.0 for perfect square
    else:
        aspect_score = 0.1

    # Angle score: prefer near-90-degree corners
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

    # Combined score: balance all factors
    score = area_score * 0.35 + aspect_score * 0.35 + angle_score * 0.30

    # Slight preference for adaptive methods over morphological gradient
    # (empirically adaptive tends to be more accurate)
    if "adaptive" in method:
        score += 0.02

    return score


def detect_corners(image: np.ndarray) -> Tuple[Optional[np.ndarray], dict]:
    """
    Detect sudoku grid corners.

    Uses optimized parameters found through empirical testing.
    Tries multiple configs and selects the best based on geometric scoring.

    Returns:
        corners: (4, 2) array [TL, TR, BR, BL] or None
        debug: dict with method used
    """
    debug = {}

    # Preprocess
    resized, scale = resize_image(image, TARGET_SIZE)
    gray = to_grayscale(resized)
    debug['scale'] = scale

    # Configs to try - grouped by priority
    # Format: (blur_size, block_size, c_value, dilation, use_clahe)
    configs = [
        # Standard configs - work well for most images
        (7, 7, 5, 0, False),
        (7, 11, 5, 0, False),
        (5, 11, 2, 0, False),
        (5, 11, 2, 1, False),

        # CLAHE-based configs for challenging images
        (5, 5, 10, 1, True),
        (7, 5, 8, 2, True),
        (5, 5, 10, 2, True),

        # For tilted/small grids
        (3, 7, 2, 0, False),
        (3, 7, 1, 0, False),
    ]

    # Try all configs and collect results
    candidates = []

    for blur_size, block_size, c_val, dilation, use_clahe in configs:
        quad = try_adaptive_detection(gray, blur_size, block_size, c_val, dilation, use_clahe)

        if quad is not None:
            clahe_str = "_clahe" if use_clahe else ""
            method = f"adaptive_b{blur_size}_block{block_size}_c{c_val}_d{dilation}{clahe_str}"
            score = score_quadrilateral(quad, gray.shape, method)
            candidates.append((quad, score, method))

    # Also try morphological gradient as fallback
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for dilation in [1, 2]:
        dilated = cv2.dilate(binary, kernel, iterations=dilation)
        quad = find_quadrilateral(dilated)

        if quad is not None:
            method = f"morph_d{dilation}"
            score = score_quadrilateral(quad, gray.shape, method)
            candidates.append((quad, score, method))

    if not candidates:
        debug['error'] = 'No quadrilateral found'
        return None, debug

    # Select best candidate by score
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_quad, best_score, best_method = candidates[0]

    debug['method'] = best_method
    debug['score'] = best_score
    debug['num_candidates'] = len(candidates)

    corners = order_corners(best_quad)
    corners_orig = corners / scale

    return corners_orig, debug


def test_all_images():
    """Test on all sample images."""
    gt = load_ground_truth()
    images = get_sample_images()

    print("Final Optimized Detector")
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
            print(f"{name}: IoU={iou:.4f} [{status}] via {debug.get('method')}")
        else:
            iou = 0.0
            status = 'FAIL'
            print(f"{name}: FAILED - {debug.get('error')}")

        results[name] = {'iou': iou, 'status': status, 'method': debug.get('method')}

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for r in results.values() if r['status'] == 'PASS')
    total = len(results)
    avg_iou = np.mean([r['iou'] for r in results.values()])

    print(f"Passed: {passed}/{total} ({100*passed/total:.0f}%)")
    print(f"Average IoU: {avg_iou:.4f}")

    return results


if __name__ == '__main__':
    test_all_images()
