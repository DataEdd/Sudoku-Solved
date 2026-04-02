#!/usr/bin/env python3
"""
Deep analysis of the hardest case: _1_5057995.jpeg

This image has a tilted grid taking only ~17% of image area.
Goal: Achieve >95% IoU through specialized techniques.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path.cwd().parent))

from src.preprocessing import resize_image, to_grayscale
from src.geometry import order_corners, compute_iou

TARGET_SIZE = 1000
GROUND_TRUTH_FILE = Path('../data/ground_truth_corners.json')
EXAMPLES_DIR = Path('../examples')
IMAGE_NAME = '_1_5057995.jpeg'


def load_ground_truth() -> np.ndarray:
    with open(GROUND_TRUTH_FILE, 'r') as f:
        data = json.load(f)
    return np.array(data[IMAGE_NAME], dtype=np.float32)


def find_quadrilateral(binary: np.ndarray, min_area_ratio: float = 0.02) -> Optional[np.ndarray]:
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

        for eps in [0.01, 0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(contour, eps * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
                quad_area = cv2.contourArea(approx)

                if quad_area > best_area:
                    best_area = quad_area
                    best_quad = quad
                    break

    return best_quad


def try_config(gray: np.ndarray, blur_size: int, block_size: int,
               c_val: int, dilation: int, use_clahe: bool = False,
               morph_close: bool = False) -> Optional[np.ndarray]:
    """Try adaptive thresholding with specific parameters."""

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_val
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    if morph_close:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    if dilation > 0:
        binary = cv2.dilate(binary, kernel, iterations=dilation)

    return find_quadrilateral(binary)


def try_canny_config(gray: np.ndarray, low: int, high: int,
                     dilation: int) -> Optional[np.ndarray]:
    """Try Canny edge detection approach."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, low, high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if dilation > 0:
        edges = cv2.dilate(edges, kernel, iterations=dilation)

    return find_quadrilateral(edges)


def try_morph_gradient(gray: np.ndarray, dilation: int) -> Optional[np.ndarray]:
    """Try morphological gradient approach."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if dilation > 0:
        binary = cv2.dilate(binary, kernel, iterations=dilation)

    return find_quadrilateral(binary)


def try_bilateral_canny(gray: np.ndarray, low: int, high: int,
                        dilation: int) -> Optional[np.ndarray]:
    """Try bilateral filter + Canny."""
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(filtered, low, high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if dilation > 0:
        edges = cv2.dilate(edges, kernel, iterations=dilation)

    return find_quadrilateral(edges)


def try_sobel_approach(gray: np.ndarray, dilation: int) -> Optional[np.ndarray]:
    """Try Sobel-based edge detection."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = np.uint8(255 * magnitude / magnitude.max())

    _, binary = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if dilation > 0:
        binary = cv2.dilate(binary, kernel, iterations=dilation)

    return find_quadrilateral(binary)


def analyze_and_optimize():
    """Run comprehensive analysis on the hard case."""
    gt_corners = load_ground_truth()
    img_path = EXAMPLES_DIR / IMAGE_NAME
    image = cv2.imread(str(img_path))

    resized, scale = resize_image(image, TARGET_SIZE)
    gray = to_grayscale(resized)

    # Scale ground truth to resized coordinates
    gt_scaled = gt_corners * scale

    print(f"Analyzing {IMAGE_NAME}")
    print(f"Image size: {image.shape[:2]}, Resized: {resized.shape[:2]}, Scale: {scale:.4f}")
    print(f"Ground truth (scaled): TL={gt_scaled[0]}, TR={gt_scaled[1]}, BR={gt_scaled[2]}, BL={gt_scaled[3]}")
    print("=" * 80)

    results = []

    # Test 1: Adaptive thresholding with extended parameter ranges
    print("\n1. ADAPTIVE THRESHOLDING (Extended)")
    for blur_size in [3, 5, 7, 9]:
        for block_size in [5, 7, 9, 11, 15, 21]:
            for c_val in [1, 2, 3, 4, 5, 6, 8, 10]:
                for dilation in [0, 1, 2]:
                    for use_clahe in [False, True]:
                        for morph_close in [False, True]:
                            quad = try_config(gray, blur_size, block_size, c_val,
                                            dilation, use_clahe, morph_close)
                            if quad is not None:
                                corners = order_corners(quad)
                                corners_orig = corners / scale
                                iou = compute_iou(corners_orig, gt_corners)

                                config_name = f"adaptive_b{blur_size}_blk{block_size}_c{c_val}_d{dilation}"
                                if use_clahe:
                                    config_name += "_clahe"
                                if morph_close:
                                    config_name += "_close"

                                results.append((iou, config_name, corners_orig))

    # Test 2: Canny-based approaches
    print("\n2. CANNY EDGE DETECTION")
    for low in [20, 30, 40, 50, 60, 70]:
        for high in [80, 100, 120, 150, 180, 200]:
            if high > low:
                for dilation in [1, 2, 3]:
                    quad = try_canny_config(gray, low, high, dilation)
                    if quad is not None:
                        corners = order_corners(quad)
                        corners_orig = corners / scale
                        iou = compute_iou(corners_orig, gt_corners)
                        results.append((iou, f"canny_l{low}_h{high}_d{dilation}", corners_orig))

    # Test 3: Bilateral + Canny
    print("\n3. BILATERAL + CANNY")
    for low in [30, 50, 70]:
        for high in [100, 150, 200]:
            for dilation in [1, 2, 3]:
                quad = try_bilateral_canny(gray, low, high, dilation)
                if quad is not None:
                    corners = order_corners(quad)
                    corners_orig = corners / scale
                    iou = compute_iou(corners_orig, gt_corners)
                    results.append((iou, f"bilateral_canny_l{low}_h{high}_d{dilation}", corners_orig))

    # Test 4: Morphological gradient
    print("\n4. MORPHOLOGICAL GRADIENT")
    for dilation in [1, 2, 3, 4]:
        quad = try_morph_gradient(gray, dilation)
        if quad is not None:
            corners = order_corners(quad)
            corners_orig = corners / scale
            iou = compute_iou(corners_orig, gt_corners)
            results.append((iou, f"morph_grad_d{dilation}", corners_orig))

    # Test 5: Sobel-based
    print("\n5. SOBEL-BASED")
    for dilation in [1, 2, 3, 4]:
        quad = try_sobel_approach(gray, dilation)
        if quad is not None:
            corners = order_corners(quad)
            corners_orig = corners / scale
            iou = compute_iou(corners_orig, gt_corners)
            results.append((iou, f"sobel_d{dilation}", corners_orig))

    # Sort and display top results
    results.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "=" * 80)
    print("TOP 20 RESULTS:")
    print("=" * 80)

    for i, (iou, config, corners) in enumerate(results[:20]):
        status = "PASS" if iou >= 0.95 else "FAIL"
        print(f"{i+1:2}. IoU={iou:.4f} [{status}] {config}")
        if i < 5:
            print(f"     Corners: TL={corners[0]}, TR={corners[1]}, BR={corners[2]}, BL={corners[3]}")

    # Analyze best result vs ground truth
    if results:
        best_iou, best_config, best_corners = results[0]
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS OF BEST RESULT:")
        print(f"Config: {best_config}")
        print(f"IoU: {best_iou:.4f}")
        print("\nPer-corner errors (detected - ground truth):")
        corner_names = ['TL', 'TR', 'BR', 'BL']
        for i, name in enumerate(corner_names):
            error = best_corners[i] - gt_corners[i]
            dist = np.linalg.norm(error)
            print(f"  {name}: dx={error[0]:+.1f}, dy={error[1]:+.1f}, dist={dist:.1f}px")

    return results


if __name__ == '__main__':
    analyze_and_optimize()
