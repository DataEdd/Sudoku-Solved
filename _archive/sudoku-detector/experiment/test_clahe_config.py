#!/usr/bin/env python3
"""Test specific CLAHE config on hard case."""

import sys
import json
from pathlib import Path

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


def find_quadrilateral(binary: np.ndarray, min_area_ratio: float = 0.02) -> np.ndarray:
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


def test_clahe():
    gt_corners = load_ground_truth()
    img_path = EXAMPLES_DIR / IMAGE_NAME
    image = cv2.imread(str(img_path))
    resized, scale = resize_image(image, TARGET_SIZE)
    gray = to_grayscale(resized)

    print(f"Image: {IMAGE_NAME}")
    print(f"Original shape: {image.shape}, Resized: {resized.shape}, Scale: {scale}")
    print()

    # Test the specific config that worked
    configs = [
        # From analyze_hard_case.py - this achieved 96.5% IoU
        (5, 5, 10, 1, True, 0.02),  # with min_area_ratio=0.02
        (7, 5, 8, 2, True, 0.02),
        # Same configs with different min_area
        (5, 5, 10, 1, True, 0.05),  # with min_area_ratio=0.05
        (7, 5, 8, 2, True, 0.05),
    ]

    for blur_size, block_size, c_val, dilation, use_clahe, min_area_ratio in configs:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(gray)

        blur = cv2.GaussianBlur(processed, (blur_size, blur_size), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c_val
        )

        if dilation > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.dilate(binary, kernel, iterations=dilation)

        # Count contours
        contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = binary.shape
        valid_contours = [c for c in contours if cv2.contourArea(c) >= h*w*min_area_ratio]

        quad = find_quadrilateral(binary, min_area_ratio)

        config_name = f"b{blur_size}_blk{block_size}_c{c_val}_d{dilation}_clahe_minarea{min_area_ratio}"

        if quad is not None:
            corners = order_corners(quad) / scale
            iou = compute_iou(corners, gt_corners)
            area_ratio = cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.int32)) / (h * w)
            print(f"{config_name}: IoU={iou:.4f}, area={area_ratio*100:.1f}%, contours={len(valid_contours)}")
        else:
            print(f"{config_name}: No quad found, contours={len(valid_contours)}")


if __name__ == '__main__':
    test_clahe()
