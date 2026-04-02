#!/usr/bin/env python3
"""Trace why CLAHE config doesn't find quad in debug script."""

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


def main():
    gt_corners = load_ground_truth()
    img_path = EXAMPLES_DIR / IMAGE_NAME
    image = cv2.imread(str(img_path))
    resized, scale = resize_image(image, TARGET_SIZE)
    gray = to_grayscale(resized)

    h, w = gray.shape
    img_area = h * w
    min_area_ratio = 0.05
    min_area = img_area * min_area_ratio
    max_area = img_area * 0.95

    print(f"Image: {IMAGE_NAME}")
    print(f"Shape: {gray.shape}, min_area={min_area:.0f}, max_area={max_area:.0f}")
    print()

    # Test CLAHE config: (5, 5, 10, 1, True)
    blur_size, block_size, c_val, dilation = 5, 5, 10, 1

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(gray)

    blur = cv2.GaussianBlur(processed, (blur_size, blur_size), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_val
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=dilation)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Total contours: {len(contours)}")
    print()

    best_quad = None
    best_area = 0

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 1000:  # Skip tiny
            continue

        peri = cv2.arcLength(contour, True)

        print(f"Contour {i}: area={area:.0f} ({area/img_area*100:.1f}%), peri={peri:.1f}")

        if area < min_area:
            print(f"  -> SKIP: area < min_area ({min_area:.0f})")
            continue
        if area > max_area:
            print(f"  -> SKIP: area > max_area")
            continue

        for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            is_convex = cv2.isContourConvex(approx) if len(approx) >= 3 else False

            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)
                quad_area = cv2.contourArea(approx)
                print(f"  eps={eps}: {len(approx)} pts, convex={is_convex}, area={quad_area:.0f}")

                if is_convex and quad_area > best_area:
                    best_area = quad_area
                    best_quad = quad
                    print(f"       -> BEST CANDIDATE!")
                    break
            else:
                print(f"  eps={eps}: {len(approx)} pts (not 4)")

    print()
    if best_quad is not None:
        corners = order_corners(best_quad) / scale
        iou = compute_iou(corners, gt_corners)
        print(f"Found quad with area={best_area:.0f} ({best_area/img_area*100:.1f}%), IoU={iou:.4f}")
    else:
        print("No quadrilateral found!")


if __name__ == '__main__':
    main()
