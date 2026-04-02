#!/usr/bin/env python3
"""Debug script to understand candidate selection."""

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


def load_ground_truth() -> Dict:
    with open(GROUND_TRUTH_FILE, 'r') as f:
        data = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in data.items() if '.' in k}


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


def score_quadrilateral(quad: np.ndarray, img_shape: Tuple[int, int]) -> Tuple[float, dict]:
    """Score a quadrilateral and return component scores."""
    h, w = img_shape
    img_area = h * w

    quad_area = cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.int32))
    area_ratio = quad_area / img_area

    if area_ratio < 0.05:
        area_score = 0.1
    elif area_ratio > 0.9:
        area_score = 0.5
    else:
        area_score = min(1.0, area_ratio * 1.5)

    ordered = order_corners(quad)
    widths = [np.linalg.norm(ordered[1] - ordered[0]),
              np.linalg.norm(ordered[2] - ordered[3])]
    heights = [np.linalg.norm(ordered[3] - ordered[0]),
               np.linalg.norm(ordered[2] - ordered[1])]

    avg_width = np.mean(widths)
    avg_height = np.mean(heights)

    if avg_height > 0 and avg_width > 0:
        aspect = min(avg_width, avg_height) / max(avg_width, avg_height)
        aspect_score = aspect
    else:
        aspect_score = 0.1

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

    total = area_score * 0.5 + aspect_score * 0.25 + angle_score * 0.25

    details = {
        'area_ratio': area_ratio,
        'area_score': area_score,
        'aspect': aspect,
        'aspect_score': aspect_score,
        'avg_angle_dev': avg_deviation,
        'angle_score': angle_score,
        'total': total
    }

    return total, details


def analyze_image(name: str, gt_corners: np.ndarray):
    img_path = EXAMPLES_DIR / name
    image = cv2.imread(str(img_path))
    resized, scale = resize_image(image, TARGET_SIZE)
    gray = to_grayscale(resized)

    print(f"\n{'='*80}")
    print(f"Image: {name}")
    print(f"{'='*80}")

    configs = [
        (7, 7, 5, 0, False),
        (7, 11, 5, 0, False),
        (5, 11, 2, 0, False),
        (5, 11, 2, 1, False),
        (5, 5, 10, 1, True),
        (7, 5, 8, 2, True),
        (5, 5, 10, 2, True),
        (3, 7, 2, 0, False),
    ]

    candidates = []

    for blur_size, block_size, c_val, dilation, use_clahe in configs:
        quad = try_adaptive_detection(gray, blur_size, block_size, c_val, dilation, use_clahe)

        if quad is not None:
            score, details = score_quadrilateral(quad, gray.shape)
            clahe_str = "_clahe" if use_clahe else ""
            method = f"b{blur_size}_blk{block_size}_c{c_val}_d{dilation}{clahe_str}"

            corners = order_corners(quad) / scale
            iou = compute_iou(corners, gt_corners)

            candidates.append({
                'method': method,
                'iou': iou,
                'score': score,
                'details': details
            })

    # Add morph gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for dilation in [1, 2]:
        dilated = cv2.dilate(binary, kernel, iterations=dilation)
        quad = find_quadrilateral(dilated)

        if quad is not None:
            score, details = score_quadrilateral(quad, gray.shape)
            corners = order_corners(quad) / scale
            iou = compute_iou(corners, gt_corners)

            candidates.append({
                'method': f"morph_d{dilation}",
                'iou': iou,
                'score': score,
                'details': details
            })

    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)

    print(f"{'Method':<30} {'IoU':>8} {'Score':>8} {'Area%':>8} {'Aspect':>8}")
    print("-" * 70)

    for c in candidates:
        d = c['details']
        iou_str = f"{c['iou']:.4f}"
        status = "PASS" if c['iou'] >= 0.95 else ""
        print(f"{c['method']:<30} {iou_str:>8} {c['score']:>8.3f} {d['area_ratio']*100:>7.1f}% {d['aspect']:>8.3f} {status}")

    # Find best by IoU and best by score
    best_by_score = candidates[0]
    best_by_iou = max(candidates, key=lambda x: x['iou'])

    print(f"\nBest by SCORE: {best_by_score['method']} (IoU={best_by_score['iou']:.4f})")
    print(f"Best by IoU:   {best_by_iou['method']} (IoU={best_by_iou['iou']:.4f})")


def main():
    gt = load_ground_truth()

    for name in ['_1_5057995.jpeg', '_118_4579703.jpeg', '_278_3243716.jpeg']:
        if name in gt:
            analyze_image(name, gt[name])


if __name__ == '__main__':
    main()
