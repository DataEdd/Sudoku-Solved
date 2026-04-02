#!/usr/bin/env python3
"""
Diagnose why corner detection fails on specific images.
Visualize each pipeline step and compare with ground truth.
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


def dbscan_cluster_lines(lines: List, is_horizontal: bool, eps: float, min_samples: int):
    """DBSCAN clustering on line intercepts."""
    if not lines:
        return [], np.array([]), np.array([])

    if is_horizontal:
        intercepts = np.array([(l[1] + l[3]) / 2 for l in lines]).reshape(-1, 1)
    else:
        intercepts = np.array([(l[0] + l[2]) / 2 for l in lines]).reshape(-1, 1)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(intercepts)

    unique_labels = set(labels)
    unique_labels.discard(-1)

    result = []
    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        cluster_intercepts = intercepts[cluster_mask].flatten()
        cluster_lines = [lines[i] for i in range(len(lines)) if labels[i] == cluster_id]
        median_intercept = np.median(cluster_intercepts)
        best_idx = np.argmin(np.abs(cluster_intercepts - median_intercept))
        result.append(cluster_lines[best_idx])

    if is_horizontal:
        result.sort(key=lambda l: (l[1] + l[3]) / 2)
    else:
        result.sort(key=lambda l: (l[0] + l[2]) / 2)

    return result, intercepts.flatten(), labels


def run_pipeline_diagnostic(img_path: Path, gt_corners: np.ndarray, params: dict) -> dict:
    """Run pipeline and collect diagnostic info at each step."""

    # Load image
    original = cv2.imread(str(img_path))
    resized, scale = resize_image(original, TARGET_SIZE)
    gray = to_grayscale(resized)

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

    # Hough lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=params.get('hough_threshold', 60),
        minLineLength=params.get('hough_min_length', 50),
        maxLineGap=params.get('hough_max_gap', 10)
    )

    if lines is None:
        return {'error': 'No lines detected', 'stage': 'hough'}

    # Classify lines
    h_lines, v_lines = classify_lines(lines)

    if not h_lines or not v_lines:
        return {'error': f'Classification failed: H={len(h_lines)}, V={len(v_lines)}', 'stage': 'classify'}

    # DBSCAN clustering
    h_clustered, h_intercepts, h_labels = dbscan_cluster_lines(
        h_lines, True,
        params.get('dbscan_eps', 20),
        params.get('dbscan_min_samples', 2)
    )
    v_clustered, v_intercepts, v_labels = dbscan_cluster_lines(
        v_lines, False,
        params.get('dbscan_eps', 20),
        params.get('dbscan_min_samples', 2)
    )

    if not h_clustered or not v_clustered:
        return {'error': f'Clustering failed: H={len(h_clustered)}, V={len(v_clustered)}', 'stage': 'cluster'}

    # Compute intersections
    intersections = compute_all_intersections(h_clustered, v_clustered, gray.shape[:2])

    if len(intersections) < 4:
        return {'error': f'Only {len(intersections)} intersections', 'stage': 'intersect'}

    # Extract corners
    corners = extract_outer_corners(intersections)

    if corners is None:
        return {'error': 'Corner extraction failed', 'stage': 'corners'}

    # Order and refine
    ordered = order_corners(corners)
    refined = refine_corners(gray, ordered)

    # Scale to original
    detected = refined / scale

    # Compute IoU
    iou = compute_iou(detected, gt_corners)

    return {
        'success': True,
        'iou': iou,
        'detected': detected,
        'gt': gt_corners,
        'resized': resized,
        'gray': gray,
        'edges': edges,
        'lines': lines,
        'h_lines': h_lines,
        'v_lines': v_lines,
        'h_clustered': h_clustered,
        'v_clustered': v_clustered,
        'intersections': intersections,
        'scale': scale,
    }


def analyze_corner_errors(detected: np.ndarray, gt: np.ndarray) -> dict:
    """Analyze per-corner errors."""
    labels = ['TL', 'TR', 'BR', 'BL']
    errors = {}

    for i, label in enumerate(labels):
        dist = np.linalg.norm(detected[i] - gt[i])
        errors[label] = {
            'distance': dist,
            'detected': detected[i].tolist(),
            'gt': gt[i].tolist(),
            'dx': detected[i][0] - gt[i][0],
            'dy': detected[i][1] - gt[i][1],
        }

    return errors


def visualize_diagnostic(result: dict, img_name: str, save_path: Path = None):
    """Visualize pipeline stages and errors."""
    import matplotlib.pyplot as plt

    if 'error' in result:
        print(f"{img_name}: FAILED at {result['stage']} - {result['error']}")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Original with GT
    ax = axes[0, 0]
    ax.imshow(cv2.cvtColor(result['resized'], cv2.COLOR_BGR2RGB))
    gt_scaled = result['gt'] * result['scale']
    gt_closed = np.vstack([gt_scaled, gt_scaled[0]])
    ax.plot(gt_closed[:, 0], gt_closed[:, 1], 'g-', linewidth=2, label='GT')
    ax.scatter(gt_scaled[:, 0], gt_scaled[:, 1], c='green', s=100, marker='o')
    ax.set_title('Ground Truth')
    ax.legend()
    ax.axis('off')

    # 2. Edges
    ax = axes[0, 1]
    ax.imshow(result['edges'], cmap='gray')
    ax.set_title(f"Canny Edges ({len(result['lines'])} lines)")
    ax.axis('off')

    # 3. All lines classified
    ax = axes[0, 2]
    img = cv2.cvtColor(result['resized'].copy(), cv2.COLOR_BGR2RGB)
    for x1, y1, x2, y2 in result['h_lines']:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    for x1, y1, x2, y2 in result['v_lines']:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    ax.imshow(img)
    ax.set_title(f"Classified: H={len(result['h_lines'])}, V={len(result['v_lines'])}")
    ax.axis('off')

    # 4. Clustered lines
    ax = axes[1, 0]
    img = cv2.cvtColor(result['resized'].copy(), cv2.COLOR_BGR2RGB)
    for x1, y1, x2, y2 in result['h_clustered']:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    for x1, y1, x2, y2 in result['v_clustered']:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    ax.imshow(img)
    ax.set_title(f"Clustered: H={len(result['h_clustered'])}, V={len(result['v_clustered'])}")
    ax.axis('off')

    # 5. Intersections
    ax = axes[1, 1]
    img = cv2.cvtColor(result['resized'].copy(), cv2.COLOR_BGR2RGB)
    for x, y in result['intersections']:
        cv2.circle(img, (int(x), int(y)), 5, (255, 255, 0), -1)
    ax.imshow(img)
    ax.set_title(f"Intersections: {len(result['intersections'])}")
    ax.axis('off')

    # 6. Final comparison
    ax = axes[1, 2]
    img = cv2.cvtColor(result['resized'].copy(), cv2.COLOR_BGR2RGB)

    # GT in green
    gt_closed = np.vstack([gt_scaled, gt_scaled[0]])
    for i in range(4):
        cv2.line(img,
                tuple(gt_closed[i].astype(int)),
                tuple(gt_closed[i+1].astype(int)),
                (0, 255, 0), 2)

    # Detected in red
    det_scaled = result['detected'] * result['scale']
    det_closed = np.vstack([det_scaled, det_scaled[0]])
    for i in range(4):
        cv2.line(img,
                tuple(det_closed[i].astype(int)),
                tuple(det_closed[i+1].astype(int)),
                (255, 0, 0), 2)

    # Corner markers
    labels = ['TL', 'TR', 'BR', 'BL']
    for i in range(4):
        cv2.circle(img, tuple(gt_scaled[i].astype(int)), 8, (0, 255, 0), -1)
        cv2.circle(img, tuple(det_scaled[i].astype(int)), 8, (255, 0, 0), -1)

    ax.imshow(img)
    ax.set_title(f"IoU: {result['iou']:.4f}\nGreen=GT, Red=Detected")
    ax.axis('off')

    plt.suptitle(f"{img_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def main():
    print("Loading ground truth and images...")
    gt = load_ground_truth()
    images = get_sample_images()

    print(f"\nAnalyzing {len(images)} images...")
    print("=" * 60)

    # Test with various parameter combinations
    param_sets = [
        {'name': 'default', 'dbscan_eps': 20, 'dbscan_min_samples': 2, 'hough_threshold': 60},
        {'name': 'best_global', 'dbscan_eps': 24, 'dbscan_min_samples': 1, 'hough_threshold': 30, 'hough_min_length': 40, 'hough_max_gap': 5},
        {'name': 'tight_cluster', 'dbscan_eps': 10, 'dbscan_min_samples': 1, 'hough_threshold': 40},
        {'name': 'loose_cluster', 'dbscan_eps': 30, 'dbscan_min_samples': 1, 'hough_threshold': 50},
    ]

    output_dir = Path('diagnostic_output')
    output_dir.mkdir(exist_ok=True)

    for img_path in images:
        name = img_path.name
        if name not in gt:
            print(f"Skipping {name} (no GT)")
            continue

        print(f"\n{name}:")
        print("-" * 40)

        gt_corners = gt[name]

        best_iou = 0
        best_result = None
        best_params = None

        for params in param_sets:
            result = run_pipeline_diagnostic(img_path, gt_corners, params)

            if 'error' in result:
                print(f"  {params['name']}: FAILED - {result['error']}")
            else:
                iou = result['iou']
                print(f"  {params['name']}: IoU = {iou:.4f}")

                if iou > best_iou:
                    best_iou = iou
                    best_result = result
                    best_params = params

        if best_result:
            # Analyze corner errors
            errors = analyze_corner_errors(best_result['detected'], gt_corners)
            print(f"\n  Best IoU: {best_iou:.4f} ({best_params['name']})")
            print(f"  Corner errors (pixels):")
            for corner, err in errors.items():
                print(f"    {corner}: {err['distance']:.1f}px (dx={err['dx']:.1f}, dy={err['dy']:.1f})")

            # Save visualization
            visualize_diagnostic(best_result, name, output_dir / f"{name.replace('.jpeg', '')}_diagnostic.png")


if __name__ == '__main__':
    main()
