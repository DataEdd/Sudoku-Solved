#!/usr/bin/env python3
"""
Visualize Polar Hough Transform (cv2.HoughLines) on Sudoku images.

This script demonstrates the polar approach which works better on rotated grids.

Usage:
    ./venv/bin/python3 scripts/visualize_hough_polar.py
"""
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from app.core.hough_standard import (
    detect_grid_polar,
    detect_lines_polar,
    filter_similar_lines,
    draw_polar_lines,
    classify_lines_by_theta,
)
from app.core.preprocessing import preprocess_for_hough_polar_full


def main():
    # Find all images in Examples/aug
    examples_dir = project_root / "Examples" / "aug"

    if not examples_dir.exists():
        print(f"Examples directory not found: {examples_dir}")
        return

    images = list(examples_dir.glob("*.jpeg")) + list(examples_dir.glob("*.jpg"))

    if not images:
        print("No images found in Examples/aug")
        return

    # Select random image
    random_image = random.choice(images)
    print(f"Selected image: {random_image.name}")

    # Load image
    image = cv2.imread(str(random_image))
    if image is None:
        print(f"Failed to load image: {random_image}")
        return

    print(f"Image shape: {image.shape}")

    # ==========================================================================
    # PREPROCESSING WITH MORPHOLOGICAL OPERATIONS
    # ==========================================================================
    print("\n--- Preprocessing ---")
    preprocess_result = preprocess_for_hough_polar_full(image)
    print(f"Edges shape: {preprocess_result.edges.shape}")

    # ==========================================================================
    # POLAR HOUGH DETECTION
    # ==========================================================================
    print("\n--- Polar Hough Detection ---")

    # Detect all lines (no filtering)
    all_lines = detect_lines_polar(preprocess_result.eroded, threshold=150)
    n_all = len(all_lines) if all_lines is not None else 0
    print(f"Total lines detected: {n_all}")

    # Filter similar lines
    if all_lines is not None and len(all_lines) > 0:
        filtered_lines = filter_similar_lines(all_lines, rho_threshold=15, theta_threshold=0.1)
        print(f"After filtering: {len(filtered_lines)}")

        # Classify by theta
        h_lines, v_lines = classify_lines_by_theta(filtered_lines)
        print(f"Horizontal: {len(h_lines)}, Vertical: {len(v_lines)}")
    else:
        filtered_lines = []
        h_lines, v_lines = [], []

    # ==========================================================================
    # FULL PIPELINE
    # ==========================================================================
    print("\n--- Full Pipeline ---")
    result = detect_grid_polar(image, return_visualization=True)
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Filtered lines: {len(result.filtered_lines)}")

    # ==========================================================================
    # SAVE OUTPUTS
    # ==========================================================================
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    # Save preprocessing steps
    cv2.imwrite(str(output_dir / "polar_01_original.jpg"), image)
    cv2.imwrite(str(output_dir / "polar_02_grayscale.jpg"), preprocess_result.grayscale)
    cv2.imwrite(str(output_dir / "polar_03_canny.jpg"), preprocess_result.edges)
    cv2.imwrite(str(output_dir / "polar_04_dilated.jpg"), preprocess_result.dilated)
    cv2.imwrite(str(output_dir / "polar_05_eroded.jpg"), preprocess_result.eroded)

    # Save line detection steps
    if all_lines is not None and len(all_lines) > 0:
        # All lines (before filtering)
        img_all = draw_polar_lines(
            image,
            [tuple(line[0]) for line in all_lines],
            color=(0, 0, 255),
            thickness=1
        )
        cv2.imwrite(str(output_dir / "polar_06_all_lines.jpg"), img_all)

    # Filtered lines
    if filtered_lines:
        img_filtered = draw_polar_lines(image, filtered_lines, color=(0, 255, 0), thickness=2)
        cv2.imwrite(str(output_dir / "polar_07_filtered_lines.jpg"), img_filtered)

    # Classified lines (H=blue, V=green)
    if h_lines or v_lines:
        img_classified = image.copy()
        img_classified = draw_polar_lines(img_classified, h_lines, color=(255, 0, 0), thickness=2)
        img_classified = draw_polar_lines(img_classified, v_lines, color=(0, 255, 0), thickness=2)
        cv2.imwrite(str(output_dir / "polar_08_classified.jpg"), img_classified)

    # Final result
    if result.annotated_image is not None:
        cv2.imwrite(str(output_dir / "polar_09_result.jpg"), result.annotated_image)

    print(f"\nOutput files saved to: {output_dir}")
    print("Files:")
    print("  polar_01_original.jpg     - Original image")
    print("  polar_02_grayscale.jpg    - Grayscale")
    print("  polar_03_canny.jpg        - Canny edges")
    print("  polar_04_dilated.jpg      - After dilation (connects gaps)")
    print("  polar_05_eroded.jpg       - After erosion (cleans noise)")
    print("  polar_06_all_lines.jpg    - All detected lines (red)")
    print("  polar_07_filtered_lines.jpg - After similarity filtering (green)")
    print("  polar_08_classified.jpg   - H=blue, V=green")
    print("  polar_09_result.jpg       - Final result")

    # Open the result
    result_path = output_dir / "polar_09_result.jpg"
    if result_path.exists():
        import subprocess
        subprocess.run(["open", str(result_path)])

    return result_path


if __name__ == "__main__":
    main()
