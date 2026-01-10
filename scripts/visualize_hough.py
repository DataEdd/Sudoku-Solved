#!/usr/bin/env python3
"""
Quick script to visualize Hough line detection on a sample image.
Selects a random image from Examples/aug and shows detected grid lines.
"""
import os
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from app.core.hough_standard import detect_grid_standard, detect_lines, classify_lines, draw_lines_on_image
from app.core.preprocessing import preprocess_for_hough


def main():
    # Find all images in Examples/aug
    examples_dir = project_root / "Examples" / "aug"

    if not examples_dir.exists():
        print(f"Examples directory not found: {examples_dir}")
        return

    images = list(examples_dir.glob("*.jpeg")) + list(examples_dir.glob("*.jpg")) + list(examples_dir.glob("*.png"))

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

    # Preprocess
    thresh = preprocess_for_hough(image)

    # Detect all lines
    all_lines = detect_lines(thresh, threshold=80, min_line_length=30, max_line_gap=15)
    print(f"Total lines detected: {len(all_lines)}")

    # Classify lines
    h_lines, v_lines = classify_lines(all_lines)
    print(f"Horizontal lines: {len(h_lines)}, Vertical lines: {len(v_lines)}")

    # Draw all detected lines (red)
    img_all_lines = draw_lines_on_image(image, all_lines, color=(0, 0, 255), thickness=2)

    # Draw classified lines (blue for H, green for V)
    img_classified = image.copy()
    img_classified = draw_lines_on_image(img_classified, h_lines, color=(255, 0, 0), thickness=2)
    img_classified = draw_lines_on_image(img_classified, v_lines, color=(0, 255, 0), thickness=2)

    # Run full detection pipeline
    result = detect_grid_standard(image, return_visualization=True)
    print(f"Grid detection confidence: {result.confidence:.3f}")
    print(f"H positions: {len(result.h_positions)}, V positions: {len(result.v_positions)}")

    # Create output directory
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    # Save results
    original_output = output_dir / "01_original.jpg"
    thresh_output = output_dir / "02_threshold.jpg"
    all_lines_output = output_dir / "03_all_lines.jpg"
    classified_output = output_dir / "04_classified_lines.jpg"
    grid_output = output_dir / "05_detected_grid.jpg"

    cv2.imwrite(str(original_output), image)
    cv2.imwrite(str(thresh_output), thresh)
    cv2.imwrite(str(all_lines_output), img_all_lines)
    cv2.imwrite(str(classified_output), img_classified)

    if result.annotated_image is not None:
        cv2.imwrite(str(grid_output), result.annotated_image)
        print(f"\nSaved annotated image to: {grid_output}")

    print(f"\nOutput files saved to: {output_dir}")
    print(f"- Original: {original_output.name}")
    print(f"- Threshold: {thresh_output.name}")
    print(f"- All lines (red): {all_lines_output.name}")
    print(f"- Classified (H=blue, V=green): {classified_output.name}")
    print(f"- Detected grid: {grid_output.name}")

    # Return path to main annotated image
    return grid_output if result.annotated_image is not None else classified_output


if __name__ == "__main__":
    output_path = main()
    if output_path:
        print(f"\nMain output: {output_path}")
