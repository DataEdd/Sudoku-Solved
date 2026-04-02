#!/usr/bin/env python3
"""Visualize each step of the detection pipeline for debugging."""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DetectionConfig
from src.contour_detection import detect_contour_path, find_quadrilateral_contours
from src.debug import draw_corners
from src.detector import SudokuDetector
from src.hough_detection import (
    classify_lines,
    cluster_lines,
    detect_hough_path,
    detect_lines,
)
from src.preprocessing import (
    adaptive_threshold,
    apply_clahe,
    apply_morphological,
    get_edges,
    load_and_validate,
    resize_image,
    to_grayscale,
)


def visualize_pipeline(image_path: str, output_dir: str = "pipeline_debug"):
    """Visualize each step of the detection pipeline for a single image.

    Useful for debugging why specific images fail.

    Args:
        image_path: Path to the input image.
        output_dir: Directory to save debug images.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = DetectionConfig()

    # Step 1: Load
    print("Step 1: Loading image...")
    image = load_and_validate(image_path)
    if image is None:
        print("Failed to load image!")
        return
    cv2.imwrite(str(output_path / "01_original.jpg"), image)
    print(f"  Original size: {image.shape}")

    # Step 2: Resize
    print("Step 2: Resizing...")
    resized, scale = resize_image(image, config.target_size)
    cv2.imwrite(str(output_path / "02_resized.jpg"), resized)
    print(f"  Resized: {resized.shape}, scale factor: {scale:.3f}")

    # Step 3: Grayscale
    print("Step 3: Converting to grayscale...")
    gray = to_grayscale(resized)
    cv2.imwrite(str(output_path / "03_grayscale.jpg"), gray)

    # Step 4: CLAHE
    print("Step 4: Applying CLAHE...")
    enhanced = apply_clahe(gray, config.clahe_clip_limit, config.clahe_tile_size)
    cv2.imwrite(str(output_path / "04_clahe.jpg"), enhanced)

    # Step 5a: Adaptive threshold (contour path)
    print("Step 5a: Adaptive threshold (contour path)...")
    binary = adaptive_threshold(enhanced, config.adaptive_block_size, config.adaptive_c)
    cv2.imwrite(str(output_path / "05a_binary.jpg"), binary)

    # Step 5b: Optional morphological
    if config.use_morphological:
        print("Step 5b: Morphological closing...")
        binary_morph = apply_morphological(binary, config.morph_kernel_size)
        cv2.imwrite(str(output_path / "05b_morphological.jpg"), binary_morph)
    else:
        binary_morph = binary
        print("Step 5b: Morphological closing SKIPPED (disabled in config)")

    # Step 6: Edges (Hough path)
    print("Step 6: Canny edges (Hough path)...")
    edges = get_edges(enhanced, config.canny_low, config.canny_high, blur_size=5)
    cv2.imwrite(str(output_path / "06_edges.jpg"), edges)

    # Step 7a: Contour detection
    print("Step 7a: Contour detection...")
    contour_candidates = detect_contour_path(binary_morph, config)
    print(f"  Found {len(contour_candidates)} contour candidates")

    if contour_candidates:
        contour_vis = resized.copy()
        for i, cand in enumerate(contour_candidates):
            # Green for best, orange for others
            color = (0, 255, 0) if i == 0 else (0, 165, 255)
            contour_vis = draw_corners(contour_vis, cand["corners"], color=color)
        cv2.imwrite(str(output_path / "07a_contour_candidates.jpg"), contour_vis)

    # Step 7b: Hough detection
    print("Step 7b: Hough line detection...")
    hough_candidates = detect_hough_path(
        edges, config, (resized.shape[0], resized.shape[1])
    )
    print(f"  Found {len(hough_candidates)} Hough candidates")

    # Visualize detected lines
    lines = detect_lines(edges, config)
    if lines is not None:
        lines_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        h_lines, v_lines = classify_lines(lines)

        # Draw horizontal lines in red
        for line in h_lines:
            cv2.line(
                lines_vis,
                (line[0], line[1]),
                (line[2], line[3]),
                (0, 0, 255),
                1,
            )

        # Draw vertical lines in blue
        for line in v_lines:
            cv2.line(
                lines_vis,
                (line[0], line[1]),
                (line[2], line[3]),
                (255, 0, 0),
                1,
            )

        cv2.imwrite(str(output_path / "07b_hough_lines.jpg"), lines_vis)
        print(f"  Horizontal lines: {len(h_lines)}, Vertical lines: {len(v_lines)}")

        # Also show clustered lines
        if h_lines and v_lines:
            # Use same threshold as hough_detection.py: image_dim / 36
            h_threshold = max(1, resized.shape[0] / 36)  # height for horizontal
            v_threshold = max(1, resized.shape[1] / 36)  # width for vertical

            h_clustered = cluster_lines(h_lines, h_threshold, is_horizontal=True)
            v_clustered = cluster_lines(v_lines, v_threshold, is_horizontal=False)

            clustered_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            for line in h_clustered:
                cv2.line(
                    clustered_vis,
                    (line[0], line[1]),
                    (line[2], line[3]),
                    (0, 0, 255),
                    2,
                )
            for line in v_clustered:
                cv2.line(
                    clustered_vis,
                    (line[0], line[1]),
                    (line[2], line[3]),
                    (255, 0, 0),
                    2,
                )
            cv2.imwrite(str(output_path / "07b_clustered_lines.jpg"), clustered_vis)
            print(
                f"  After clustering: {len(h_clustered)} horizontal, "
                f"{len(v_clustered)} vertical"
            )
    else:
        print("  No lines detected")

    if hough_candidates:
        hough_vis = resized.copy()
        for i, cand in enumerate(hough_candidates):
            color = (255, 0, 0) if i == 0 else (255, 165, 0)
            hough_vis = draw_corners(hough_vis, cand["corners"], color=color)
        cv2.imwrite(str(output_path / "07b_hough_candidates.jpg"), hough_vis)

    # Step 8: Final detection
    print("Step 8: Running full detector...")
    detector = SudokuDetector(config)
    result = detector.detect(image)

    if result.success:
        print(
            f"  SUCCESS! Method: {result.detection_method}, "
            f"Confidence: {result.confidence:.3f}"
        )
        cv2.imwrite(str(output_path / "08_result_warped.jpg"), result.warped_image)

        final_vis = image.copy()
        final_vis = draw_corners(final_vis, result.corners)
        cv2.imwrite(str(output_path / "08_result_annotated.jpg"), final_vis)
    else:
        print(f"  FAILED: {result.error_message}")

    print(f"\nAll debug images saved to {output_path}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_pipeline.py <image_path> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "pipeline_debug"

    visualize_pipeline(image_path, output_dir)
