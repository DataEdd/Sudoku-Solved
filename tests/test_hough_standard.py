"""
Visual tests for Standard Hough Transform line detection.

Run with: pytest tests/test_hough_standard.py -v -s
"""

import cv2
import numpy as np
import pytest


class TestStandardHoughVisualization:
    """Visual tests for standard Hough line detection."""

    def test_line_detection_all(self, sample_sudoku_image, display_images):
        """
        Show all detected lines overlaid on image.

        Red lines = all lines detected by HoughLinesP
        """
        from app.core.preprocessing import preprocess_for_hough
        from app.core.hough_standard import detect_lines, draw_lines_on_image

        thresh = preprocess_for_hough(sample_sudoku_image)
        lines = detect_lines(thresh)

        # Draw all lines in red
        img_lines = draw_lines_on_image(
            sample_sudoku_image, lines,
            color=(0, 0, 255), thickness=2
        )

        display_images({
            "Original": sample_sudoku_image,
            "Threshold": thresh,
            f"Detected Lines ({len(lines)})": img_lines,
        }, title="Standard Hough: All Detected Lines")

    def test_line_classification(self, sample_sudoku_image, display_images):
        """
        Show horizontal (blue) vs vertical (green) line classification.

        Classification is based on line angle:
        - Horizontal: |angle| < 10°
        - Vertical: ||angle| - 90°| < 10°
        """
        from app.core.preprocessing import preprocess_for_hough
        from app.core.hough_standard import (
            detect_lines, classify_lines, draw_lines_on_image
        )

        thresh = preprocess_for_hough(sample_sudoku_image)
        lines = detect_lines(thresh)
        h_lines, v_lines = classify_lines(lines)

        # Draw classified lines
        img_classified = sample_sudoku_image.copy()
        img_classified = draw_lines_on_image(
            img_classified, h_lines,
            color=(255, 0, 0), thickness=2  # Blue for horizontal
        )
        img_classified = draw_lines_on_image(
            img_classified, v_lines,
            color=(0, 255, 0), thickness=2  # Green for vertical
        )

        display_images({
            "Original": sample_sudoku_image,
            f"H={len(h_lines)} (blue), V={len(v_lines)} (green)": img_classified,
        }, title="Line Classification: Horizontal vs Vertical")

    def test_line_clustering(self, sample_sudoku_image, display_images):
        """
        Show clustered lines vs raw detected lines.

        Multiple lines may be detected for thick grid lines.
        Clustering merges nearby lines.
        """
        from app.core.preprocessing import preprocess_for_hough
        from app.core.hough_standard import (
            detect_lines, classify_lines, cluster_lines,
            draw_lines_on_image, draw_grid_overlay
        )

        thresh = preprocess_for_hough(sample_sudoku_image)
        lines = detect_lines(thresh)
        h_lines, v_lines = classify_lines(lines)

        h_clustered = cluster_lines(h_lines, is_horizontal=True)
        v_clustered = cluster_lines(v_lines, is_horizontal=False)

        # Draw raw lines
        img_raw = draw_lines_on_image(sample_sudoku_image, h_lines, (255, 0, 0), 1)
        img_raw = draw_lines_on_image(img_raw, v_lines, (0, 255, 0), 1)

        # Draw clustered lines
        img_clustered = draw_grid_overlay(
            sample_sudoku_image, h_clustered, v_clustered,
            thickness=2
        )

        display_images({
            f"Raw ({len(h_lines)}H, {len(v_lines)}V)": img_raw,
            f"Clustered ({len(h_clustered)}H, {len(v_clustered)}V)": img_clustered,
        }, title="Line Clustering: Before and After")

    def test_grid_detection_full(self, sample_sudoku_image, display_images):
        """
        Show complete grid detection with intersection points.

        Yellow dots = intersection points (should be 10×10 = 100)
        """
        from app.core.hough_standard import detect_grid_standard

        result = detect_grid_standard(sample_sudoku_image, return_visualization=True)

        info_text = (
            f"H: {len(result.h_positions)}, "
            f"V: {len(result.v_positions)}, "
            f"Conf: {result.confidence:.2f}"
        )

        display_images({
            "Original": sample_sudoku_image,
            f"Grid Detection ({info_text})": result.annotated_image,
        }, title="Standard Hough: Complete Grid Detection")

    def test_intersection_accuracy(self, sample_sudoku_image, display_images):
        """
        Visualize intersection points with zoomed view.

        Shows how accurately intersections align with actual grid.
        """
        from app.core.hough_standard import detect_grid_standard, draw_intersections

        result = detect_grid_standard(sample_sudoku_image)

        if len(result.intersections) > 0:
            # Draw intersections with larger circles
            img_intersections = draw_intersections(
                sample_sudoku_image,
                result.intersections,
                color=(0, 255, 255),
                radius=5
            )

            # Zoom to center region
            h, w = sample_sudoku_image.shape[:2]
            margin = h // 4
            zoomed = img_intersections[margin:h-margin, margin:w-margin]
            zoomed = cv2.resize(zoomed, (h, w))

            display_images({
                "Full Grid": img_intersections,
                "Center Zoomed": zoomed,
            }, title="Intersection Point Accuracy")

    def test_detection_on_variations(self, test_image_collection, display_images):
        """
        Test grid detection on various image conditions.

        Shows robustness to rotation, perspective, noise.
        """
        from app.core.hough_standard import detect_grid_standard

        results = {}
        for name, img in test_image_collection.items():
            result = detect_grid_standard(img, return_visualization=True)
            if result.annotated_image is not None:
                results[f"{name} (conf={result.confidence:.2f})"] = result.annotated_image
            else:
                results[f"{name} (FAILED)"] = img

        display_images(results, title="Standard Hough: Robustness Test")


class TestStandardHoughMath:
    """Tests verifying the mathematical operations."""

    def test_line_angle_calculation(self):
        """Test that line angles are computed correctly."""
        from app.core.hough_standard import compute_line_angle

        # Horizontal line: angle should be ~0°
        h_line = np.array([0, 100, 200, 100])
        assert abs(compute_line_angle(h_line)) < 1

        # Vertical line: angle should be ~90°
        v_line = np.array([100, 0, 100, 200])
        assert abs(abs(compute_line_angle(v_line)) - 90) < 1

        # 45° line
        d_line = np.array([0, 0, 100, 100])
        assert abs(compute_line_angle(d_line) - 45) < 1

    def test_classification_threshold(self):
        """Test line classification with different angles."""
        from app.core.hough_standard import classify_lines

        lines = np.array([
            [0, 100, 200, 100],    # Horizontal (0°)
            [0, 100, 200, 110],    # Nearly horizontal (small angle)
            [100, 0, 100, 200],    # Vertical (90°)
            [0, 0, 100, 100],      # Diagonal (45°) - should be discarded
        ])

        h, v = classify_lines(lines, angle_threshold=10)

        assert len(h) == 2, "Should have 2 horizontal lines"
        assert len(v) == 1, "Should have 1 vertical line"

    def test_clustering_merges_nearby(self):
        """Test that clustering merges nearby lines."""
        from app.core.hough_standard import cluster_lines

        # Two lines at y=100 and y=105 should merge
        lines = np.array([
            [0, 100, 200, 100],
            [0, 105, 200, 105],
        ])

        positions = cluster_lines(lines, is_horizontal=True, distance_threshold=20)

        assert len(positions) == 1, "Should merge into 1 line"
        assert 100 <= positions[0] <= 105, "Merged position should be between originals"

    def test_interpolation_fills_gaps(self):
        """Test that interpolation fills missing lines."""
        from app.core.hough_standard import interpolate_grid_lines

        # Only 5 lines detected, need 10
        positions = [0, 100, 200, 300, 400]

        interpolated = interpolate_grid_lines(positions, target_count=10)

        assert len(interpolated) == 10, "Should have 10 lines"

        # Should be roughly evenly spaced
        spacings = np.diff(interpolated)
        assert np.std(spacings) < np.mean(spacings) * 0.1, "Should be evenly spaced"

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        from app.core.hough_standard import compute_confidence

        # Perfectly regular grid should have high confidence
        regular_h = list(range(0, 1000, 100))  # 10 lines, 100px apart
        regular_v = list(range(0, 1000, 100))

        confidence = compute_confidence(regular_h, regular_v)
        assert confidence > 0.9, f"Regular grid should have high confidence, got {confidence}"

        # Irregular grid should have lower confidence
        irregular_h = [0, 80, 200, 280, 400, 520, 600, 700, 800, 950]
        irregular_v = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

        confidence_irregular = compute_confidence(irregular_h, irregular_v)
        assert confidence_irregular < confidence, "Irregular should have lower confidence"
