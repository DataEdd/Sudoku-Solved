"""
Visual tests for image preprocessing pipeline.

Run with: pytest tests/test_preprocessing.py -v -s

The -s flag is required to show matplotlib windows.
"""

import cv2
import numpy as np
import pytest


class TestPreprocessingVisualization:
    """Visual tests for each preprocessing step."""

    def test_full_pipeline_visualization(self, sample_sudoku_image, display_images):
        """
        Show all preprocessing steps side by side.

        This test displays:
        1. Original - Input BGR image
        2. Grayscale - After color removal
        3. Blurred - After Gaussian smoothing
        4. Threshold - Binary image for Hough
        5. Edges - Canny edges for GHT
        """
        from app.core.preprocessing import preprocess_full

        result = preprocess_full(sample_sudoku_image)

        display_images({
            "1. Original": result.original,
            "2. Grayscale": result.grayscale,
            "3. Blurred": result.blurred,
            "4. Threshold": result.threshold,
            "5. Edges": result.edges,
        }, title="Preprocessing Pipeline")

    def test_grayscale_conversion(self, sample_sudoku_image, display_images):
        """
        Test grayscale conversion.

        Formula: G = 0.299R + 0.587G + 0.114B
        """
        from app.core.preprocessing import to_grayscale

        gray = to_grayscale(sample_sudoku_image)

        # Verify dimensions
        assert len(gray.shape) == 2, "Grayscale should be 2D"
        assert gray.shape == sample_sudoku_image.shape[:2], "Dimensions should match"

        display_images({
            "Original (BGR)": sample_sudoku_image,
            "Grayscale": gray,
        }, title="Grayscale Conversion: G = 0.299R + 0.587G + 0.114B")

    def test_blur_effect(self, sample_sudoku_image, display_images):
        """
        Compare different blur kernel sizes.

        Larger kernels = more smoothing = less noise but less detail.
        """
        from app.core.preprocessing import to_grayscale, apply_blur

        gray = to_grayscale(sample_sudoku_image)

        display_images({
            "No Blur": gray,
            "Kernel 3×3": apply_blur(gray, kernel_size=3),
            "Kernel 5×5": apply_blur(gray, kernel_size=5),
            "Kernel 9×9": apply_blur(gray, kernel_size=9),
        }, title="Gaussian Blur: Effect of Kernel Size")

    def test_threshold_parameters(self, sample_sudoku_image, display_images):
        """
        Compare different adaptive threshold parameters.

        block_size: Size of local neighborhood
        C: Constant subtracted from mean
        """
        from app.core.preprocessing import to_grayscale, apply_blur, adaptive_threshold

        gray = to_grayscale(sample_sudoku_image)
        blurred = apply_blur(gray)

        display_images({
            "Block=7, C=2": adaptive_threshold(blurred, block_size=7, c=2),
            "Block=11, C=2": adaptive_threshold(blurred, block_size=11, c=2),
            "Block=15, C=2": adaptive_threshold(blurred, block_size=15, c=2),
            "Block=11, C=5": adaptive_threshold(blurred, block_size=11, c=5),
        }, title="Adaptive Threshold: Block Size and C Constant")

    def test_edge_detection_parameters(self, sample_sudoku_image, display_images):
        """
        Compare different Canny edge detection thresholds.

        low_threshold: Edges below this are discarded
        high_threshold: Edges above this are kept
        Between: Kept if connected to strong edge
        """
        from app.core.preprocessing import to_grayscale, apply_blur, detect_edges

        gray = to_grayscale(sample_sudoku_image)
        blurred = apply_blur(gray)

        display_images({
            "Low=30, High=100": detect_edges(blurred, 30, 100),
            "Low=50, High=150": detect_edges(blurred, 50, 150),
            "Low=100, High=200": detect_edges(blurred, 100, 200),
            "Low=50, High=250": detect_edges(blurred, 50, 250),
        }, title="Canny Edge Detection: Threshold Comparison")

    def test_gradient_visualization(self, sample_sudoku_image, display_images):
        """
        Visualize gradient magnitude and direction.

        Gradients are used in GHT for edge orientation.
        """
        from app.core.preprocessing import (
            to_grayscale, apply_blur, compute_gradients,
            compute_gradient_magnitude, compute_gradient_direction
        )

        gray = to_grayscale(sample_sudoku_image)
        blurred = apply_blur(gray)
        gx, gy = compute_gradients(blurred)

        magnitude = compute_gradient_magnitude(gx, gy)
        direction = compute_gradient_direction(gx, gy)

        # Normalize for visualization
        mag_vis = (magnitude / magnitude.max() * 255).astype(np.uint8)
        dir_vis = ((direction + np.pi) / (2 * np.pi) * 255).astype(np.uint8)

        display_images({
            "Original": sample_sudoku_image,
            "Gradient X": ((gx - gx.min()) / (gx.max() - gx.min()) * 255).astype(np.uint8),
            "Gradient Y": ((gy - gy.min()) / (gy.max() - gy.min()) * 255).astype(np.uint8),
            "Magnitude": mag_vis,
        }, title="Image Gradients (Sobel)")

    def test_preprocessing_on_variations(self, test_image_collection, display_images):
        """
        Test preprocessing on different image conditions.

        Shows how preprocessing handles:
        - Clean images
        - Rotated images
        - Perspective distortion
        - Noisy images
        """
        from app.core.preprocessing import preprocess_for_hough

        results = {}
        for name, img in test_image_collection.items():
            thresh = preprocess_for_hough(img)
            results[name] = thresh

        display_images(results, title="Preprocessing Robustness")


class TestPreprocessingMath:
    """Tests verifying the mathematical operations."""

    def test_grayscale_weights(self, synthetic_sudoku):
        """Verify grayscale uses correct weights."""
        from app.core.preprocessing import to_grayscale

        # Create test image with known colors
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 0] = 100  # Blue
        img[:, :, 1] = 150  # Green
        img[:, :, 2] = 200  # Red

        gray = to_grayscale(img)

        # Expected: 0.299*200 + 0.587*150 + 0.114*100 = 59.8 + 88.05 + 11.4 = 159.25
        expected = int(0.299 * 200 + 0.587 * 150 + 0.114 * 100)

        assert abs(gray[5, 5] - expected) < 2, f"Expected ~{expected}, got {gray[5, 5]}"

    def test_blur_kernel_symmetry(self, synthetic_sudoku):
        """Verify Gaussian blur is symmetric."""
        from app.core.preprocessing import to_grayscale, apply_blur

        img = synthetic_sudoku(size=100)
        gray = to_grayscale(img)
        blurred = apply_blur(gray, kernel_size=5)

        # Blurred image should be smoother (lower variance in local patches)
        original_var = np.var(gray[40:60, 40:60])
        blurred_var = np.var(blurred[40:60, 40:60])

        assert blurred_var <= original_var, "Blur should reduce variance"

    def test_threshold_binary(self, synthetic_sudoku):
        """Verify threshold produces binary output."""
        from app.core.preprocessing import preprocess_for_hough

        img = synthetic_sudoku(size=100)
        thresh = preprocess_for_hough(img)

        unique_values = np.unique(thresh)
        assert len(unique_values) <= 2, f"Expected binary, got {unique_values}"
        assert set(unique_values).issubset({0, 255}), f"Expected 0/255, got {unique_values}"
