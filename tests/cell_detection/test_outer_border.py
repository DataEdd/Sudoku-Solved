"""
Unit tests for outer border detection.

Tests the contour-based quadrilateral detection algorithm.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add project and notebook utils to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "cell_detection"))

from utils.geometry import (
    order_corners,
    compute_quad_area,
    is_valid_quadrilateral,
    point_distance,
)


class TestOrderCorners:
    """Tests for corner ordering function."""

    def test_orders_axis_aligned_rectangle(self):
        """Test ordering of axis-aligned rectangle corners."""
        # Points in random order
        pts = np.array([
            [200, 200],  # BR
            [100, 100],  # TL
            [200, 100],  # TR
            [100, 200],  # BL
        ], dtype=np.float32)

        ordered = order_corners(pts)

        # Verify order: TL, TR, BR, BL
        assert np.allclose(ordered[0], [100, 100]), "TL should be (100, 100)"
        assert np.allclose(ordered[1], [200, 100]), "TR should be (200, 100)"
        assert np.allclose(ordered[2], [200, 200]), "BR should be (200, 200)"
        assert np.allclose(ordered[3], [100, 200]), "BL should be (100, 200)"

    def test_orders_rotated_square(self):
        """Test ordering when square is rotated 45 degrees."""
        # Diamond shape (square rotated 45 degrees)
        pts = np.array([
            [150, 50],   # Top (becomes TL after ordering by sum)
            [250, 150],  # Right
            [150, 250],  # Bottom
            [50, 150],   # Left
        ], dtype=np.float32)

        ordered = order_corners(pts)

        # TL has min sum, BR has max sum
        assert ordered[0][0] + ordered[0][1] < ordered[2][0] + ordered[2][1]

    def test_handles_non_square_quadrilateral(self):
        """Test ordering of non-square quadrilateral."""
        pts = np.array([
            [50, 100],   # TL
            [300, 80],   # TR (slightly higher)
            [320, 280],  # BR
            [40, 300],   # BL
        ], dtype=np.float32)

        ordered = order_corners(pts)

        # Should still produce valid ordering
        assert len(ordered) == 4
        assert ordered.shape == (4, 2)

        # TL should have smallest sum
        sums = ordered.sum(axis=1)
        assert np.argmin(sums) == 0, "TL should have smallest sum"
        assert np.argmax(sums) == 2, "BR should have largest sum"

    def test_handles_reshape_from_contour(self):
        """Test ordering when input is contour format (N, 1, 2)."""
        pts = np.array([
            [[100, 100]],
            [[200, 100]],
            [[200, 200]],
            [[100, 200]],
        ], dtype=np.float32)

        ordered = order_corners(pts)

        assert ordered.shape == (4, 2)
        assert np.allclose(ordered[0], [100, 100])


class TestComputeQuadArea:
    """Tests for quadrilateral area computation."""

    def test_square_area(self):
        """Test area of a perfect square."""
        corners = np.array([
            [0, 0],
            [100, 0],
            [100, 100],
            [0, 100],
        ])

        area = compute_quad_area(corners)
        assert area == 10000.0

    def test_rectangle_area(self):
        """Test area of a rectangle."""
        corners = np.array([
            [0, 0],
            [200, 0],
            [200, 100],
            [0, 100],
        ])

        area = compute_quad_area(corners)
        assert area == 20000.0

    def test_parallelogram_area(self):
        """Test area of a parallelogram (non-rectangular)."""
        # Parallelogram with base 100, height 100, slant 50
        corners = np.array([
            [50, 0],
            [150, 0],
            [100, 100],
            [0, 100],
        ])

        area = compute_quad_area(corners)
        # Area should be base * height = 100 * 100 = 10000
        assert area == 10000.0

    def test_zero_area_degenerate(self):
        """Test degenerate case with collinear points."""
        corners = np.array([
            [0, 0],
            [50, 0],
            [100, 0],
            [150, 0],
        ])

        area = compute_quad_area(corners)
        assert area == 0.0


class TestIsValidQuadrilateral:
    """Tests for quadrilateral validation."""

    def test_valid_large_quad(self):
        """Test that a reasonably-sized quad is valid."""
        corners = np.array([
            [100, 100],
            [400, 100],
            [400, 400],
            [100, 400],
        ])
        image_shape = (500, 500)

        assert is_valid_quadrilateral(corners, image_shape)

    def test_rejects_too_small_quad(self):
        """Test that very small quads are rejected."""
        corners = np.array([
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
        ])
        image_shape = (500, 500)

        # Area = 100, image area = 250000
        # Ratio = 0.0004 < 0.05 threshold
        assert not is_valid_quadrilateral(corners, image_shape)

    def test_rejects_too_large_quad(self):
        """Test that quads covering almost entire image are rejected."""
        corners = np.array([
            [0, 0],
            [499, 0],
            [499, 499],
            [0, 499],
        ])
        image_shape = (500, 500)

        # Area ≈ 249000, ratio ≈ 0.996 > 0.95 threshold
        assert not is_valid_quadrilateral(corners, image_shape)

    def test_rejects_extreme_aspect_ratio(self):
        """Test that very narrow quads are rejected."""
        corners = np.array([
            [0, 0],
            [400, 0],
            [400, 50],
            [0, 50],
        ])
        image_shape = (500, 500)

        # Aspect ratio = 400/50 = 8.0 > 2.0 threshold
        assert not is_valid_quadrilateral(corners, image_shape)

    def test_custom_thresholds(self):
        """Test with custom validation thresholds."""
        corners = np.array([
            [0, 0],
            [100, 0],
            [100, 100],
            [0, 100],
        ])
        image_shape = (500, 500)

        # With default thresholds, this would be too small (4% area)
        assert not is_valid_quadrilateral(corners, image_shape)

        # With lower threshold, it should pass
        assert is_valid_quadrilateral(
            corners, image_shape, min_area_ratio=0.01
        )


class TestPointDistance:
    """Tests for point distance calculation."""

    def test_horizontal_distance(self):
        """Test distance between horizontal points."""
        p1 = np.array([0, 0])
        p2 = np.array([100, 0])

        assert point_distance(p1, p2) == 100.0

    def test_vertical_distance(self):
        """Test distance between vertical points."""
        p1 = np.array([0, 0])
        p2 = np.array([0, 100])

        assert point_distance(p1, p2) == 100.0

    def test_diagonal_distance(self):
        """Test distance with Pythagorean calculation."""
        p1 = np.array([0, 0])
        p2 = np.array([3, 4])

        assert point_distance(p1, p2) == 5.0

    def test_same_point(self):
        """Test distance between same point is zero."""
        p1 = np.array([50, 50])
        p2 = np.array([50, 50])

        assert point_distance(p1, p2) == 0.0


class TestOuterBorderDetection:
    """Integration tests for the full border detection pipeline."""

    def test_detects_border_on_synthetic_grid(self, synthetic_grid_image):
        """Test detection on a clean synthetic grid."""
        image, expected_corners = synthetic_grid_image(size=450, rotation=0)

        # Import the detection functions from notebook utils
        # (These would be extracted to a module in practice)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find largest quadrilateral
        found_quad = None
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                found_quad = approx
                break

        assert found_quad is not None, "Should find a quadrilateral"

        # Order corners
        ordered = order_corners(found_quad.reshape(4, 2))

        # Check corners are close to expected (within 10 pixels)
        for i in range(4):
            dist = point_distance(ordered[i], expected_corners[i])
            assert dist < 10, f"Corner {i} should be within 10px of expected"

    def test_detects_border_on_rotated_grid(self, synthetic_grid_image):
        """Test detection on a rotated synthetic grid."""
        image, expected_corners = synthetic_grid_image(size=450, rotation=15)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find largest quadrilateral
        found_quad = None
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                found_quad = approx
                break

        assert found_quad is not None, "Should find a quadrilateral on rotated grid"

    def test_handles_noisy_image(self, synthetic_grid_image):
        """Test detection on a noisy synthetic grid."""
        image, _ = synthetic_grid_image(size=450, add_noise=True)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Should still find contours
        assert len(contours) > 0, "Should find contours even with noise"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_image(self):
        """Test handling of completely black image."""
        image = np.zeros((100, 100), dtype=np.uint8)
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        assert len(contours) == 0

    def test_white_image(self):
        """Test handling of completely white image.

        Note: A completely white image creates a single contour around
        the entire image boundary (the whole image is a white blob).
        """
        image = np.full((100, 100), 255, dtype=np.uint8)
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # White image creates one contour (the image boundary)
        assert len(contours) == 1
        # This contour covers the entire image
        area = cv2.contourArea(contours[0])
        assert area > 9000  # Close to 100*100

    def test_single_point(self):
        """Test order_corners with coincident points doesn't crash."""
        pts = np.array([
            [100, 100],
            [100, 100],
            [100, 100],
            [100, 100],
        ], dtype=np.float32)

        # Should not crash
        ordered = order_corners(pts)
        assert ordered.shape == (4, 2)
