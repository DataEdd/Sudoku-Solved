"""
Unit tests for cell edge detection.

Tests ROI extraction, Canny edge detection, Hough lines, and line classification.
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

from utils.geometry import line_angle, line_intersection, classify_line_orientation


class TestROIExtraction:
    """Tests for Region of Interest extraction."""

    def test_roi_contains_bbox(self):
        """Test that ROI contains the original bounding box."""
        image = np.zeros((450, 450), dtype=np.uint8)
        bbox = (100, 100, 50, 50)  # x, y, w, h
        margin_ratio = 0.3

        x, y, w, h = bbox
        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)

        roi = image[y1:y2, x1:x2]

        # ROI should be larger than original bbox
        assert roi.shape[0] >= h
        assert roi.shape[1] >= w

    def test_roi_handles_edge_bbox(self):
        """Test ROI extraction when bbox is near image edge."""
        image = np.zeros((100, 100), dtype=np.uint8)
        bbox = (0, 0, 30, 30)  # At corner
        margin_ratio = 0.5

        x, y, w, h = bbox
        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)

        # Should be clamped to 0
        assert x1 == 0
        assert y1 == 0

    def test_roi_offset_tracking(self):
        """Test that ROI offset is correctly computed."""
        bbox = (50, 60, 40, 40)
        margin_ratio = 0.2

        x, y, w, h = bbox
        margin_x = int(w * margin_ratio)  # 8
        margin_y = int(h * margin_ratio)  # 8

        roi_x = max(0, x - margin_x)
        roi_y = max(0, y - margin_y)

        assert roi_x == 42  # 50 - 8
        assert roi_y == 52  # 60 - 8


class TestCannyEdgeDetection:
    """Tests for Canny edge detection."""

    def test_detects_edges_on_simple_shape(self):
        """Test edge detection on a simple rectangle."""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 80), 255, 2)

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Should detect edges
        assert edges.sum() > 0

    def test_no_edges_on_uniform_image(self):
        """Test that uniform image produces no edges."""
        image = np.full((100, 100), 128, dtype=np.uint8)

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Should have no edges
        assert edges.sum() == 0

    def test_threshold_sensitivity(self):
        """Test that higher thresholds produce fewer edges."""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 80), 200, 1)  # Light gray rectangle

        edges_low = cv2.Canny(image, 30, 90)
        edges_high = cv2.Canny(image, 100, 200)

        # Lower threshold should produce more edge pixels
        assert edges_low.sum() >= edges_high.sum()


class TestHoughLines:
    """Tests for Hough line detection."""

    def test_detects_horizontal_line(self):
        """Test detection of a horizontal line."""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(image, (10, 50), (90, 50), 255, 2)

        lines = cv2.HoughLinesP(
            image, 1, np.pi/180, threshold=30,
            minLineLength=30, maxLineGap=5
        )

        assert lines is not None
        assert len(lines) > 0

    def test_detects_vertical_line(self):
        """Test detection of a vertical line."""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(image, (50, 10), (50, 90), 255, 2)

        lines = cv2.HoughLinesP(
            image, 1, np.pi/180, threshold=30,
            minLineLength=30, maxLineGap=5
        )

        assert lines is not None
        assert len(lines) > 0

    def test_min_line_length_filter(self):
        """Test that short lines are filtered by minLineLength."""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(image, (10, 50), (20, 50), 255, 2)  # Short line (10px)

        lines = cv2.HoughLinesP(
            image, 1, np.pi/180, threshold=5,
            minLineLength=50,  # Longer than the line
            maxLineGap=5
        )

        # Should not detect the short line
        assert lines is None or len(lines) == 0

    def test_detects_multiple_lines(self):
        """Test detection of multiple lines."""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(image, (10, 30), (90, 30), 255, 2)  # Horizontal
        cv2.line(image, (30, 10), (30, 90), 255, 2)  # Vertical

        lines = cv2.HoughLinesP(
            image, 1, np.pi/180, threshold=30,
            minLineLength=30, maxLineGap=5
        )

        assert lines is not None
        assert len(lines) >= 2


class TestLineAngle:
    """Tests for line angle calculation."""

    def test_horizontal_line_angle(self):
        """Test angle of horizontal line is ~0 or ~180."""
        line = np.array([0, 50, 100, 50])
        angle = line_angle(line)

        assert angle < 5 or angle > 175

    def test_vertical_line_angle(self):
        """Test angle of vertical line is ~90."""
        line = np.array([50, 0, 50, 100])
        angle = line_angle(line)

        assert 85 < angle < 95

    def test_diagonal_line_angle(self):
        """Test angle of 45-degree diagonal."""
        line = np.array([0, 0, 100, 100])
        angle = line_angle(line)

        assert 40 < angle < 50

    def test_handles_zero_length_line(self):
        """Test handling of zero-length line (single point)."""
        line = np.array([50, 50, 50, 50])
        angle = line_angle(line)

        # Should return some angle without crashing
        assert 0 <= angle <= 180


class TestLineClassification:
    """Tests for line classification as horizontal/vertical/diagonal."""

    def test_classifies_horizontal(self):
        """Test classification of horizontal line."""
        line = np.array([0, 50, 100, 50])
        result = classify_line_orientation(line, angle_threshold=15.0)

        assert result == "horizontal"

    def test_classifies_vertical(self):
        """Test classification of vertical line."""
        line = np.array([50, 0, 50, 100])
        result = classify_line_orientation(line, angle_threshold=15.0)

        assert result == "vertical"

    def test_classifies_diagonal(self):
        """Test classification of diagonal line."""
        line = np.array([0, 0, 100, 100])
        result = classify_line_orientation(line, angle_threshold=15.0)

        assert result == "diagonal"

    def test_near_horizontal_within_threshold(self):
        """Test line slightly off horizontal is still classified as horizontal."""
        # Line at ~5 degree angle
        line = np.array([0, 50, 100, 59])  # rises 9 over 100
        result = classify_line_orientation(line, angle_threshold=15.0)

        assert result == "horizontal"


class TestLineIntersection:
    """Tests for line intersection calculation."""

    def test_perpendicular_intersection(self):
        """Test intersection of perpendicular lines."""
        line1 = np.array([0, 50, 100, 50])   # Horizontal at y=50
        line2 = np.array([50, 0, 50, 100])   # Vertical at x=50

        result = line_intersection(line1, line2)

        assert result is not None
        assert abs(result[0] - 50) < 1
        assert abs(result[1] - 50) < 1

    def test_parallel_lines_no_intersection(self):
        """Test that parallel lines return None."""
        line1 = np.array([0, 30, 100, 30])   # Horizontal at y=30
        line2 = np.array([0, 70, 100, 70])   # Horizontal at y=70

        result = line_intersection(line1, line2)

        assert result is None

    def test_angled_intersection(self):
        """Test intersection of angled lines."""
        # Line from (0,0) to (100,100)
        line1 = np.array([0, 0, 100, 100])
        # Line from (0,100) to (100,0)
        line2 = np.array([0, 100, 100, 0])

        result = line_intersection(line1, line2)

        assert result is not None
        assert abs(result[0] - 50) < 1
        assert abs(result[1] - 50) < 1

    def test_intersection_outside_segments(self):
        """Test intersection even when point is outside segment bounds."""
        # Short segments that would intersect if extended
        line1 = np.array([0, 0, 10, 0])   # Short horizontal
        line2 = np.array([50, 0, 50, 10])  # Short vertical

        result = line_intersection(line1, line2)

        # Should still find the intersection point
        assert result is not None


class TestEdgeSelection:
    """Tests for selecting best cell edges."""

    def test_selects_topmost_horizontal(self):
        """Test that topmost horizontal line is selected."""
        h_lines = np.array([
            [0, 50, 100, 50],   # y=50
            [0, 20, 100, 20],   # y=20 (topmost)
            [0, 80, 100, 80],   # y=80
        ])

        # Select line with minimum y
        top = min(h_lines, key=lambda l: (l[1] + l[3]) / 2)

        assert (top[1] + top[3]) / 2 == 20

    def test_selects_leftmost_vertical(self):
        """Test that leftmost vertical line is selected."""
        v_lines = np.array([
            [50, 0, 50, 100],   # x=50
            [20, 0, 20, 100],   # x=20 (leftmost)
            [80, 0, 80, 100],   # x=80
        ])

        # Select line with minimum x
        left = min(v_lines, key=lambda l: (l[0] + l[2]) / 2)

        assert (left[0] + left[2]) / 2 == 20


class TestCellCornerComputation:
    """Tests for computing cell corners from intersection."""

    def test_computes_square_cell(self):
        """Test corner computation for square cell."""
        top_left = (10.0, 20.0)
        width, height = 50, 50

        corners = np.array([
            [top_left[0], top_left[1]],
            [top_left[0] + width, top_left[1]],
            [top_left[0] + width, top_left[1] + height],
            [top_left[0], top_left[1] + height],
        ])

        assert np.allclose(corners[0], [10, 20])   # TL
        assert np.allclose(corners[1], [60, 20])   # TR
        assert np.allclose(corners[2], [60, 70])   # BR
        assert np.allclose(corners[3], [10, 70])   # BL

    def test_computes_rectangular_cell(self):
        """Test corner computation for rectangular cell."""
        top_left = (0.0, 0.0)
        width, height = 100, 50

        corners = np.array([
            [top_left[0], top_left[1]],
            [top_left[0] + width, top_left[1]],
            [top_left[0] + width, top_left[1] + height],
            [top_left[0], top_left[1] + height],
        ])

        assert np.allclose(corners[0], [0, 0])
        assert np.allclose(corners[1], [100, 0])
        assert np.allclose(corners[2], [100, 50])
        assert np.allclose(corners[3], [0, 50])


class TestIntegration:
    """Integration tests for the full edge detection pipeline."""

    def test_detects_cell_edges_in_synthetic_grid(self, synthetic_grid_image):
        """Test edge detection on synthetic grid."""
        image, corners = synthetic_grid_image(size=450, rotation=0)

        # Convert to grayscale and apply Canny
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Extract ROI around expected first cell location (~50,50)
        margin = 15
        roi = edges[50-margin:100+margin, 50-margin:100+margin]

        # Should have edges in the ROI
        assert roi.sum() > 0

    def test_hough_finds_lines_in_synthetic_grid(self, synthetic_grid_image):
        """Test Hough line detection on synthetic grid."""
        image, _ = synthetic_grid_image(size=450)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=50,
            minLineLength=30, maxLineGap=10
        )

        # Should find many lines (grid has lots of lines)
        assert lines is not None
        assert len(lines) > 10


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_no_lines_detected(self):
        """Test handling when no lines are detected."""
        # Black image - no edges, no lines
        edges = np.zeros((100, 100), dtype=np.uint8)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=30,
            minLineLength=30, maxLineGap=5
        )

        assert lines is None

    def test_handles_only_horizontal_lines(self):
        """Test handling when only horizontal lines are found."""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(image, (10, 50), (90, 50), 255, 2)

        lines = cv2.HoughLinesP(
            image, 1, np.pi/180, threshold=30,
            minLineLength=30, maxLineGap=5
        )

        if lines is not None:
            # Classify
            h_lines = []
            v_lines = []
            for line in lines:
                angle = line_angle(line[0])
                if angle < 20 or angle > 160:
                    h_lines.append(line)
                elif 70 < angle < 110:
                    v_lines.append(line)

            # Should have horizontal but no vertical
            assert len(h_lines) > 0
            assert len(v_lines) == 0

    def test_handles_noisy_image(self, synthetic_grid_image):
        """Test edge detection on noisy image."""
        image, _ = synthetic_grid_image(size=450, add_noise=True)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply stronger blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Should still detect some edges
        assert edges.sum() > 0
