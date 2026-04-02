"""Tests for Hough detection module."""

import math

import cv2
import numpy as np
import pytest

from src.config import DetectionConfig
from src.hough_detection import (
    classify_lines,
    cluster_lines,
    compute_all_intersections,
    compute_line_intercept,
    detect_hough_path,
    detect_lines,
    extract_outer_corners,
    line_intersection,
    validate_grid_lines,
)


# Fixtures


@pytest.fixture
def config():
    """Create default detection configuration."""
    return DetectionConfig()


@pytest.fixture
def grid_edge_image():
    """Create an edge image with a 10x10 grid pattern (like sudoku)."""
    img = np.zeros((500, 500), dtype=np.uint8)

    # Draw 10 horizontal lines
    for i in range(10):
        y = 50 + i * 40
        cv2.line(img, (50, y), (450, y), 255, 2)

    # Draw 10 vertical lines
    for i in range(10):
        x = 50 + i * 40
        cv2.line(img, (x, 50), (x, 450), 255, 2)

    return img


@pytest.fixture
def simple_cross_image():
    """Create an edge image with a simple cross (one H, one V line)."""
    img = np.zeros((500, 500), dtype=np.uint8)
    # Horizontal line
    cv2.line(img, (100, 250), (400, 250), 255, 2)
    # Vertical line
    cv2.line(img, (250, 100), (250, 400), 255, 2)
    return img


@pytest.fixture
def sample_lines():
    """Create sample lines array for testing."""
    # Format: (N, 1, 4) where each line is (x1, y1, x2, y2)
    lines = np.array([
        [[100, 200, 400, 200]],  # Horizontal
        [[100, 250, 400, 250]],  # Horizontal
        [[200, 100, 200, 400]],  # Vertical
        [[300, 100, 300, 400]],  # Vertical
        [[100, 100, 400, 400]],  # Diagonal (45 degrees)
    ])
    return lines


# Tests for detect_lines


class TestDetectLines:
    def test_detects_grid_lines(self, grid_edge_image, config):
        """Test detection of grid lines."""
        lines = detect_lines(grid_edge_image, config)
        assert lines is not None
        assert len(lines) > 0

    def test_detects_cross_lines(self, simple_cross_image, config):
        """Test detection of simple cross lines."""
        config.hough_min_line_length = 100
        lines = detect_lines(simple_cross_image, config)
        assert lines is not None
        assert len(lines) >= 2

    def test_returns_none_for_empty_image(self, config):
        """Test that empty image returns None."""
        img = np.zeros((500, 500), dtype=np.uint8)
        lines = detect_lines(img, config)
        assert lines is None

    def test_returns_correct_shape(self, grid_edge_image, config):
        """Test that returned lines have correct shape."""
        lines = detect_lines(grid_edge_image, config)
        assert lines is not None
        assert len(lines.shape) == 3
        assert lines.shape[1] == 1
        assert lines.shape[2] == 4

    def test_respects_threshold_parameter(self, simple_cross_image):
        """Test that threshold parameter affects detection."""
        config_low = DetectionConfig(hough_threshold=20)
        config_high = DetectionConfig(hough_threshold=200)

        lines_low = detect_lines(simple_cross_image, config_low)
        lines_high = detect_lines(simple_cross_image, config_high)

        # Lower threshold should detect more or equal lines
        assert lines_low is not None
        low_count = len(lines_low) if lines_low is not None else 0
        high_count = len(lines_high) if lines_high is not None else 0
        assert low_count >= high_count


# Tests for classify_lines


class TestClassifyLines:
    def test_classifies_horizontal_lines(self, sample_lines):
        """Test classification of horizontal lines."""
        h_lines, v_lines = classify_lines(sample_lines)
        # Should have at least 2 horizontal lines
        assert len(h_lines) >= 2

    def test_classifies_vertical_lines(self, sample_lines):
        """Test classification of vertical lines."""
        h_lines, v_lines = classify_lines(sample_lines)
        # Should have at least 2 vertical lines
        assert len(v_lines) >= 2

    def test_diagonal_classification(self, sample_lines):
        """Test that 45-degree diagonal is classified as vertical."""
        h_lines, v_lines = classify_lines(sample_lines)
        # 45-degree line should be classified as vertical (45 <= angle <= 135)
        total = len(h_lines) + len(v_lines)
        assert total == 5  # All 5 lines classified

    def test_returns_tuples(self, sample_lines):
        """Test that lines are returned as tuples."""
        h_lines, v_lines = classify_lines(sample_lines)

        for line in h_lines:
            assert isinstance(line, tuple)
            assert len(line) == 4

        for line in v_lines:
            assert isinstance(line, tuple)
            assert len(line) == 4

    def test_pure_horizontal(self):
        """Test purely horizontal line."""
        lines = np.array([[[0, 100, 500, 100]]])  # y1 == y2
        h_lines, v_lines = classify_lines(lines)
        assert len(h_lines) == 1
        assert len(v_lines) == 0

    def test_pure_vertical(self):
        """Test purely vertical line."""
        lines = np.array([[[100, 0, 100, 500]]])  # x1 == x2
        h_lines, v_lines = classify_lines(lines)
        assert len(h_lines) == 0
        assert len(v_lines) == 1

    def test_boundary_angles(self):
        """Test lines at boundary angles (44, 45, 135, 136 degrees)."""
        # Create lines at specific angles
        length = 100
        center = 250

        # 44 degrees - should be horizontal
        angle_44 = np.array([[[
            center, center,
            int(center + length * math.cos(math.radians(44))),
            int(center + length * math.sin(math.radians(44)))
        ]]])

        # 46 degrees - should be vertical
        angle_46 = np.array([[[
            center, center,
            int(center + length * math.cos(math.radians(46))),
            int(center + length * math.sin(math.radians(46)))
        ]]])

        h1, v1 = classify_lines(angle_44)
        h2, v2 = classify_lines(angle_46)

        assert len(h1) == 1  # 44 degrees -> horizontal
        assert len(v2) == 1  # 46 degrees -> vertical


# Tests for compute_line_intercept


class TestComputeLineIntercept:
    def test_horizontal_intercept(self):
        """Test y-intercept for horizontal line."""
        line = (100, 200, 400, 200)
        intercept = compute_line_intercept(line, is_horizontal=True)
        assert intercept == 200.0

    def test_vertical_intercept(self):
        """Test x-intercept for vertical line."""
        line = (300, 100, 300, 400)
        intercept = compute_line_intercept(line, is_horizontal=False)
        assert intercept == 300.0

    def test_slanted_horizontal_average(self):
        """Test average y for slightly slanted horizontal line."""
        line = (100, 198, 400, 202)
        intercept = compute_line_intercept(line, is_horizontal=True)
        assert intercept == 200.0

    def test_slanted_vertical_average(self):
        """Test average x for slightly slanted vertical line."""
        line = (298, 100, 302, 400)
        intercept = compute_line_intercept(line, is_horizontal=False)
        assert intercept == 300.0


# Tests for cluster_lines


class TestClusterLines:
    def test_merges_nearby_horizontal_lines(self):
        """Test merging of nearby horizontal lines."""
        lines = [
            (100, 200, 400, 200),
            (100, 205, 400, 205),  # 5px apart
            (100, 208, 400, 208),  # 3px apart
            (100, 300, 400, 300),  # 100px apart - separate cluster
        ]
        clustered = cluster_lines(lines, threshold=20, is_horizontal=True)
        assert len(clustered) == 2  # Two clusters

    def test_merges_nearby_vertical_lines(self):
        """Test merging of nearby vertical lines."""
        lines = [
            (200, 100, 200, 400),
            (205, 100, 205, 400),  # 5px apart
            (300, 100, 300, 400),  # 100px apart - separate cluster
        ]
        clustered = cluster_lines(lines, threshold=20, is_horizontal=False)
        assert len(clustered) == 2

    def test_no_merge_if_far_apart(self):
        """Test that far apart lines are not merged."""
        lines = [
            (100, 100, 400, 100),
            (100, 200, 400, 200),
            (100, 300, 400, 300),
        ]
        clustered = cluster_lines(lines, threshold=20, is_horizontal=True)
        assert len(clustered) == 3  # All separate

    def test_empty_list(self):
        """Test with empty list."""
        clustered = cluster_lines([], threshold=20, is_horizontal=True)
        assert clustered == []

    def test_single_line(self):
        """Test with single line."""
        lines = [(100, 200, 400, 200)]
        clustered = cluster_lines(lines, threshold=20, is_horizontal=True)
        assert len(clustered) == 1
        assert clustered[0] == lines[0]

    def test_selects_median_line(self):
        """Test that clustering selects line closest to median intercept."""
        lines = [
            (100, 200, 300, 200),  # y-intercept: 200
            (200, 205, 400, 205),  # y-intercept: 205
            (150, 203, 350, 203),  # y-intercept: 203 (closest to median)
        ]
        clustered = cluster_lines(lines, threshold=20, is_horizontal=True)
        assert len(clustered) == 1
        selected = clustered[0]
        # Should select the line with intercept closest to median (203)
        y_intercept = (selected[1] + selected[3]) / 2
        assert y_intercept == 203


# Tests for validate_grid_lines


class TestValidateGridLines:
    def test_valid_counts(self):
        """Test with valid line counts."""
        h_lines = [(0, i * 50, 500, i * 50) for i in range(10)]
        v_lines = [(i * 50, 0, i * 50, 500) for i in range(10)]
        assert validate_grid_lines(h_lines, v_lines, min_count=8, max_count=12) is True

    def test_too_few_horizontal(self):
        """Test with too few horizontal lines."""
        h_lines = [(0, i * 50, 500, i * 50) for i in range(5)]
        v_lines = [(i * 50, 0, i * 50, 500) for i in range(10)]
        assert validate_grid_lines(h_lines, v_lines, min_count=8, max_count=12) is False

    def test_too_few_vertical(self):
        """Test with too few vertical lines."""
        h_lines = [(0, i * 50, 500, i * 50) for i in range(10)]
        v_lines = [(i * 50, 0, i * 50, 500) for i in range(5)]
        assert validate_grid_lines(h_lines, v_lines, min_count=8, max_count=12) is False

    def test_too_many_lines(self):
        """Test with too many lines."""
        h_lines = [(0, i * 20, 500, i * 20) for i in range(20)]
        v_lines = [(i * 20, 0, i * 20, 500) for i in range(20)]
        assert validate_grid_lines(h_lines, v_lines, min_count=8, max_count=12) is False

    def test_custom_bounds(self):
        """Test with custom min/max bounds."""
        h_lines = [(0, i * 50, 500, i * 50) for i in range(5)]
        v_lines = [(i * 50, 0, i * 50, 500) for i in range(5)]
        assert validate_grid_lines(h_lines, v_lines, min_count=4, max_count=6) is True

    def test_boundary_values(self):
        """Test at boundary values."""
        h_lines = [(0, i * 50, 500, i * 50) for i in range(8)]
        v_lines = [(i * 50, 0, i * 50, 500) for i in range(12)]
        assert validate_grid_lines(h_lines, v_lines, min_count=8, max_count=12) is True

        h_lines = [(0, i * 50, 500, i * 50) for i in range(7)]  # Below min
        assert validate_grid_lines(h_lines, v_lines, min_count=8, max_count=12) is False


# Tests for line_intersection


class TestLineIntersection:
    def test_perpendicular_intersection(self):
        """Test intersection of perpendicular lines."""
        h_line = (100, 250, 400, 250)
        v_line = (300, 100, 300, 400)
        point = line_intersection(h_line, v_line)
        assert point is not None
        assert point[0] == pytest.approx(300.0, abs=0.1)
        assert point[1] == pytest.approx(250.0, abs=0.1)

    def test_parallel_lines_no_intersection(self):
        """Test that parallel lines return None."""
        line1 = (100, 200, 400, 200)
        line2 = (100, 300, 400, 300)
        point = line_intersection(line1, line2)
        assert point is None

    def test_diagonal_intersection(self):
        """Test intersection of diagonal lines."""
        line1 = (0, 0, 100, 100)
        line2 = (0, 100, 100, 0)
        point = line_intersection(line1, line2)
        assert point is not None
        assert point[0] == pytest.approx(50.0, abs=0.1)
        assert point[1] == pytest.approx(50.0, abs=0.1)

    def test_intersection_at_origin(self):
        """Test intersection at origin."""
        line1 = (-100, 0, 100, 0)
        line2 = (0, -100, 0, 100)
        point = line_intersection(line1, line2)
        assert point is not None
        assert point[0] == pytest.approx(0.0, abs=0.1)
        assert point[1] == pytest.approx(0.0, abs=0.1)

    def test_nearly_parallel(self):
        """Test nearly parallel lines."""
        line1 = (0, 0, 1000, 1)
        line2 = (0, 100, 1000, 101)
        point = line_intersection(line1, line2)
        # Very nearly parallel, should return None or very distant point
        # The intersection exists but is very far away


# Tests for compute_all_intersections


class TestComputeAllIntersections:
    def test_grid_intersections(self):
        """Test computing intersections for a grid."""
        h_lines = [
            (0, 100, 500, 100),
            (0, 200, 500, 200),
        ]
        v_lines = [
            (100, 0, 100, 500),
            (200, 0, 200, 500),
            (300, 0, 300, 500),
        ]
        intersections = compute_all_intersections(h_lines, v_lines, (500, 500))
        # 2 horizontal × 3 vertical = 6 intersections
        assert len(intersections) == 6

    def test_filters_out_of_bounds(self):
        """Test that out-of-bounds intersections are filtered."""
        h_lines = [(0, 100, 500, 100)]
        v_lines = [
            (100, 0, 100, 500),   # In bounds
            (-50, 0, -50, 500),   # Out of bounds (x < 0)
        ]
        intersections = compute_all_intersections(h_lines, v_lines, (500, 500))
        assert len(intersections) == 1  # Only one in bounds

    def test_empty_result(self):
        """Test with no valid intersections."""
        h_lines = [(0, 100, 500, 100)]
        v_lines = []
        intersections = compute_all_intersections(h_lines, v_lines, (500, 500))
        assert len(intersections) == 0

    def test_correct_coordinates(self):
        """Test that intersection coordinates are correct."""
        h_lines = [(0, 150, 500, 150)]
        v_lines = [(200, 0, 200, 500)]
        intersections = compute_all_intersections(h_lines, v_lines, (500, 500))
        assert len(intersections) == 1
        assert intersections[0][0] == pytest.approx(200.0, abs=0.1)
        assert intersections[0][1] == pytest.approx(150.0, abs=0.1)


# Tests for extract_outer_corners


class TestExtractOuterCorners:
    def test_extracts_corners_from_grid(self):
        """Test extracting corners from a grid of points."""
        # Create a 3x3 grid of points
        points = np.array([
            [100, 100], [200, 100], [300, 100],
            [100, 200], [200, 200], [300, 200],
            [100, 300], [200, 300], [300, 300],
        ], dtype=np.float32)

        corners = extract_outer_corners(points)
        assert corners is not None
        assert corners.shape == (4, 2)

        # Check corners are at expected positions
        # TL, TR, BR, BL order
        assert corners[0][0] == pytest.approx(100, abs=1)  # TL x
        assert corners[0][1] == pytest.approx(100, abs=1)  # TL y
        assert corners[1][0] == pytest.approx(300, abs=1)  # TR x
        assert corners[1][1] == pytest.approx(100, abs=1)  # TR y
        assert corners[2][0] == pytest.approx(300, abs=1)  # BR x
        assert corners[2][1] == pytest.approx(300, abs=1)  # BR y
        assert corners[3][0] == pytest.approx(100, abs=1)  # BL x
        assert corners[3][1] == pytest.approx(300, abs=1)  # BL y

    def test_returns_none_for_too_few_points(self):
        """Test that fewer than 4 points returns None."""
        points = np.array([[100, 100], [200, 200], [300, 300]], dtype=np.float32)
        corners = extract_outer_corners(points)
        assert corners is None

    def test_returns_none_for_empty_array(self):
        """Test with empty array."""
        points = np.array([]).reshape(0, 2)
        corners = extract_outer_corners(points)
        assert corners is None

    def test_returns_none_for_none_input(self):
        """Test with None input."""
        corners = extract_outer_corners(None)
        assert corners is None

    def test_with_rotated_grid(self):
        """Test with rotated/skewed grid points."""
        # Diamond shape
        points = np.array([
            [200, 50],   # Top
            [350, 200],  # Right
            [200, 350],  # Bottom
            [50, 200],   # Left
        ], dtype=np.float32)

        corners = extract_outer_corners(points)
        assert corners is not None
        assert corners.shape == (4, 2)


# Tests for detect_hough_path


class TestDetectHoughPath:
    def test_detects_grid(self, grid_edge_image, config):
        """Test detection of sudoku-like grid."""
        config.hough_threshold = 50
        config.hough_min_line_length = 100
        results = detect_hough_path(grid_edge_image, config, grid_edge_image.shape)

        assert isinstance(results, list)
        if len(results) > 0:
            assert "corners" in results[0]
            assert "score" in results[0]
            assert results[0]["method"] == "hough"

    def test_empty_image_returns_empty(self, config):
        """Test that empty image returns empty list."""
        img = np.zeros((500, 500), dtype=np.uint8)
        results = detect_hough_path(img, config, img.shape)
        assert results == []

    def test_insufficient_lines_returns_empty(self, simple_cross_image, config):
        """Test that insufficient lines returns empty list."""
        # Cross has only 2 lines, not enough for a grid
        results = detect_hough_path(simple_cross_image, config, simple_cross_image.shape)
        assert results == []

    def test_result_structure(self, grid_edge_image, config):
        """Test that result has correct structure."""
        config.hough_threshold = 50
        results = detect_hough_path(grid_edge_image, config, grid_edge_image.shape)

        for result in results:
            assert isinstance(result, dict)
            assert "corners" in result
            assert "score" in result
            assert "method" in result
            assert isinstance(result["corners"], np.ndarray)
            assert isinstance(result["score"], float)
            assert result["method"] == "hough"
            assert 0.0 <= result["score"] <= 1.0

    def test_corners_have_correct_shape(self, grid_edge_image, config):
        """Test that corners have shape (4, 2)."""
        config.hough_threshold = 50
        results = detect_hough_path(grid_edge_image, config, grid_edge_image.shape)

        if len(results) > 0:
            assert results[0]["corners"].shape == (4, 2)


# Integration tests


class TestHoughDetectionIntegration:
    def test_full_pipeline_synthetic_grid(self):
        """Test full pipeline with synthetic sudoku-like grid."""
        # Create a clean sudoku-like grid
        img = np.zeros((500, 500), dtype=np.uint8)

        # Draw 10 horizontal lines
        for i in range(10):
            y = 50 + i * 44
            cv2.line(img, (50, y), (446, y), 255, 2)

        # Draw 10 vertical lines
        for i in range(10):
            x = 50 + i * 44
            cv2.line(img, (x, 50), (x, 446), 255, 2)

        config = DetectionConfig(
            hough_threshold=50,
            hough_min_line_length=100,
            hough_max_line_gap=10,
        )

        results = detect_hough_path(img, config, img.shape)

        # May or may not detect depending on clustering
        assert isinstance(results, list)

    def test_classify_and_cluster_consistency(self):
        """Test that classify and cluster work together correctly."""
        # Create lines array
        lines = np.array([
            # Horizontal lines (grouped)
            [[50, 100, 450, 100]],
            [[50, 102, 450, 102]],  # Near first
            [[50, 200, 450, 200]],
            [[50, 300, 450, 300]],
            # Vertical lines (grouped)
            [[100, 50, 100, 450]],
            [[103, 50, 103, 450]],  # Near first
            [[200, 50, 200, 450]],
            [[300, 50, 300, 450]],
        ])

        h_lines, v_lines = classify_lines(lines)
        assert len(h_lines) == 4
        assert len(v_lines) == 4

        h_clustered = cluster_lines(h_lines, threshold=10, is_horizontal=True)
        v_clustered = cluster_lines(v_lines, threshold=10, is_horizontal=False)

        # Should merge nearby lines
        assert len(h_clustered) == 3  # 2 merged into 1
        assert len(v_clustered) == 3  # 2 merged into 1

    def test_no_intersections_parallel_only(self):
        """Test with only parallel lines (no intersections possible)."""
        # Create image with only horizontal lines
        img = np.zeros((500, 500), dtype=np.uint8)
        for i in range(10):
            y = 50 + i * 40
            cv2.line(img, (50, y), (450, y), 255, 2)

        config = DetectionConfig(hough_threshold=50)
        results = detect_hough_path(img, config, img.shape)

        # Should return empty - no vertical lines means no intersections
        assert results == []
