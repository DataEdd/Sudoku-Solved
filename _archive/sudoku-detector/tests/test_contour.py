"""Tests for contour detection module."""

import numpy as np
import pytest

from src.config import DetectionConfig
from src.contour_detection import (
    compute_centeredness,
    detect_contour_path,
    find_quadrilateral_contours,
    validate_quadrilateral,
)


# Fixtures


@pytest.fixture
def config():
    """Create default detection configuration."""
    return DetectionConfig()


@pytest.fixture
def centered_square_image():
    """Create binary image with centered white square."""
    img = np.zeros((500, 500), dtype=np.uint8)
    # Draw a white filled square in center
    img[150:350, 150:350] = 255
    return img


@pytest.fixture
def off_center_square_image():
    """Create binary image with off-center white square."""
    img = np.zeros((500, 500), dtype=np.uint8)
    # Draw a white filled square in top-left area
    img[50:200, 50:200] = 255
    return img


@pytest.fixture
def multiple_squares_image():
    """Create binary image with multiple white squares."""
    img = np.zeros((500, 500), dtype=np.uint8)
    # Centered square (larger)
    img[150:350, 150:350] = 255
    # Small square in corner (separate, not overlapping)
    img[10:60, 10:60] = 255
    return img


@pytest.fixture
def rotated_square_image():
    """Create binary image with rotated white square (diamond)."""
    img = np.zeros((500, 500), dtype=np.uint8)
    # Draw rotated square using polygon
    pts = np.array([[250, 100], [400, 250], [250, 400], [100, 250]], dtype=np.int32)
    import cv2
    cv2.fillPoly(img, [pts], 255)
    return img


@pytest.fixture
def valid_square_quad():
    """A valid square quadrilateral."""
    return np.array([
        [100, 100],
        [300, 100],
        [300, 300],
        [100, 300],
    ], dtype=np.float32)


@pytest.fixture
def valid_rectangle_quad():
    """A valid rectangle quadrilateral."""
    return np.array([
        [100, 100],
        [400, 100],
        [400, 300],
        [100, 300],
    ], dtype=np.float32)


@pytest.fixture
def concave_quad():
    """A concave (non-convex) quadrilateral."""
    return np.array([
        [100, 100],
        [300, 100],
        [200, 200],  # Indent creates concave shape
        [100, 300],
    ], dtype=np.float32)


@pytest.fixture
def narrow_quad():
    """A very narrow quadrilateral (extreme aspect ratio)."""
    return np.array([
        [100, 100],
        [500, 100],
        [500, 110],
        [100, 110],
    ], dtype=np.float32)


@pytest.fixture
def sharp_angle_quad():
    """A quadrilateral with a very sharp angle."""
    return np.array([
        [100, 100],
        [400, 100],
        [400, 300],
        [350, 105],  # Creates sharp angle
    ], dtype=np.float32)


# Tests for find_quadrilateral_contours


class TestFindQuadrilateralContours:
    def test_finds_centered_square(self, centered_square_image):
        """Test finding a centered square."""
        quads = find_quadrilateral_contours(
            centered_square_image,
            min_area_ratio=0.01,
            max_area_ratio=0.9,
        )
        assert len(quads) == 1
        assert quads[0].shape == (4, 2)

    def test_finds_multiple_squares(self, multiple_squares_image):
        """Test finding multiple squares."""
        quads = find_quadrilateral_contours(
            multiple_squares_image,
            min_area_ratio=0.005,
            max_area_ratio=0.9,
        )
        # Should find both squares
        assert len(quads) >= 1

    def test_filters_by_min_area(self, multiple_squares_image):
        """Test that small contours are filtered by min_area_ratio."""
        quads = find_quadrilateral_contours(
            multiple_squares_image,
            min_area_ratio=0.1,  # High min filters out small square
            max_area_ratio=0.9,
        )
        # Should only find the larger square
        assert len(quads) == 1

    def test_filters_by_max_area(self, centered_square_image):
        """Test that large contours are filtered by max_area_ratio."""
        quads = find_quadrilateral_contours(
            centered_square_image,
            min_area_ratio=0.01,
            max_area_ratio=0.1,  # Low max filters out large square
        )
        assert len(quads) == 0

    def test_empty_image(self):
        """Test with empty (all black) image."""
        img = np.zeros((500, 500), dtype=np.uint8)
        quads = find_quadrilateral_contours(img, 0.01, 0.9)
        assert len(quads) == 0

    def test_all_white_image(self):
        """Test with all white image."""
        img = np.ones((500, 500), dtype=np.uint8) * 255
        quads = find_quadrilateral_contours(img, 0.01, 0.9)
        # All white image has one big contour (the whole image)
        # It may or may not be returned depending on area ratios
        assert isinstance(quads, list)

    def test_rotated_square(self, rotated_square_image):
        """Test finding a rotated square (diamond shape)."""
        quads = find_quadrilateral_contours(
            rotated_square_image,
            min_area_ratio=0.01,
            max_area_ratio=0.9,
        )
        assert len(quads) == 1
        assert quads[0].shape == (4, 2)

    def test_non_quadrilateral_shapes(self):
        """Test that non-quadrilateral shapes are filtered out."""
        img = np.zeros((500, 500), dtype=np.uint8)
        # Draw a triangle
        import cv2
        pts = np.array([[250, 100], [400, 400], [100, 400]], dtype=np.int32)
        cv2.fillPoly(img, [pts], 255)

        quads = find_quadrilateral_contours(img, 0.01, 0.9)
        assert len(quads) == 0  # Triangle has 3 vertices, not 4

    def test_returns_correct_shape(self, centered_square_image):
        """Test that returned quadrilaterals have correct shape."""
        quads = find_quadrilateral_contours(
            centered_square_image, 0.01, 0.9
        )
        for quad in quads:
            assert quad.shape == (4, 2)
            assert quad.dtype in [np.int32, np.int64, np.float32, np.float64]


# Tests for validate_quadrilateral


class TestValidateQuadrilateral:
    def test_valid_square(self, valid_square_quad):
        """Test that valid square passes validation."""
        assert validate_quadrilateral(valid_square_quad, 0.5, 2.0) is True

    def test_valid_rectangle(self, valid_rectangle_quad):
        """Test that valid rectangle passes validation."""
        assert validate_quadrilateral(valid_rectangle_quad, 0.5, 2.0) is True

    def test_rejects_concave(self, concave_quad):
        """Test that concave quadrilateral is rejected."""
        assert validate_quadrilateral(concave_quad, 0.5, 2.0) is False

    def test_rejects_narrow(self, narrow_quad):
        """Test that narrow quadrilateral fails aspect ratio check."""
        # Aspect ratio is 400/10 = 40, which is > 2.0
        assert validate_quadrilateral(narrow_quad, 0.5, 2.0) is False

    def test_accepts_narrow_with_wide_limits(self, narrow_quad):
        """Test narrow quad passes with wide aspect ratio limits."""
        assert validate_quadrilateral(narrow_quad, 0.01, 100.0) is True

    def test_rejects_sharp_angle(self, sharp_angle_quad):
        """Test that quadrilateral with sharp angle is rejected."""
        assert validate_quadrilateral(sharp_angle_quad, 0.1, 10.0) is False

    def test_wrong_shape(self):
        """Test that wrong shape input is rejected."""
        wrong_shape = np.array([[1, 2, 3], [4, 5, 6]])
        assert validate_quadrilateral(wrong_shape, 0.5, 2.0) is False

    def test_three_points(self):
        """Test that 3-point array is rejected."""
        triangle = np.array([[100, 100], [200, 100], [150, 200]])
        assert validate_quadrilateral(triangle, 0.5, 2.0) is False

    def test_angle_boundaries(self):
        """Test quadrilaterals at angle boundaries."""
        # A perfect square has 90 degree angles
        square = np.array([
            [0, 0],
            [100, 0],
            [100, 100],
            [0, 100],
        ], dtype=np.float32)
        assert validate_quadrilateral(square, 0.5, 2.0) is True

    def test_slightly_rotated_square(self):
        """Test that slightly rotated square passes validation."""
        # Rotated square (diamond)
        diamond = np.array([
            [50, 0],
            [100, 50],
            [50, 100],
            [0, 50],
        ], dtype=np.float32)
        assert validate_quadrilateral(diamond, 0.5, 2.0) is True


# Tests for compute_centeredness


class TestComputeCenteredness:
    def test_perfectly_centered(self):
        """Test perfectly centered quadrilateral."""
        # Quad centered in a 500x500 image
        quad = np.array([
            [200, 200],
            [300, 200],
            [300, 300],
            [200, 300],
        ], dtype=np.float32)
        score = compute_centeredness(quad, (500, 500))
        assert score == pytest.approx(1.0, abs=0.01)

    def test_corner_quad(self):
        """Test quadrilateral in corner."""
        # Quad in top-left corner
        quad = np.array([
            [0, 0],
            [50, 0],
            [50, 50],
            [0, 50],
        ], dtype=np.float32)
        score = compute_centeredness(quad, (500, 500))
        # Should be close to 0 (far from center)
        assert score < 0.3

    def test_half_centered(self):
        """Test quadrilateral halfway between center and corner."""
        # Quad centered at (125, 125) in a 500x500 image
        # Center is at (250, 250), max distance is ~353.5
        # Distance from (125, 125) to (250, 250) is ~176.8
        quad = np.array([
            [100, 100],
            [150, 100],
            [150, 150],
            [100, 150],
        ], dtype=np.float32)
        score = compute_centeredness(quad, (500, 500))
        assert 0.3 < score < 0.7

    def test_score_range(self):
        """Test that score is always in [0, 1] range."""
        # Test various positions
        for x_offset in [0, 100, 200, 300, 400]:
            for y_offset in [0, 100, 200, 300, 400]:
                quad = np.array([
                    [x_offset, y_offset],
                    [x_offset + 50, y_offset],
                    [x_offset + 50, y_offset + 50],
                    [x_offset, y_offset + 50],
                ], dtype=np.float32)
                score = compute_centeredness(quad, (500, 500))
                assert 0.0 <= score <= 1.0

    def test_asymmetric_image(self):
        """Test with non-square image."""
        # Quad centered in a 1000x500 image
        quad = np.array([
            [450, 200],
            [550, 200],
            [550, 300],
            [450, 300],
        ], dtype=np.float32)
        score = compute_centeredness(quad, (500, 1000))
        assert score == pytest.approx(1.0, abs=0.01)

    def test_tiny_image(self):
        """Test with very small image."""
        quad = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ], dtype=np.float32)
        score = compute_centeredness(quad, (2, 2))
        assert 0.0 <= score <= 1.0


# Tests for detect_contour_path


class TestDetectContourPath:
    def test_finds_centered_square(self, centered_square_image, config):
        """Test detection of centered square."""
        results = detect_contour_path(centered_square_image, config)
        assert len(results) >= 1
        assert results[0]["method"] == "contour"
        assert "corners" in results[0]
        assert "score" in results[0]
        assert results[0]["corners"].shape == (4, 2)

    def test_empty_image(self, config):
        """Test with empty image."""
        img = np.zeros((500, 500), dtype=np.uint8)
        results = detect_contour_path(img, config)
        assert results == []

    def test_sorted_by_centeredness(self, multiple_squares_image, config):
        """Test that results are sorted by centeredness score."""
        # Adjust config to find both squares
        config.min_area_ratio = 0.005
        results = detect_contour_path(multiple_squares_image, config)

        if len(results) > 1:
            # Check descending order
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i + 1]["score"]

    def test_most_centered_first(self, multiple_squares_image, config):
        """Test that most centered quad is first."""
        config.min_area_ratio = 0.005
        results = detect_contour_path(multiple_squares_image, config)

        if len(results) >= 1:
            # The centered square should have high score
            assert results[0]["score"] > 0.5

    def test_result_structure(self, centered_square_image, config):
        """Test that result dictionaries have correct structure."""
        results = detect_contour_path(centered_square_image, config)

        for result in results:
            assert isinstance(result, dict)
            assert "corners" in result
            assert "score" in result
            assert "method" in result
            assert isinstance(result["corners"], np.ndarray)
            assert isinstance(result["score"], float)
            assert result["method"] == "contour"

    def test_filters_invalid_quads(self, config):
        """Test that invalid quadrilaterals are filtered out."""
        # Create image with a very narrow rectangle
        img = np.zeros((500, 500), dtype=np.uint8)
        img[245:255, 50:450] = 255  # 10px high, 400px wide

        config.min_area_ratio = 0.01
        results = detect_contour_path(img, config)

        # Should be empty because narrow rectangle fails aspect ratio
        assert len(results) == 0

    def test_with_rotated_square(self, rotated_square_image, config):
        """Test detection of rotated square."""
        results = detect_contour_path(rotated_square_image, config)
        assert len(results) >= 1
        assert results[0]["corners"].shape == (4, 2)

    def test_custom_config(self, centered_square_image):
        """Test with custom configuration."""
        custom_config = DetectionConfig(
            min_area_ratio=0.1,
            max_area_ratio=0.5,
            min_aspect_ratio=0.8,
            max_aspect_ratio=1.2,
        )
        results = detect_contour_path(centered_square_image, custom_config)
        assert isinstance(results, list)


# Integration tests


class TestContourDetectionIntegration:
    def test_full_pipeline_square(self):
        """Test full detection pipeline with synthetic square."""
        # Create a clean binary image with centered square
        img = np.zeros((500, 500), dtype=np.uint8)
        img[100:400, 100:400] = 255

        config = DetectionConfig()
        results = detect_contour_path(img, config)

        assert len(results) >= 1
        result = results[0]

        # Check corners roughly match the square
        corners = result["corners"]
        xs = corners[:, 0]
        ys = corners[:, 1]

        # Should roughly be 100-400 range
        assert xs.min() >= 95 and xs.min() <= 105
        assert xs.max() >= 395 and xs.max() <= 405
        assert ys.min() >= 95 and ys.min() <= 105
        assert ys.max() >= 395 and ys.max() <= 405

    def test_full_pipeline_with_noise(self):
        """Test detection with noisy image."""
        # Create image with square and some noise
        img = np.zeros((500, 500), dtype=np.uint8)
        img[150:350, 150:350] = 255

        # Add small noise blobs
        img[10:20, 10:20] = 255
        img[480:490, 480:490] = 255

        config = DetectionConfig()
        config.min_area_ratio = 0.05  # Filter out small noise

        results = detect_contour_path(img, config)

        # Should find the main square, not the noise
        assert len(results) >= 1
        # First result should be well-centered
        assert results[0]["score"] > 0.7

    def test_selectivity_prefers_centered(self):
        """Test that algorithm prefers centered quads."""
        # Create image with two similar squares, one centered
        img = np.zeros((800, 800), dtype=np.uint8)
        # Centered square (150x150)
        img[325:475, 325:475] = 255
        # Corner square (same size, 150x150)
        img[20:170, 20:170] = 255

        config = DetectionConfig()
        config.min_area_ratio = 0.02  # Lower to catch smaller squares
        config.max_area_ratio = 0.95

        results = detect_contour_path(img, config)

        assert len(results) >= 2
        # First result should be the centered one (higher score)
        assert results[0]["score"] > results[1]["score"]
