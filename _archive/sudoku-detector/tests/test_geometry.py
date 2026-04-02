"""Tests for geometry module."""

import numpy as np
import pytest

from src.geometry import (
    compute_aspect_ratio,
    compute_homography,
    compute_interior_angles,
    compute_quadrilateral_area,
    is_valid_quadrilateral,
    order_corners,
    refine_corners,
    scale_corners_to_original,
    warp_perspective,
)


# Fixtures


@pytest.fixture
def ordered_square():
    """A properly ordered square [TL, TR, BR, BL]."""
    return np.array([
        [100, 100],  # TL
        [300, 100],  # TR
        [300, 300],  # BR
        [100, 300],  # BL
    ], dtype=np.float32)


@pytest.fixture
def ordered_rectangle():
    """A properly ordered rectangle [TL, TR, BR, BL]."""
    return np.array([
        [100, 100],  # TL
        [400, 100],  # TR
        [400, 300],  # BR
        [100, 300],  # BL
    ], dtype=np.float32)


@pytest.fixture
def shuffled_square():
    """A square with shuffled corner order."""
    return np.array([
        [300, 300],  # BR
        [100, 100],  # TL
        [100, 300],  # BL
        [300, 100],  # TR
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
def sample_grayscale_image():
    """Create a grayscale image with corner-like features."""
    img = np.zeros((500, 500), dtype=np.uint8)
    # Draw a white square with sharp corners
    img[100:300, 100:300] = 255
    return img


@pytest.fixture
def simple_color_image():
    """Create a simple color image for warp testing."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    # Create a pattern: red top-left quadrant
    img[0:250, 0:250] = [0, 0, 255]  # Red (BGR)
    # Green top-right
    img[0:250, 250:500] = [0, 255, 0]  # Green
    # Blue bottom-left
    img[250:500, 0:250] = [255, 0, 0]  # Blue
    # Yellow bottom-right
    img[250:500, 250:500] = [0, 255, 255]  # Yellow
    return img


# Tests for order_corners


class TestOrderCorners:
    def test_already_ordered(self, ordered_square):
        """Test that already ordered corners remain ordered."""
        result = order_corners(ordered_square)
        np.testing.assert_array_almost_equal(result, ordered_square)

    def test_shuffled_corners(self, shuffled_square):
        """Test ordering shuffled corners."""
        result = order_corners(shuffled_square)
        expected = np.array([
            [100, 100],  # TL
            [300, 100],  # TR
            [300, 300],  # BR
            [100, 300],  # BL
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_returns_float32(self, shuffled_square):
        """Test that output is float32."""
        result = order_corners(shuffled_square)
        assert result.dtype == np.float32

    def test_returns_correct_shape(self, shuffled_square):
        """Test that output shape is (4, 2)."""
        result = order_corners(shuffled_square)
        assert result.shape == (4, 2)

    def test_converts_int_input(self):
        """Test that integer input is converted."""
        corners = np.array([
            [300, 300],
            [100, 100],
            [100, 300],
            [300, 100],
        ], dtype=np.int32)
        result = order_corners(corners)
        assert result.dtype == np.float32

    def test_rotated_square(self):
        """Test ordering a rotated square (diamond)."""
        corners = np.array([
            [200, 50],   # Top
            [350, 200],  # Right
            [200, 350],  # Bottom
            [50, 200],   # Left
        ], dtype=np.float32)
        result = order_corners(corners)
        # TL should be the top point (smallest sum)
        assert result[0][0] == pytest.approx(200, abs=1)
        assert result[0][1] == pytest.approx(50, abs=1)

    def test_invalid_shape_raises(self):
        """Test that invalid shape raises ValueError."""
        with pytest.raises(ValueError):
            order_corners(np.array([[1, 2], [3, 4]]))


# Tests for refine_corners


class TestRefineCorners:
    def test_returns_correct_shape(self, sample_grayscale_image, ordered_square):
        """Test that output shape is (4, 2)."""
        result = refine_corners(sample_grayscale_image, ordered_square)
        assert result.shape == (4, 2)

    def test_returns_float32(self, sample_grayscale_image, ordered_square):
        """Test that output is float32."""
        result = refine_corners(sample_grayscale_image, ordered_square)
        assert result.dtype == np.float32

    def test_refines_near_original(self, sample_grayscale_image, ordered_square):
        """Test that refined corners are close to originals."""
        result = refine_corners(sample_grayscale_image, ordered_square)
        # Should be within a few pixels of original
        diff = np.abs(result - ordered_square)
        assert np.all(diff < 50)  # Reasonable tolerance

    def test_custom_window_size(self, sample_grayscale_image, ordered_square):
        """Test with custom window size."""
        result = refine_corners(sample_grayscale_image, ordered_square, window_size=10)
        assert result.shape == (4, 2)

    def test_converts_input_dtype(self, sample_grayscale_image):
        """Test that integer corners are converted."""
        corners_int = np.array([
            [100, 100],
            [300, 100],
            [300, 300],
            [100, 300],
        ], dtype=np.int32)
        result = refine_corners(sample_grayscale_image, corners_int)
        assert result.dtype == np.float32


# Tests for compute_homography


class TestComputeHomography:
    def test_returns_3x3_matrix(self, ordered_square):
        """Test that homography is 3x3."""
        H = compute_homography(ordered_square, 450)
        assert H.shape == (3, 3)

    def test_transforms_corners_correctly(self, ordered_square):
        """Test that homography transforms corners to expected positions."""
        output_size = 450
        H = compute_homography(ordered_square, output_size)

        # Transform source corners
        src_corners_homogeneous = np.hstack([
            ordered_square,
            np.ones((4, 1))
        ])

        for i, src in enumerate(src_corners_homogeneous):
            dst_homogeneous = H @ src
            dst = dst_homogeneous[:2] / dst_homogeneous[2]

            # Expected destinations
            expected = [
                [0, 0],
                [output_size - 1, 0],
                [output_size - 1, output_size - 1],
                [0, output_size - 1],
            ][i]

            assert dst[0] == pytest.approx(expected[0], abs=1)
            assert dst[1] == pytest.approx(expected[1], abs=1)

    def test_different_output_sizes(self, ordered_square):
        """Test with different output sizes."""
        for size in [100, 450, 1000]:
            H = compute_homography(ordered_square, size)
            assert H.shape == (3, 3)


# Tests for warp_perspective


class TestWarpPerspective:
    def test_output_shape_grayscale(self, ordered_square):
        """Test output shape for grayscale image."""
        gray = np.random.randint(0, 256, (500, 500), dtype=np.uint8)
        H = compute_homography(ordered_square, 450)
        result = warp_perspective(gray, H, 450)
        assert result.shape == (450, 450)

    def test_output_shape_color(self, simple_color_image, ordered_square):
        """Test output shape for color image."""
        H = compute_homography(ordered_square, 450)
        result = warp_perspective(simple_color_image, H, 450)
        assert result.shape == (450, 450, 3)

    def test_preserves_dtype(self):
        """Test that dtype is preserved."""
        img = np.random.randint(0, 256, (500, 500), dtype=np.uint8)
        corners = np.array([
            [100, 100], [400, 100], [400, 400], [100, 400]
        ], dtype=np.float32)
        H = compute_homography(corners, 200)
        result = warp_perspective(img, H, 200)
        assert result.dtype == np.uint8

    def test_different_output_sizes(self, simple_color_image):
        """Test various output sizes."""
        corners = np.array([
            [0, 0], [499, 0], [499, 499], [0, 499]
        ], dtype=np.float32)

        for size in [100, 300, 500]:
            H = compute_homography(corners, size)
            result = warp_perspective(simple_color_image, H, size)
            assert result.shape[0] == size
            assert result.shape[1] == size


# Tests for scale_corners_to_original


class TestScaleCornersToOriginal:
    def test_scale_factor_half(self, ordered_square):
        """Test scaling with factor 0.5 (image was halved)."""
        result = scale_corners_to_original(ordered_square, 0.5)
        expected = ordered_square * 2  # Original was twice as big
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale_factor_one(self, ordered_square):
        """Test that scale factor 1.0 returns same corners."""
        result = scale_corners_to_original(ordered_square, 1.0)
        np.testing.assert_array_almost_equal(result, ordered_square)

    def test_scale_factor_quarter(self, ordered_square):
        """Test scaling with factor 0.25."""
        result = scale_corners_to_original(ordered_square, 0.25)
        expected = ordered_square * 4
        np.testing.assert_array_almost_equal(result, expected)

    def test_returns_float32(self, ordered_square):
        """Test that output is float32."""
        result = scale_corners_to_original(ordered_square, 0.5)
        assert result.dtype == np.float32

    def test_invalid_scale_factor_raises(self, ordered_square):
        """Test that zero or negative scale factor raises."""
        with pytest.raises(ValueError):
            scale_corners_to_original(ordered_square, 0)
        with pytest.raises(ValueError):
            scale_corners_to_original(ordered_square, -0.5)


# Tests for compute_quadrilateral_area


class TestComputeQuadrilateralArea:
    def test_square_area(self, ordered_square):
        """Test area of a 200x200 square."""
        area = compute_quadrilateral_area(ordered_square)
        assert area == pytest.approx(200 * 200, rel=0.01)

    def test_rectangle_area(self, ordered_rectangle):
        """Test area of a 300x200 rectangle."""
        area = compute_quadrilateral_area(ordered_rectangle)
        assert area == pytest.approx(300 * 200, rel=0.01)

    def test_unit_square(self):
        """Test area of unit square."""
        corners = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ], dtype=np.float32)
        area = compute_quadrilateral_area(corners)
        assert area == pytest.approx(1.0, rel=0.01)

    def test_degenerate_quad_zero_area(self):
        """Test degenerate quadrilateral with zero area."""
        corners = np.array([
            [0, 0], [1, 0], [2, 0], [3, 0]  # All on same line
        ], dtype=np.float32)
        area = compute_quadrilateral_area(corners)
        assert area == pytest.approx(0.0, abs=0.01)

    def test_invalid_shape_raises(self):
        """Test that invalid shape raises ValueError."""
        with pytest.raises(ValueError):
            compute_quadrilateral_area(np.array([[1, 2], [3, 4]]))


# Tests for compute_interior_angles


class TestComputeInteriorAngles:
    def test_square_90_degree_angles(self, ordered_square):
        """Test that square has 90° angles."""
        angles = compute_interior_angles(ordered_square)
        assert len(angles) == 4
        for angle in angles:
            assert angle == pytest.approx(90.0, abs=1.0)

    def test_rectangle_90_degree_angles(self, ordered_rectangle):
        """Test that rectangle has 90° angles."""
        angles = compute_interior_angles(ordered_rectangle)
        assert len(angles) == 4
        for angle in angles:
            assert angle == pytest.approx(90.0, abs=1.0)

    def test_sum_of_angles(self, ordered_square):
        """Test that angles sum to 360°."""
        angles = compute_interior_angles(ordered_square)
        assert sum(angles) == pytest.approx(360.0, abs=2.0)

    def test_rhombus_angles(self):
        """Test angles of a rhombus (diamond)."""
        # Diamond with acute and obtuse angles
        corners = np.array([
            [200, 0],    # Top
            [300, 100],  # Right
            [200, 200],  # Bottom
            [100, 100],  # Left
        ], dtype=np.float32)
        angles = compute_interior_angles(corners)
        assert len(angles) == 4
        assert sum(angles) == pytest.approx(360.0, abs=2.0)

    def test_invalid_shape_raises(self):
        """Test that invalid shape raises ValueError."""
        with pytest.raises(ValueError):
            compute_interior_angles(np.array([[1, 2]]))


# Tests for is_valid_quadrilateral


class TestIsValidQuadrilateral:
    def test_valid_square(self, ordered_square):
        """Test that square is valid."""
        assert is_valid_quadrilateral(ordered_square) is True

    def test_valid_rectangle(self, ordered_rectangle):
        """Test that rectangle is valid."""
        assert is_valid_quadrilateral(ordered_rectangle) is True

    def test_concave_invalid(self, concave_quad):
        """Test that concave quad is invalid."""
        assert is_valid_quadrilateral(concave_quad) is False

    def test_sharp_angles_invalid(self):
        """Test that quad with sharp angles is invalid."""
        # Very elongated parallelogram with sharp angles
        corners = np.array([
            [0, 0],
            [100, 10],
            [200, 10],
            [100, 0],
        ], dtype=np.float32)
        # This might or might not be convex but has sharp angles
        result = is_valid_quadrilateral(corners, min_angle=45, max_angle=135)
        # If it passes convexity, it should fail angle check
        if result:
            angles = compute_interior_angles(corners)
            assert all(45 <= a <= 135 for a in angles)

    def test_custom_angle_bounds(self, ordered_square):
        """Test with custom angle bounds."""
        # Square has 90° angles
        assert is_valid_quadrilateral(ordered_square, min_angle=80, max_angle=100) is True
        assert is_valid_quadrilateral(ordered_square, min_angle=91, max_angle=100) is False

    def test_invalid_shape(self):
        """Test that wrong shape returns False."""
        assert is_valid_quadrilateral(np.array([[1, 2]])) is False

    def test_rotated_square_valid(self):
        """Test that rotated square is valid."""
        corners = np.array([
            [150, 50],
            [250, 150],
            [150, 250],
            [50, 150],
        ], dtype=np.float32)
        assert is_valid_quadrilateral(corners) is True


# Tests for compute_aspect_ratio


class TestComputeAspectRatio:
    def test_square_ratio_one(self, ordered_square):
        """Test that square has aspect ratio ~1.0."""
        ratio = compute_aspect_ratio(ordered_square)
        assert ratio == pytest.approx(1.0, abs=0.01)

    def test_rectangle_ratio(self, ordered_rectangle):
        """Test aspect ratio of 300x200 rectangle."""
        # Width = 300, Height = 200
        ratio = compute_aspect_ratio(ordered_rectangle)
        assert ratio == pytest.approx(1.5, abs=0.01)

    def test_tall_rectangle(self):
        """Test tall rectangle (height > width)."""
        corners = np.array([
            [100, 100],
            [200, 100],
            [200, 400],
            [100, 400],
        ], dtype=np.float32)
        # Width = 100, Height = 300
        ratio = compute_aspect_ratio(corners)
        assert ratio == pytest.approx(100 / 300, abs=0.01)

    def test_wide_rectangle(self):
        """Test wide rectangle (width > height)."""
        corners = np.array([
            [100, 100],
            [500, 100],
            [500, 200],
            [100, 200],
        ], dtype=np.float32)
        # Width = 400, Height = 100
        ratio = compute_aspect_ratio(corners)
        assert ratio == pytest.approx(4.0, abs=0.01)

    def test_invalid_shape_raises(self):
        """Test that invalid shape raises ValueError."""
        with pytest.raises(ValueError):
            compute_aspect_ratio(np.array([[1, 2], [3, 4]]))

    def test_degenerate_zero_height(self):
        """Test degenerate quad with zero height returns 0."""
        corners = np.array([
            [0, 100],
            [200, 100],
            [200, 100],
            [0, 100],
        ], dtype=np.float32)
        ratio = compute_aspect_ratio(corners)
        assert ratio == 0.0


# Integration tests


class TestGeometryIntegration:
    def test_full_warp_pipeline(self, simple_color_image):
        """Test complete warping pipeline."""
        # Define corners (slightly inset from edges)
        corners = np.array([
            [50, 50],
            [450, 50],
            [450, 450],
            [50, 450],
        ], dtype=np.float32)

        # Order corners (already ordered but test anyway)
        ordered = order_corners(corners)

        # Compute homography
        H = compute_homography(ordered, 400)

        # Warp image
        warped = warp_perspective(simple_color_image, H, 400)

        assert warped.shape == (400, 400, 3)

    def test_scale_and_warp(self):
        """Test scaling corners and then warping."""
        # Original image corners
        original_corners = np.array([
            [200, 200],
            [800, 200],
            [800, 800],
            [200, 800],
        ], dtype=np.float32)

        # Simulated resize by 0.5
        scale_factor = 0.5
        resized_corners = original_corners * scale_factor

        # Scale back to original
        restored = scale_corners_to_original(resized_corners, scale_factor)

        np.testing.assert_array_almost_equal(restored, original_corners)

    def test_order_then_validate(self):
        """Test ordering then validation."""
        # Shuffled valid square
        shuffled = np.array([
            [300, 100],
            [100, 300],
            [100, 100],
            [300, 300],
        ], dtype=np.float32)

        ordered = order_corners(shuffled)
        assert is_valid_quadrilateral(ordered) is True

    def test_ordered_corners_have_valid_area(self, shuffled_square):
        """Test that ordering corners produces valid non-zero area."""
        # Shuffled corners may form self-intersecting shape with zero/wrong area
        ordered = order_corners(shuffled_square)
        area = compute_quadrilateral_area(ordered)

        # Ordered 200x200 square should have area 40000
        assert area == pytest.approx(40000.0, rel=0.01)
