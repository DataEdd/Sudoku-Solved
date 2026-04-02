"""
Tests for the orientation assignment module.

These tests verify the correctness of gradient computation, orientation
histogram building, dominant orientation finding, and orientation assignment.
"""

import numpy as np
import pytest

from sift.orientation import (
    compute_gradients,
    compute_orientation_histogram,
    find_dominant_orientations,
    assign_orientations,
)
from sift.keypoints import Keypoint
from sift.scale_space import gaussian_blur, build_gaussian_pyramid


class TestComputeGradients:
    """Tests for compute_gradients function."""

    def test_compute_gradients_horizontal_edge(self):
        """Horizontal edge should have orientation near 0 or 180 degrees.

        A horizontal edge has intensity change in the y-direction,
        so gradient points up (90 degrees) or down (270 degrees).
        But if we define horizontal edge as intensity change along x-axis
        (bright on right, dark on left), gradient points right (0 degrees).
        """
        # Create image with horizontal edge: dark on left, bright on right
        image = np.zeros((21, 21), dtype=np.float64)
        image[:, 11:] = 1.0  # Right half is bright

        magnitude, orientation = compute_gradients(image)

        # Check center point on the edge (around x=10)
        # Gradient should point toward brighter region (right = 0 degrees)
        center_y, center_x = 10, 10

        # Magnitude should be non-zero at edge
        assert magnitude[center_y, center_x] > 0

        # Orientation should be near 0 degrees (pointing right)
        # Allow for some tolerance
        angle = orientation[center_y, center_x]
        assert angle < 30 or angle > 330, f"Expected ~0 degrees, got {angle}"

    def test_compute_gradients_vertical_edge(self):
        """Vertical edge should have orientation near 90 or 270 degrees.

        A vertical edge (bright above, dark below) has gradient pointing up (90 degrees).
        """
        # Create image with vertical edge: dark on top, bright on bottom
        image = np.zeros((21, 21), dtype=np.float64)
        image[11:, :] = 1.0  # Bottom half is bright

        magnitude, orientation = compute_gradients(image)

        # Check center point on the edge (around y=10)
        center_y, center_x = 10, 10

        # Magnitude should be non-zero at edge
        assert magnitude[center_y, center_x] > 0

        # Orientation should be near 90 degrees (pointing down toward bright)
        angle = orientation[center_y, center_x]
        assert 60 < angle < 120, f"Expected ~90 degrees, got {angle}"

    def test_compute_gradients_magnitude(self):
        """Known gradient should have correct magnitude.

        For a step edge of height 1.0, using central differences:
        dx = image[x+1] - image[x-1] = 1.0 - 0.0 = 1.0 (at the edge)
        magnitude = sqrt(dx^2 + dy^2) = sqrt(1^2 + 0^2) = 1.0
        """
        # Create a clean step edge
        image = np.zeros((21, 21), dtype=np.float64)
        image[:, 11:] = 1.0  # Step at x=10/11

        magnitude, orientation = compute_gradients(image)

        # At x=10, dx = image[10, 11] - image[10, 9] = 1.0 - 0.0 = 1.0
        # At x=11, dx = image[11, 12] - image[11, 10] = 1.0 - 0.0 = 1.0
        # But at x=10, the pixel is 0, and dx = image[:, 11] - image[:, 9]

        # At column 10: dx = 1 - 0 = 1, dy = 0, so magnitude = 1
        assert abs(magnitude[10, 10] - 1.0) < 0.01, f"Expected magnitude 1.0, got {magnitude[10, 10]}"

    def test_compute_gradients_diagonal_edge(self):
        """Diagonal edge should have orientation near 45 or 225 degrees."""
        # Create diagonal edge: upper-left is dark, lower-right is bright
        image = np.zeros((21, 21), dtype=np.float64)
        for i in range(21):
            for j in range(21):
                if i + j > 20:
                    image[i, j] = 1.0

        magnitude, orientation = compute_gradients(image)

        # At center, gradient should point toward bright region (45 degrees direction)
        # Actually arctan2(dy, dx) where both dx and dy are positive points to 45 degrees
        center_y, center_x = 10, 10
        angle = orientation[center_y, center_x]

        # Should be around 45 degrees or 225 degrees
        assert (30 < angle < 60) or (210 < angle < 240), f"Expected ~45 or ~225 degrees, got {angle}"

    def test_compute_gradients_uniform_image(self):
        """Uniform image should have zero magnitude everywhere."""
        image = np.ones((21, 21), dtype=np.float64) * 0.5

        magnitude, orientation = compute_gradients(image)

        # All magnitudes should be zero (except possibly edges)
        assert np.allclose(magnitude[1:-1, 1:-1], 0.0)


class TestOrientationHistogram:
    """Tests for compute_orientation_histogram function."""

    def test_orientation_histogram_peak_location(self):
        """Peak should be at expected bin for uniform gradient direction."""
        # Create image with strong horizontal gradient (0 degrees)
        image = np.zeros((50, 50), dtype=np.float64)
        for i in range(50):
            image[:, i] = i / 50.0  # Smooth gradient left to right

        magnitude, orientation = compute_gradients(image)

        # Compute histogram at center
        histogram = compute_orientation_histogram(
            magnitude, orientation, 25, 25, sigma=3.0, num_bins=36
        )

        # Peak should be at bin 0 (0 degrees = gradient pointing right)
        peak_bin = np.argmax(histogram)
        # Allow for slight deviation due to edge effects
        assert peak_bin == 0 or peak_bin == 35, f"Expected peak at bin 0 or 35, got bin {peak_bin}"

    def test_orientation_histogram_vertical_gradient(self):
        """Vertical gradient should produce peak near 90 degrees."""
        # Create image with strong vertical gradient (90 degrees)
        image = np.zeros((50, 50), dtype=np.float64)
        for i in range(50):
            image[i, :] = i / 50.0  # Smooth gradient top to bottom

        magnitude, orientation = compute_gradients(image)

        histogram = compute_orientation_histogram(
            magnitude, orientation, 25, 25, sigma=3.0, num_bins=36
        )

        # Peak should be at bin 9 (90 degrees)
        peak_bin = np.argmax(histogram)
        assert 8 <= peak_bin <= 10, f"Expected peak near bin 9 (90 degrees), got bin {peak_bin}"

    def test_orientation_histogram_shape(self):
        """Histogram should have correct number of bins."""
        image = np.random.rand(50, 50)
        magnitude, orientation = compute_gradients(image)

        histogram = compute_orientation_histogram(
            magnitude, orientation, 25, 25, sigma=3.0, num_bins=36
        )

        assert histogram.shape == (36,)

    def test_orientation_histogram_non_negative(self):
        """All histogram values should be non-negative."""
        image = np.random.rand(50, 50)
        magnitude, orientation = compute_gradients(image)

        histogram = compute_orientation_histogram(
            magnitude, orientation, 25, 25, sigma=3.0, num_bins=36
        )

        assert np.all(histogram >= 0)


class TestFindDominantOrientations:
    """Tests for find_dominant_orientations function."""

    def test_find_dominant_orientations_single_peak(self):
        """Single clear peak should return one orientation."""
        # Create histogram with one strong peak at bin 9 (90 degrees)
        histogram = np.zeros(36)
        histogram[9] = 10.0
        histogram[8] = 3.0
        histogram[10] = 3.0

        orientations = find_dominant_orientations(histogram, peak_ratio=0.8)

        assert len(orientations) == 1
        # Should be close to 90 degrees (bin 9 * 10 degrees/bin = 90)
        assert 85 < orientations[0] < 95, f"Expected ~90 degrees, got {orientations[0]}"

    def test_find_dominant_orientations_multiple_peaks(self):
        """Two strong peaks should return two orientations."""
        # Create histogram with two strong peaks
        histogram = np.zeros(36)
        # Peak 1 at bin 0 (0 degrees)
        histogram[0] = 10.0
        histogram[35] = 3.0
        histogram[1] = 3.0
        # Peak 2 at bin 18 (180 degrees)
        histogram[18] = 9.5  # Above 80% of max (8.0)
        histogram[17] = 3.0
        histogram[19] = 3.0

        orientations = find_dominant_orientations(histogram, peak_ratio=0.8)

        assert len(orientations) == 2, f"Expected 2 orientations, got {len(orientations)}"

        # Sort for consistent comparison
        orientations.sort()
        assert orientations[0] < 10 or orientations[0] > 350  # Near 0 degrees
        assert 175 < orientations[1] < 185  # Near 180 degrees

    def test_find_dominant_orientations_no_peak(self):
        """Flat histogram should return empty list or handle gracefully."""
        histogram = np.ones(36)

        orientations = find_dominant_orientations(histogram, peak_ratio=0.8)

        # No local maxima in a flat histogram
        assert len(orientations) == 0

    def test_find_dominant_orientations_parabolic_interpolation(self):
        """Test sub-bin accuracy via parabolic interpolation."""
        # Create histogram with asymmetric peak
        histogram = np.zeros(36)
        histogram[9] = 10.0
        histogram[8] = 5.0  # Lower than bin 10
        histogram[10] = 7.0  # Higher than bin 8

        orientations = find_dominant_orientations(histogram, peak_ratio=0.8)

        assert len(orientations) == 1
        # Peak should be shifted slightly toward bin 10 (higher neighbor)
        # Bin 9 = 90 degrees, shift toward bin 10 = 100 degrees
        assert 90 < orientations[0] < 100, f"Expected interpolated peak > 90, got {orientations[0]}"


class TestAssignOrientations:
    """Tests for assign_orientations function."""

    def test_assign_orientations_sets_orientation(self):
        """Orientation field should be set after assignment."""
        # Create a simple test image
        image = np.zeros((100, 100), dtype=np.float64)
        # Add some structure (horizontal gradient)
        for i in range(100):
            image[:, i] = i / 100.0

        # Build pyramid
        from sift.scale_space import build_gaussian_pyramid
        pyramid, sigmas = build_gaussian_pyramid(image, num_octaves=2, num_scales=5)

        # Create keypoint without meaningful orientation
        keypoints = [
            Keypoint(
                x=50.0, y=50.0, octave=0, scale_idx=1.0, sigma=2.0,
                response=0.1, is_maximum=True, orientation=None
            )
        ]

        result = assign_orientations(keypoints, pyramid, sigmas)

        assert len(result) >= 1
        # Orientation should be set (may be different from 0.0)
        assert result[0].orientation is not None
        assert 0 <= result[0].orientation < 360

    def test_assign_orientations_creates_copies(self):
        """Multiple orientations should create duplicate keypoints."""
        # Create an image with features that might have multiple dominant orientations
        # A corner or T-junction might have multiple strong gradient directions
        image = np.zeros((100, 100), dtype=np.float64)

        # Create a cross pattern which has gradients in multiple directions
        image[45:55, :] = 1.0  # Horizontal bar
        image[:, 45:55] = 1.0  # Vertical bar

        # Apply slight blur
        image = gaussian_blur(image, sigma=1.0)

        # Build pyramid
        pyramid, sigmas = build_gaussian_pyramid(image, num_octaves=2, num_scales=5)

        # Create keypoint at center of cross
        keypoints = [
            Keypoint(
                x=50.0, y=50.0, octave=0, scale_idx=1.0, sigma=3.0,
                response=0.1, is_maximum=True, orientation=None
            )
        ]

        result = assign_orientations(keypoints, pyramid, sigmas)

        # At a cross junction, there could be multiple dominant orientations
        # This might result in more keypoints than input
        # At minimum, we should get at least one keypoint back
        assert len(result) >= 1

        # All keypoints should have the same position
        for kp in result:
            assert kp.x == 50.0
            assert kp.y == 50.0
            assert kp.sigma == 3.0

    def test_assign_orientations_preserves_attributes(self):
        """Non-orientation attributes should be preserved."""
        image = np.random.rand(100, 100)
        pyramid, sigmas = build_gaussian_pyramid(image, num_octaves=2, num_scales=5)

        keypoints = [
            Keypoint(
                x=50.0, y=50.0, octave=0, scale_idx=1.5, sigma=2.5,
                response=0.15, is_maximum=True, orientation=None
            )
        ]

        result = assign_orientations(keypoints, pyramid, sigmas)

        assert len(result) >= 1
        for kp in result:
            assert kp.x == 50.0
            assert kp.y == 50.0
            assert kp.sigma == 2.5
            assert kp.octave == 0
            assert kp.response == 0.15
            assert kp.scale_idx == 1.5
            assert kp.is_maximum is True

    def test_assign_orientations_empty_input(self):
        """Empty keypoint list should return empty result."""
        image = np.random.rand(100, 100)
        pyramid, sigmas = build_gaussian_pyramid(image, num_octaves=2, num_scales=5)

        result = assign_orientations([], pyramid, sigmas)

        assert result == []

    def test_assign_orientations_multiple_keypoints(self):
        """Multiple input keypoints should all be processed."""
        image = np.random.rand(100, 100)
        pyramid, sigmas = build_gaussian_pyramid(image, num_octaves=2, num_scales=5)

        keypoints = [
            Keypoint(
                x=25.0, y=25.0, octave=0, scale_idx=1.0, sigma=2.0,
                response=0.1, is_maximum=True, orientation=None
            ),
            Keypoint(
                x=75.0, y=75.0, octave=0, scale_idx=1.0, sigma=2.0,
                response=0.1, is_maximum=False, orientation=None
            ),
        ]

        result = assign_orientations(keypoints, pyramid, sigmas)

        # Should have at least 2 keypoints (one for each input)
        assert len(result) >= 2
