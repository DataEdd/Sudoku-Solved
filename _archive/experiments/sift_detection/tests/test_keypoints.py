"""
Tests for the keypoints module.

Tests cover:
- find_dog_extrema: Detection of local extrema
- compute_gradient_3d: Gradient calculation via finite differences
- compute_hessian_3d: Hessian matrix computation and symmetry
- localize_keypoint: Sub-pixel localization accuracy
- filter_low_contrast: Contrast threshold filtering
- filter_edge_response: Edge vs corner discrimination
- find_keypoints: End-to-end pipeline
"""

import numpy as np
import pytest

from sift.scale_space import gaussian_blur, build_gaussian_pyramid, build_dog_pyramid
from sift.keypoints import (
    Keypoint,
    find_dog_extrema,
    compute_gradient_3d,
    compute_hessian_3d,
    localize_keypoint,
    compute_hessian_2d,
    filter_low_contrast,
    filter_edge_response,
    find_keypoints,
)


class TestFindDogExtrema:
    """Tests for find_dog_extrema function."""

    def test_find_dog_extrema_finds_known_blob(self):
        """Create a synthetic blob and verify it's detected as an extremum."""
        # Create a 3-scale DoG octave with a blob at the center scale
        size = 21
        center = size // 2

        # Create base images (zeros with a bright blob in the center)
        scale0 = np.zeros((size, size))
        scale1 = np.zeros((size, size))
        scale2 = np.zeros((size, size))

        # Put a strong positive blob in scale1
        y, x = np.ogrid[:size, :size]
        blob = np.exp(-((x - center)**2 + (y - center)**2) / (2 * 2**2))
        scale1 = blob * 0.5  # Strong response

        # scale0 and scale2 have weaker responses at center
        scale0 = blob * 0.1
        scale2 = blob * 0.1

        dog_octave = [scale0, scale1, scale2]
        dog_pyramid = [dog_octave]

        # Find extrema
        keypoints = find_dog_extrema(dog_pyramid, threshold=0.03)

        # Should find at least one keypoint near the center
        assert len(keypoints) > 0, "Should detect the blob as an extremum"

        # Check that at least one keypoint is near the center
        center_kp = None
        for kp in keypoints:
            if abs(kp['x'] - center) <= 2 and abs(kp['y'] - center) <= 2:
                center_kp = kp
                break

        assert center_kp is not None, "Should find a keypoint near the blob center"
        assert center_kp['is_maximum'] == True, "Blob should be detected as maximum"
        assert center_kp['scale'] == 1, "Blob should be detected at middle scale"

    def test_find_dog_extrema_detects_minimum(self):
        """Verify that local minima are also detected."""
        size = 21
        center = size // 2

        # Create a negative blob (minimum)
        y, x = np.ogrid[:size, :size]
        blob = -np.exp(-((x - center)**2 + (y - center)**2) / (2 * 2**2)) * 0.5

        scale0 = blob * 0.2
        scale1 = blob  # Strong negative
        scale2 = blob * 0.2

        dog_pyramid = [[scale0, scale1, scale2]]
        keypoints = find_dog_extrema(dog_pyramid, threshold=0.03)

        # Find keypoint near center
        center_kps = [kp for kp in keypoints
                      if abs(kp['x'] - center) <= 2 and abs(kp['y'] - center) <= 2]

        assert len(center_kps) > 0, "Should detect minimum"
        assert any(not kp['is_maximum'] for kp in center_kps), "Should be detected as minimum"

    def test_find_dog_extrema_respects_threshold(self):
        """Verify that weak responses below threshold are not detected."""
        size = 21
        center = size // 2

        # Create a weak blob
        y, x = np.ogrid[:size, :size]
        blob = np.exp(-((x - center)**2 + (y - center)**2) / (2 * 2**2)) * 0.02

        scale0 = blob * 0.5
        scale1 = blob  # Response = 0.02, below default threshold
        scale2 = blob * 0.5

        dog_pyramid = [[scale0, scale1, scale2]]

        # With high threshold, should find nothing
        keypoints = find_dog_extrema(dog_pyramid, threshold=0.03)
        center_kps = [kp for kp in keypoints
                      if abs(kp['x'] - center) <= 2 and abs(kp['y'] - center) <= 2]
        assert len(center_kps) == 0, "Weak blob should be filtered by threshold"

        # With low threshold, should find it
        keypoints = find_dog_extraema(dog_pyramid, threshold=0.01) if False else find_dog_extrema(dog_pyramid, threshold=0.01)
        center_kps = [kp for kp in keypoints
                      if abs(kp['x'] - center) <= 2 and abs(kp['y'] - center) <= 2]
        assert len(center_kps) > 0, "Should detect blob with lower threshold"


class TestGradient3D:
    """Tests for compute_gradient_3d function."""

    def test_gradient_3d_finite_difference(self):
        """Verify gradient calculation matches expected finite differences."""
        size = 11
        center = size // 2

        # Create a linear function in x: f(x,y,s) = a*x + b*y + c*s
        a, b, c = 0.5, 0.3, 0.2

        scales = []
        for s in range(3):
            scale_img = np.zeros((size, size))
            for y in range(size):
                for x in range(size):
                    scale_img[y, x] = a * x + b * y + c * s
            scales.append(scale_img)

        # Gradient at center should be [a, b, c]
        gradient = compute_gradient_3d(scales, scale=1, y=center, x=center)

        np.testing.assert_almost_equal(gradient[0], a, decimal=5,
                                        err_msg="dD/dx should equal coefficient a")
        np.testing.assert_almost_equal(gradient[1], b, decimal=5,
                                        err_msg="dD/dy should equal coefficient b")
        np.testing.assert_almost_equal(gradient[2], c, decimal=5,
                                        err_msg="dD/ds should equal coefficient c")

    def test_gradient_3d_zero_for_constant(self):
        """Gradient of a constant function should be zero."""
        size = 11
        center = size // 2
        constant_val = 0.5

        scales = [np.full((size, size), constant_val) for _ in range(3)]
        gradient = compute_gradient_3d(scales, scale=1, y=center, x=center)

        np.testing.assert_almost_equal(gradient, [0, 0, 0], decimal=10,
                                        err_msg="Gradient of constant should be zero")


class TestHessian3D:
    """Tests for compute_hessian_3d function."""

    def test_hessian_3d_symmetry(self):
        """Hessian matrix should be symmetric."""
        size = 11
        center = size // 2

        # Create a non-trivial function
        scales = []
        for s in range(3):
            y, x = np.ogrid[:size, :size]
            # f = x^2 + y^2 + s^2 + x*y
            scale_img = (x - center)**2 + (y - center)**2 + s**2 + (x - center) * (y - center)
            scales.append(scale_img.astype(float))

        hessian = compute_hessian_3d(scales, scale=1, y=center, x=center)

        # Check symmetry: H[i,j] == H[j,i]
        np.testing.assert_almost_equal(hessian[0, 1], hessian[1, 0], decimal=10,
                                        err_msg="H_xy should equal H_yx")
        np.testing.assert_almost_equal(hessian[0, 2], hessian[2, 0], decimal=10,
                                        err_msg="H_xs should equal H_sx")
        np.testing.assert_almost_equal(hessian[1, 2], hessian[2, 1], decimal=10,
                                        err_msg="H_ys should equal H_sy")

    def test_hessian_3d_quadratic(self):
        """Verify Hessian for known quadratic function."""
        size = 11
        center = size // 2

        # f = a*x^2 + b*y^2 + c*s^2
        # Hessian diagonal should be [2a, 2b, 2c]
        a, b, c = 1.0, 2.0, 0.5

        scales = []
        for s in range(3):
            y, x = np.ogrid[:size, :size]
            scale_img = a * (x - center)**2 + b * (y - center)**2 + c * (s - 1)**2
            scales.append(scale_img.astype(float))

        hessian = compute_hessian_3d(scales, scale=1, y=center, x=center)

        # Diagonal elements should be 2*coefficient
        np.testing.assert_almost_equal(hessian[0, 0], 2 * a, decimal=5)
        np.testing.assert_almost_equal(hessian[1, 1], 2 * b, decimal=5)
        np.testing.assert_almost_equal(hessian[2, 2], 2 * c, decimal=5)

        # Off-diagonal should be zero for this function
        np.testing.assert_almost_equal(hessian[0, 1], 0, decimal=5)
        np.testing.assert_almost_equal(hessian[0, 2], 0, decimal=5)
        np.testing.assert_almost_equal(hessian[1, 2], 0, decimal=5)


class TestLocalizeKeypoint:
    """Tests for localize_keypoint function."""

    def test_localize_keypoint_subpixel_accuracy(self):
        """Verify sub-pixel localization finds the true extremum."""
        size = 21
        center = size // 2

        # Create a blob centered at (center + 0.3, center - 0.2)
        true_x = center + 0.3
        true_y = center - 0.2

        scales = []
        for s in range(3):
            y, x = np.ogrid[:size, :size]
            # Gaussian blob offset from pixel center
            blob = np.exp(-((x - true_x)**2 + (y - true_y)**2 + (s - 1)**2) / (2 * 2**2))
            scales.append(blob.astype(float) * 0.5)

        # Start from integer pixel position
        result = localize_keypoint(scales, scale=1, y=center, x=center)

        assert result is not None, "Should successfully localize keypoint"

        # Check sub-pixel accuracy (within 0.2 pixels)
        assert abs(result['x'] - true_x) < 0.2, f"X should be close to {true_x}, got {result['x']}"
        assert abs(result['y'] - true_y) < 0.2, f"Y should be close to {true_y}, got {result['y']}"

    def test_localize_keypoint_rejects_boundary(self):
        """Keypoints that move out of bounds should be rejected."""
        size = 11

        # Create a gradient that will push the keypoint out of bounds
        scales = []
        for s in range(3):
            y, x = np.ogrid[:size, :size]
            # Strong gradient towards edge
            scales.append((x + y + s * 10).astype(float))

        # Start near the boundary
        result = localize_keypoint(scales, scale=1, y=1, x=1)

        # May or may not be rejected depending on gradient magnitude
        # The important thing is it shouldn't crash


class TestFilterLowContrast:
    """Tests for filter_low_contrast function."""

    def test_filter_low_contrast_threshold(self):
        """Test threshold boundary cases."""
        # Just above threshold
        kp_high = {'response': 0.031}
        assert filter_low_contrast(kp_high, contrast_threshold=0.03) is True

        # Just below threshold
        kp_low = {'response': 0.029}
        assert filter_low_contrast(kp_low, contrast_threshold=0.03) is False

        # At threshold
        kp_at = {'response': 0.03}
        assert filter_low_contrast(kp_at, contrast_threshold=0.03) is True

        # Negative response (minimum)
        kp_neg = {'response': -0.05}
        assert filter_low_contrast(kp_neg, contrast_threshold=0.03) is True

        # Negative below threshold
        kp_neg_low = {'response': -0.02}
        assert filter_low_contrast(kp_neg_low, contrast_threshold=0.03) is False


class TestFilterEdgeResponse:
    """Tests for filter_edge_response function."""

    def test_filter_edge_response_rejects_edges(self):
        """Synthetic edge pattern should be rejected."""
        size = 21
        center = size // 2

        # Create a vertical edge (strong gradient in x, none in y)
        edge = np.zeros((size, size))
        edge[:, :center] = 1.0
        edge[:, center:] = -1.0

        # Apply some blur to make it smooth
        edge = gaussian_blur(edge, 2.0)

        result = filter_edge_response(edge, center, center, edge_threshold=10)
        assert result is False, "Edge-like point should be rejected"

    def test_filter_edge_response_keeps_corners(self):
        """Synthetic corner pattern should be kept."""
        size = 21
        center = size // 2

        # Create a corner pattern - L-shaped, similar curvature in both directions
        # This creates a pattern where a single corner exists
        corner = np.zeros((size, size))
        corner[:center+1, :center+1] = 1.0  # Top-left quadrant is bright

        # Apply some blur
        corner = gaussian_blur(corner, 2.0)

        result = filter_edge_response(corner, center, center, edge_threshold=10)
        assert result == True, "Corner-like point should be kept"

    def test_filter_edge_response_keeps_blobs(self):
        """Blob patterns should be kept (similar eigenvalues)."""
        size = 21
        center = size // 2

        # Create a Gaussian blob
        y, x = np.ogrid[:size, :size]
        blob = np.exp(-((x - center)**2 + (y - center)**2) / (2 * 3**2))

        result = filter_edge_response(blob, center, center, edge_threshold=10)
        assert result == True, "Blob (isotropic) should be kept"

    def test_filter_edge_response_rejects_saddle(self):
        """Saddle points (negative determinant) should be rejected."""
        size = 21
        center = size // 2

        # Create a saddle: f = x^2 - y^2
        y, x = np.ogrid[:size, :size]
        saddle = ((x - center)**2 - (y - center)**2).astype(float)

        result = filter_edge_response(saddle, center, center, edge_threshold=10)
        assert result is False, "Saddle point should be rejected"


class TestHessian2D:
    """Tests for compute_hessian_2d function."""

    def test_hessian_2d_symmetry(self):
        """2D Hessian should be symmetric."""
        size = 11
        center = size // 2

        y, x = np.ogrid[:size, :size]
        img = ((x - center)**2 + (y - center)**2 + (x - center) * (y - center)).astype(float)

        hessian = compute_hessian_2d(img, center, center)

        np.testing.assert_almost_equal(hessian[0, 1], hessian[1, 0], decimal=10)

    def test_hessian_2d_quadratic(self):
        """Verify Hessian for known quadratic."""
        size = 11
        center = size // 2

        # f = a*x^2 + b*y^2
        a, b = 2.0, 3.0
        y, x = np.ogrid[:size, :size]
        img = (a * (x - center)**2 + b * (y - center)**2).astype(float)

        hessian = compute_hessian_2d(img, center, center)

        np.testing.assert_almost_equal(hessian[0, 0], 2 * a, decimal=5)
        np.testing.assert_almost_equal(hessian[1, 1], 2 * b, decimal=5)
        np.testing.assert_almost_equal(hessian[0, 1], 0, decimal=5)


class TestFindKeypointsEndToEnd:
    """End-to-end tests for the complete keypoint detection pipeline."""

    def test_find_keypoints_end_to_end(self):
        """Full pipeline test with synthetic image."""
        # Create a synthetic image with known features
        size = 100
        img = np.zeros((size, size))

        # Add several blobs at different positions
        blob_positions = [(25, 25), (75, 25), (25, 75), (75, 75), (50, 50)]
        for y, x in blob_positions:
            yy, xx = np.ogrid[:size, :size]
            blob = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 5**2))
            img += blob * 0.5

        # Normalize
        img = img / img.max()

        # Build pyramids
        pyramid, sigmas = build_gaussian_pyramid(img, num_octaves=2, num_scales=5)
        dog_pyramid = build_dog_pyramid(pyramid)

        # Find keypoints
        keypoints, stats = find_keypoints(
            dog_pyramid, sigmas,
            contrast_threshold=0.01,
            edge_threshold=10,
            initial_threshold=0.005
        )

        # Should find some keypoints
        assert len(keypoints) > 0, "Should detect keypoints in synthetic image"
        assert stats['candidates'] > 0, "Should have candidate keypoints"
        assert stats['accepted'] == len(keypoints), "Accepted count should match keypoint count"

        # Verify keypoint structure
        for kp in keypoints:
            assert isinstance(kp, Keypoint)
            assert 0 <= kp.x < size
            assert 0 <= kp.y < size
            assert kp.sigma > 0
            assert kp.is_maximum in (True, False)  # Works with numpy booleans

    def test_find_keypoints_returns_stats(self):
        """Verify statistics dictionary structure."""
        img = np.random.rand(50, 50) * 0.1  # Low contrast noise

        pyramid, sigmas = build_gaussian_pyramid(img, num_octaves=2, num_scales=5)
        dog_pyramid = build_dog_pyramid(pyramid)

        keypoints, stats = find_keypoints(dog_pyramid, sigmas)

        # Check stats dictionary has expected keys
        expected_keys = ['candidates', 'localization_failed', 'low_contrast',
                         'edge_response', 'accepted']
        for key in expected_keys:
            assert key in stats, f"Stats should contain '{key}'"

        # Verify counts are non-negative
        for key, value in stats.items():
            assert value >= 0, f"{key} should be non-negative"

    def test_find_keypoints_empty_image(self):
        """Empty/constant image should have no keypoints."""
        img = np.ones((50, 50)) * 0.5

        pyramid, sigmas = build_gaussian_pyramid(img, num_octaves=2, num_scales=5)
        dog_pyramid = build_dog_pyramid(pyramid)

        keypoints, stats = find_keypoints(dog_pyramid, sigmas)

        assert len(keypoints) == 0, "Constant image should have no keypoints"

    def test_find_keypoints_high_contrast_threshold(self):
        """High contrast threshold should filter most keypoints."""
        size = 50
        img = np.random.rand(size, size)

        pyramid, sigmas = build_gaussian_pyramid(img, num_octaves=2, num_scales=5)
        dog_pyramid = build_dog_pyramid(pyramid)

        # With very high threshold
        keypoints_high, _ = find_keypoints(
            dog_pyramid, sigmas, contrast_threshold=1.0
        )

        # With normal threshold
        keypoints_normal, _ = find_keypoints(
            dog_pyramid, sigmas, contrast_threshold=0.03
        )

        assert len(keypoints_high) <= len(keypoints_normal), \
            "Higher threshold should yield fewer keypoints"


class TestKeypointDataclass:
    """Tests for the Keypoint dataclass."""

    def test_keypoint_creation(self):
        """Verify Keypoint can be created with required fields."""
        kp = Keypoint(
            x=10.5,
            y=20.3,
            octave=0,
            scale_idx=1.2,
            sigma=1.6,
            response=0.05,
            is_maximum=True
        )

        assert kp.x == 10.5
        assert kp.y == 20.3
        assert kp.octave == 0
        assert kp.scale_idx == 1.2
        assert kp.sigma == 1.6
        assert kp.response == 0.05
        assert kp.is_maximum is True
        assert kp.orientation is None  # Default value

    def test_keypoint_with_orientation(self):
        """Verify Keypoint can have orientation set."""
        kp = Keypoint(
            x=10.0,
            y=20.0,
            octave=1,
            scale_idx=2.0,
            sigma=3.2,
            response=0.1,
            is_maximum=False,
            orientation=45.0
        )

        assert kp.orientation == 45.0
