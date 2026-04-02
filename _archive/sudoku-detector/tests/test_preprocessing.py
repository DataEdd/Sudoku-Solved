"""Tests for preprocessing module."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from src.config import DetectionConfig
from src.preprocessing import (
    adaptive_threshold,
    apply_clahe,
    apply_morphological,
    get_edges,
    load_and_validate,
    preprocess_for_contour,
    preprocess_for_hough,
    resize_image,
    to_grayscale,
)


# Fixtures


@pytest.fixture
def sample_bgr_image():
    """Create a sample BGR image for testing."""
    return np.random.randint(0, 256, (500, 400, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image for testing."""
    return np.random.randint(0, 256, (500, 400), dtype=np.uint8)


@pytest.fixture
def sample_large_image():
    """Create a large image that needs resizing."""
    return np.random.randint(0, 256, (2000, 1500, 3), dtype=np.uint8)


@pytest.fixture
def sample_small_image():
    """Create a small image below minimum size."""
    return np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)


@pytest.fixture
def config():
    """Create default detection configuration."""
    return DetectionConfig()


@pytest.fixture
def grid_like_image():
    """Create an image with grid-like pattern for threshold testing."""
    img = np.ones((300, 300), dtype=np.uint8) * 200
    # Draw grid lines (dark lines on light background)
    for i in range(0, 300, 30):
        img[i : i + 2, :] = 50  # Horizontal lines
        img[:, i : i + 2] = 50  # Vertical lines
    return img


# Tests for load_and_validate


class TestLoadAndValidate:
    def test_load_valid_image(self, sample_bgr_image):
        """Test loading a valid image file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, sample_bgr_image)
            try:
                result = load_and_validate(f.name, min_size=200)
                assert result is not None
                assert result.shape == sample_bgr_image.shape
            finally:
                os.unlink(f.name)

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        result = load_and_validate("/nonexistent/path/image.png")
        assert result is None

    def test_load_invalid_file(self):
        """Test loading a file that isn't an image."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not an image")
            f.flush()
            try:
                result = load_and_validate(f.name)
                assert result is None
            finally:
                os.unlink(f.name)

    def test_load_image_too_small(self, sample_small_image):
        """Test rejecting images below minimum size."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, sample_small_image)
            try:
                result = load_and_validate(f.name, min_size=200)
                assert result is None
            finally:
                os.unlink(f.name)

    def test_load_image_at_minimum_size(self):
        """Test accepting images exactly at minimum size."""
        img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, img)
            try:
                result = load_and_validate(f.name, min_size=200)
                assert result is not None
            finally:
                os.unlink(f.name)

    def test_custom_min_size(self, sample_bgr_image):
        """Test with custom minimum size threshold."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, sample_bgr_image)
            try:
                # Should fail with high min_size
                result = load_and_validate(f.name, min_size=600)
                assert result is None

                # Should pass with low min_size
                result = load_and_validate(f.name, min_size=100)
                assert result is not None
            finally:
                os.unlink(f.name)


# Tests for resize_image


class TestResizeImage:
    def test_no_resize_needed(self, sample_bgr_image):
        """Test that small images are not resized."""
        resized, scale = resize_image(sample_bgr_image, target_size=1000)
        assert scale == 1.0
        assert resized.shape == sample_bgr_image.shape

    def test_resize_large_image(self, sample_large_image):
        """Test resizing large images."""
        resized, scale = resize_image(sample_large_image, target_size=1000)
        assert scale < 1.0
        assert max(resized.shape[:2]) == 1000
        # Check aspect ratio preserved
        orig_aspect = sample_large_image.shape[1] / sample_large_image.shape[0]
        new_aspect = resized.shape[1] / resized.shape[0]
        assert abs(orig_aspect - new_aspect) < 0.01

    def test_resize_preserves_channels(self, sample_large_image):
        """Test that color channels are preserved after resize."""
        resized, _ = resize_image(sample_large_image, target_size=500)
        assert len(resized.shape) == 3
        assert resized.shape[2] == 3

    def test_resize_grayscale(self, sample_grayscale_image):
        """Test resizing grayscale images."""
        large_gray = np.random.randint(0, 256, (1500, 1200), dtype=np.uint8)
        resized, scale = resize_image(large_gray, target_size=1000)
        assert max(resized.shape[:2]) == 1000
        assert len(resized.shape) == 2

    def test_scale_factor_accuracy(self):
        """Test that scale factor is calculated correctly."""
        img = np.zeros((2000, 1000, 3), dtype=np.uint8)
        resized, scale = resize_image(img, target_size=1000)
        assert scale == 0.5
        assert resized.shape[0] == 1000
        assert resized.shape[1] == 500

    def test_returns_copy_when_no_resize(self, sample_bgr_image):
        """Test that a copy is returned even when no resize is needed."""
        resized, _ = resize_image(sample_bgr_image, target_size=1000)
        # Modify original shouldn't affect result
        original_val = resized[0, 0, 0]
        sample_bgr_image[0, 0, 0] = (sample_bgr_image[0, 0, 0] + 100) % 256
        assert resized[0, 0, 0] == original_val


# Tests for to_grayscale


class TestToGrayscale:
    def test_bgr_to_grayscale(self, sample_bgr_image):
        """Test converting BGR image to grayscale."""
        gray = to_grayscale(sample_bgr_image)
        assert len(gray.shape) == 2
        assert gray.shape == sample_bgr_image.shape[:2]

    def test_already_grayscale(self, sample_grayscale_image):
        """Test that already-gray images are handled correctly."""
        gray = to_grayscale(sample_grayscale_image)
        assert len(gray.shape) == 2
        assert gray.shape == sample_grayscale_image.shape

    def test_single_channel_3d(self):
        """Test handling 3D array with single channel."""
        img = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)
        gray = to_grayscale(img)
        assert len(gray.shape) == 2
        assert gray.shape == (100, 100)

    def test_returns_copy(self, sample_grayscale_image):
        """Test that a copy is returned for already-gray images."""
        gray = to_grayscale(sample_grayscale_image)
        original_val = gray[0, 0]
        sample_grayscale_image[0, 0] = (sample_grayscale_image[0, 0] + 100) % 256
        assert gray[0, 0] == original_val


# Tests for apply_clahe


class TestApplyCLAHE:
    def test_output_shape(self, sample_grayscale_image):
        """Test that CLAHE preserves image dimensions."""
        result = apply_clahe(sample_grayscale_image)
        assert result.shape == sample_grayscale_image.shape

    def test_output_dtype(self, sample_grayscale_image):
        """Test that CLAHE preserves data type."""
        result = apply_clahe(sample_grayscale_image)
        assert result.dtype == np.uint8

    def test_custom_parameters(self, sample_grayscale_image):
        """Test CLAHE with custom parameters."""
        result = apply_clahe(
            sample_grayscale_image, clip_limit=4.0, tile_size=(16, 16)
        )
        assert result.shape == sample_grayscale_image.shape

    def test_enhances_contrast(self):
        """Test that CLAHE increases contrast in low-contrast image."""
        # Create low contrast image
        low_contrast = np.full((100, 100), 128, dtype=np.uint8)
        low_contrast[40:60, 40:60] = 130  # Small bright region

        result = apply_clahe(low_contrast, clip_limit=2.0)
        # CLAHE should increase the range
        original_range = low_contrast.max() - low_contrast.min()
        result_range = result.max() - result.min()
        assert result_range >= original_range


# Tests for adaptive_threshold


class TestAdaptiveThreshold:
    def test_output_is_binary(self, sample_grayscale_image):
        """Test that output is binary (only 0 and 255)."""
        result = adaptive_threshold(sample_grayscale_image)
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)

    def test_output_shape(self, sample_grayscale_image):
        """Test that output shape matches input."""
        result = adaptive_threshold(sample_grayscale_image)
        assert result.shape == sample_grayscale_image.shape

    def test_grid_lines_white(self, grid_like_image):
        """Test that dark grid lines become white in output."""
        result = adaptive_threshold(grid_like_image, block_size=11, c=2)
        # Check that line regions have white pixels
        # Line at row 0-1 should have white pixels
        assert result[1, 150] == 255  # Horizontal line
        assert result[150, 1] == 255  # Vertical line

    def test_custom_parameters(self, sample_grayscale_image):
        """Test with custom block_size and c values."""
        result = adaptive_threshold(sample_grayscale_image, block_size=15, c=5)
        assert result.shape == sample_grayscale_image.shape
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)


# Tests for apply_morphological


class TestApplyMorphological:
    def test_output_shape(self):
        """Test that morphological closing preserves shape."""
        binary = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
        result = apply_morphological(binary, kernel_size=3)
        assert result.shape == binary.shape

    def test_fills_small_gaps(self):
        """Test that small gaps in lines are filled."""
        # Create line with small gap
        binary = np.zeros((100, 100), dtype=np.uint8)
        binary[50, 20:45] = 255  # Line segment 1
        binary[50, 48:80] = 255  # Line segment 2 (gap of 3 pixels)

        result = apply_morphological(binary, kernel_size=5)
        # Gap should be filled
        assert result[50, 46] == 255

    def test_custom_kernel_size(self):
        """Test with different kernel sizes."""
        binary = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
        result3 = apply_morphological(binary, kernel_size=3)
        result5 = apply_morphological(binary, kernel_size=5)
        # Larger kernel should fill more gaps
        assert result3.shape == result5.shape


# Tests for get_edges


class TestGetEdges:
    def test_output_is_binary(self, sample_grayscale_image):
        """Test that edge output is binary."""
        edges = get_edges(sample_grayscale_image)
        unique_values = np.unique(edges)
        assert all(v in [0, 255] for v in unique_values)

    def test_output_shape(self, sample_grayscale_image):
        """Test that output shape matches input."""
        edges = get_edges(sample_grayscale_image)
        assert edges.shape == sample_grayscale_image.shape

    def test_detects_edges(self):
        """Test that edges are detected at boundaries."""
        # Create image with sharp edge
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255  # Sharp vertical edge at column 50

        edges = get_edges(img, low=50, high=150)
        # Should have edge pixels near column 50
        edge_cols = np.where(edges.sum(axis=0) > 0)[0]
        assert len(edge_cols) > 0
        assert any(45 <= c <= 55 for c in edge_cols)

    def test_custom_parameters(self, sample_grayscale_image):
        """Test with custom Canny thresholds."""
        edges_low = get_edges(sample_grayscale_image, low=30, high=100)
        edges_high = get_edges(sample_grayscale_image, low=100, high=200)
        # Lower thresholds should detect more edges
        assert edges_low.sum() >= edges_high.sum()

    def test_blur_size(self, sample_grayscale_image):
        """Test with different blur sizes."""
        edges_small = get_edges(sample_grayscale_image, blur_size=3)
        edges_large = get_edges(sample_grayscale_image, blur_size=7)
        # Both should produce valid output
        assert edges_small.shape == sample_grayscale_image.shape
        assert edges_large.shape == sample_grayscale_image.shape


# Tests for preprocess_for_contour


class TestPreprocessForContour:
    def test_returns_tuple(self, sample_bgr_image, config):
        """Test that function returns correct tuple structure."""
        result = preprocess_for_contour(sample_bgr_image, config)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_binary_output(self, sample_bgr_image, config):
        """Test that first output is binary image."""
        binary, _, _ = preprocess_for_contour(sample_bgr_image, config)
        unique_values = np.unique(binary)
        assert all(v in [0, 255] for v in unique_values)

    def test_resized_color_output(self, sample_large_image, config):
        """Test that second output is resized color image."""
        _, resized, scale = preprocess_for_contour(sample_large_image, config)
        assert len(resized.shape) == 3
        assert max(resized.shape[:2]) <= config.target_size
        assert scale < 1.0

    def test_scale_factor(self, sample_large_image, config):
        """Test that scale factor is correct."""
        _, resized, scale = preprocess_for_contour(sample_large_image, config)
        expected_scale = config.target_size / max(sample_large_image.shape[:2])
        assert abs(scale - expected_scale) < 0.01

    def test_with_morphological(self, sample_bgr_image):
        """Test preprocessing with morphological closing enabled."""
        config = DetectionConfig(use_morphological=True, morph_kernel_size=5)
        binary, _, _ = preprocess_for_contour(sample_bgr_image, config)
        assert binary.shape == sample_bgr_image.shape[:2]

    def test_no_resize_for_small_image(self, sample_bgr_image, config):
        """Test that small images don't get resized."""
        _, resized, scale = preprocess_for_contour(sample_bgr_image, config)
        assert scale == 1.0
        assert resized.shape == sample_bgr_image.shape


# Tests for preprocess_for_hough


class TestPreprocessForHough:
    def test_returns_tuple(self, sample_bgr_image, config):
        """Test that function returns correct tuple structure."""
        result = preprocess_for_hough(sample_bgr_image, config)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_edge_output(self, sample_bgr_image, config):
        """Test that first output is edge image."""
        edges, _, _ = preprocess_for_hough(sample_bgr_image, config)
        unique_values = np.unique(edges)
        assert all(v in [0, 255] for v in unique_values)

    def test_resized_color_output(self, sample_large_image, config):
        """Test that second output is resized color image."""
        _, resized, scale = preprocess_for_hough(sample_large_image, config)
        assert len(resized.shape) == 3
        assert max(resized.shape[:2]) <= config.target_size
        assert scale < 1.0

    def test_scale_factor(self, sample_large_image, config):
        """Test that scale factor is correct."""
        _, resized, scale = preprocess_for_hough(sample_large_image, config)
        expected_scale = config.target_size / max(sample_large_image.shape[:2])
        assert abs(scale - expected_scale) < 0.01

    def test_uses_canny_parameters(self, sample_bgr_image):
        """Test that Canny parameters from config are used."""
        config_low = DetectionConfig(canny_low=30, canny_high=100)
        config_high = DetectionConfig(canny_low=100, canny_high=200)

        edges_low, _, _ = preprocess_for_hough(sample_bgr_image, config_low)
        edges_high, _, _ = preprocess_for_hough(sample_bgr_image, config_high)

        # Lower thresholds should detect more edges
        assert edges_low.sum() >= edges_high.sum()

    def test_no_resize_for_small_image(self, sample_bgr_image, config):
        """Test that small images don't get resized."""
        _, resized, scale = preprocess_for_hough(sample_bgr_image, config)
        assert scale == 1.0
        assert resized.shape == sample_bgr_image.shape


# Integration tests


class TestIntegration:
    def test_contour_pipeline_on_real_pattern(self, grid_like_image, config):
        """Test full contour pipeline on grid-like image."""
        # Convert to BGR
        bgr = cv2.cvtColor(grid_like_image, cv2.COLOR_GRAY2BGR)
        binary, resized, scale = preprocess_for_contour(bgr, config)

        # Should detect grid lines as white
        assert binary.sum() > 0
        assert resized.shape[:2] == binary.shape

    def test_hough_pipeline_on_real_pattern(self, grid_like_image, config):
        """Test full Hough pipeline on grid-like image."""
        # Convert to BGR
        bgr = cv2.cvtColor(grid_like_image, cv2.COLOR_GRAY2BGR)
        edges, resized, scale = preprocess_for_hough(bgr, config)

        # Should detect edges
        assert edges.sum() > 0
        assert resized.shape[:2] == edges.shape

    def test_both_pipelines_same_scale(self, sample_large_image, config):
        """Test that both pipelines produce same scale factor."""
        _, _, scale_contour = preprocess_for_contour(sample_large_image, config)
        _, _, scale_hough = preprocess_for_hough(sample_large_image, config)

        assert scale_contour == scale_hough
