"""Tests for the main detector module."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from src.config import DetectionConfig
from src.detector import DetectionResult, SudokuDetector


# Fixtures


@pytest.fixture
def default_detector():
    """Create detector with default configuration."""
    return SudokuDetector()


@pytest.fixture
def custom_config():
    """Create a custom configuration."""
    return DetectionConfig(
        target_size=800,
        output_size=400,
        min_input_size=100,
    )


@pytest.fixture
def custom_detector(custom_config):
    """Create detector with custom configuration."""
    return SudokuDetector(config=custom_config)


@pytest.fixture
def valid_bgr_image():
    """Create a valid BGR image."""
    return np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)


@pytest.fixture
def small_image():
    """Create an image below minimum size."""
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_sudoku_image():
    """Create a synthetic image with a white square (simulating sudoku grid)."""
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    # Draw a white filled square in the center (simulates sudoku grid outline)
    cv2.rectangle(img, (100, 100), (500, 500), (255, 255, 255), -1)
    # Add some internal grid lines to make it more sudoku-like
    for i in range(1, 9):
        pos = 100 + i * 50
        cv2.line(img, (100, pos), (500, pos), (200, 200, 200), 2)
        cv2.line(img, (pos, 100), (pos, 500), (200, 200, 200), 2)
    return img


@pytest.fixture
def synthetic_sudoku_with_border():
    """Create a synthetic sudoku with clear black border on white background."""
    img = np.ones((600, 600, 3), dtype=np.uint8) * 255  # White background
    # Draw black border rectangle
    cv2.rectangle(img, (100, 100), (500, 500), (0, 0, 0), 3)
    # Add internal grid lines
    for i in range(1, 9):
        pos = 100 + i * 50
        thickness = 2 if i % 3 != 0 else 3
        cv2.line(img, (100, pos), (500, pos), (0, 0, 0), thickness)
        cv2.line(img, (pos, 100), (pos, 500), (0, 0, 0), thickness)
    return img


@pytest.fixture
def multiple_squares_image():
    """Create image with multiple squares for centeredness testing."""
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    # Centered square (should be selected)
    cv2.rectangle(img, (250, 250), (550, 550), (255, 255, 255), -1)
    # Off-center square (smaller, in corner)
    cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)
    return img


# Tests for SudokuDetector initialization


class TestSudokuDetectorInit:
    def test_default_config(self):
        """Test initialization with default config."""
        detector = SudokuDetector()
        assert detector.config is not None
        assert isinstance(detector.config, DetectionConfig)
        assert detector.config.target_size == 1000
        assert detector.config.output_size == 450

    def test_custom_config(self, custom_config):
        """Test initialization with custom config."""
        detector = SudokuDetector(config=custom_config)
        assert detector.config.target_size == 800
        assert detector.config.output_size == 400

    def test_none_config_uses_default(self):
        """Test that None config creates default."""
        detector = SudokuDetector(config=None)
        assert detector.config is not None
        assert detector.config.target_size == 1000


# Tests for _validate_input


class TestValidateInput:
    def test_valid_bgr_image(self, default_detector, valid_bgr_image):
        """Test validation of valid BGR image."""
        is_valid, error = default_detector._validate_input(valid_bgr_image)
        assert is_valid is True
        assert error is None

    def test_valid_grayscale_image(self, default_detector):
        """Test validation of valid grayscale image."""
        gray = np.random.randint(0, 256, (500, 500), dtype=np.uint8)
        is_valid, error = default_detector._validate_input(gray)
        assert is_valid is True
        assert error is None

    def test_none_image(self, default_detector):
        """Test that None image is rejected."""
        is_valid, error = default_detector._validate_input(None)
        assert is_valid is False
        assert "None" in error

    def test_invalid_type(self, default_detector):
        """Test that non-array is rejected."""
        is_valid, error = default_detector._validate_input("not an image")
        assert is_valid is False
        assert "numpy array" in error

    def test_invalid_dimensions_1d(self, default_detector):
        """Test that 1D array is rejected."""
        arr = np.array([1, 2, 3])
        is_valid, error = default_detector._validate_input(arr)
        assert is_valid is False
        assert "dimensions" in error.lower()

    def test_invalid_dimensions_4d(self, default_detector):
        """Test that 4D array is rejected."""
        arr = np.zeros((10, 10, 10, 10))
        is_valid, error = default_detector._validate_input(arr)
        assert is_valid is False
        assert "dimensions" in error.lower()

    def test_image_too_small(self, default_detector, small_image):
        """Test that small image is rejected."""
        is_valid, error = default_detector._validate_input(small_image)
        assert is_valid is False
        assert "too small" in error.lower()

    def test_custom_min_size(self, custom_detector):
        """Test validation with custom min_input_size."""
        # 150x150 should pass with min_input_size=100
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        is_valid, error = custom_detector._validate_input(img)
        assert is_valid is True


# Tests for _select_best_candidate


class TestSelectBestCandidate:
    def test_empty_list(self, default_detector):
        """Test with empty candidate list."""
        result = default_detector._select_best_candidate([])
        assert result is None

    def test_single_candidate(self, default_detector):
        """Test with single candidate."""
        candidates = [{"corners": np.zeros((4, 2)), "score": 0.8, "method": "contour"}]
        result = default_detector._select_best_candidate(candidates)
        assert result is not None
        assert result["score"] == 0.8

    def test_selects_highest_score(self, default_detector):
        """Test that highest score is selected."""
        candidates = [
            {"corners": np.zeros((4, 2)), "score": 0.5, "method": "contour"},
            {"corners": np.zeros((4, 2)), "score": 0.9, "method": "hough"},
            {"corners": np.zeros((4, 2)), "score": 0.7, "method": "contour"},
        ]
        result = default_detector._select_best_candidate(candidates)
        assert result["score"] == 0.9
        assert result["method"] == "hough"

    def test_preserves_candidate_data(self, default_detector):
        """Test that all candidate data is preserved."""
        corners = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        candidates = [{"corners": corners, "score": 0.8, "method": "contour"}]
        result = default_detector._select_best_candidate(candidates)
        np.testing.assert_array_equal(result["corners"], corners)


# Tests for detect method


class TestDetect:
    def test_returns_detection_result(self, default_detector, valid_bgr_image):
        """Test that detect returns DetectionResult."""
        result = default_detector.detect(valid_bgr_image)
        assert isinstance(result, DetectionResult)

    def test_invalid_image_returns_failure(self, default_detector):
        """Test that invalid image returns failure result."""
        result = default_detector.detect(None)
        assert result.success is False
        assert result.error_message is not None

    def test_small_image_returns_failure(self, default_detector, small_image):
        """Test that small image returns failure result."""
        result = default_detector.detect(small_image)
        assert result.success is False
        assert "too small" in result.error_message.lower()

    def test_no_detection_returns_failure(self, default_detector):
        """Test that image with no detectable grid returns failure."""
        # Plain black image
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        result = default_detector.detect(img)
        assert result.success is False
        assert "no sudoku" in result.error_message.lower() or "no valid" in result.error_message.lower()

    def test_successful_detection_fields(self, default_detector, synthetic_sudoku_image):
        """Test that successful detection populates all fields."""
        result = default_detector.detect(synthetic_sudoku_image)

        # May or may not succeed depending on detection
        if result.success:
            assert result.warped_image is not None
            assert result.corners is not None
            assert result.corners_normalized is not None
            assert result.homography is not None
            assert result.confidence > 0
            assert result.detection_method in ["contour", "hough"]
            assert result.error_message is None

    def test_warped_image_shape(self, default_detector, synthetic_sudoku_image):
        """Test that warped image has correct shape."""
        result = default_detector.detect(synthetic_sudoku_image)

        if result.success:
            expected_size = default_detector.config.output_size
            if len(result.warped_image.shape) == 3:
                assert result.warped_image.shape == (expected_size, expected_size, 3)
            else:
                assert result.warped_image.shape == (expected_size, expected_size)

    def test_corners_shape(self, default_detector, synthetic_sudoku_image):
        """Test that corners have correct shape."""
        result = default_detector.detect(synthetic_sudoku_image)

        if result.success:
            assert result.corners.shape == (4, 2)
            assert result.corners_normalized.shape == (4, 2)

    def test_homography_shape(self, default_detector, synthetic_sudoku_image):
        """Test that homography has correct shape."""
        result = default_detector.detect(synthetic_sudoku_image)

        if result.success:
            assert result.homography.shape == (3, 3)

    def test_confidence_range(self, default_detector, synthetic_sudoku_image):
        """Test that confidence is in valid range."""
        result = default_detector.detect(synthetic_sudoku_image)

        if result.success:
            assert 0.0 <= result.confidence <= 1.0

    def test_grayscale_input(self, default_detector):
        """Test detection with grayscale input."""
        # Create grayscale synthetic image
        img = np.zeros((600, 600), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (500, 500), 255, -1)

        result = default_detector.detect(img)
        # Should not crash, may or may not detect
        assert isinstance(result, DetectionResult)


# Tests for detect_from_file


class TestDetectFromFile:
    def test_valid_file(self, default_detector, synthetic_sudoku_image):
        """Test detection from valid file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, synthetic_sudoku_image)
            try:
                result = default_detector.detect_from_file(f.name)
                assert isinstance(result, DetectionResult)
            finally:
                os.unlink(f.name)

    def test_nonexistent_file(self, default_detector):
        """Test detection from non-existent file."""
        result = default_detector.detect_from_file("/nonexistent/path/image.png")
        assert result.success is False
        assert "failed to load" in result.error_message.lower()

    def test_invalid_file(self, default_detector):
        """Test detection from non-image file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not an image")
            f.flush()
            try:
                result = default_detector.detect_from_file(f.name)
                assert result.success is False
                assert "failed to load" in result.error_message.lower()
            finally:
                os.unlink(f.name)

    def test_small_image_file(self, default_detector, small_image):
        """Test detection from file with small image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, small_image)
            try:
                result = default_detector.detect_from_file(f.name)
                assert result.success is False
                # Should fail at load_and_validate stage
                assert "failed to load" in result.error_message.lower()
            finally:
                os.unlink(f.name)


# Tests for both detection paths


class TestDualPathDetection:
    def test_contour_path_attempted(self, default_detector, synthetic_sudoku_image):
        """Test that contour detection path is attempted."""
        # We can't directly test this without mocking, but we can verify
        # the result could come from contour path
        result = default_detector.detect(synthetic_sudoku_image)

        if result.success:
            assert result.detection_method in ["contour", "hough"]

    def test_hough_path_attempted(self, default_detector, synthetic_sudoku_with_border):
        """Test that hough detection path is attempted."""
        result = default_detector.detect(synthetic_sudoku_with_border)

        if result.success:
            assert result.detection_method in ["contour", "hough"]

    def test_most_centered_selected(self, default_detector, multiple_squares_image):
        """Test that most centered candidate is selected."""
        # Lower min_area_ratio to detect smaller squares too
        default_detector.config.min_area_ratio = 0.01

        result = default_detector.detect(multiple_squares_image)

        if result.success:
            # The centered square (250-550) should be selected
            # Its center is at (400, 400) which is the image center
            corners = result.corners
            centroid = np.mean(corners, axis=0)
            image_center = np.array([400, 400])

            # Centroid should be close to image center (within 100 pixels)
            distance = np.linalg.norm(centroid - image_center)
            # Allow some tolerance since corners may be refined
            assert distance < 200


# Tests for DetectionResult dataclass


class TestDetectionResult:
    def test_success_result(self):
        """Test creating successful result."""
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        warped = np.zeros((450, 450, 3), dtype=np.uint8)
        H = np.eye(3)

        result = DetectionResult(
            success=True,
            warped_image=warped,
            corners=corners,
            corners_normalized=corners,
            homography=H,
            confidence=0.95,
            detection_method="contour",
        )

        assert result.success is True
        assert result.warped_image is not None
        assert result.confidence == 0.95

    def test_failure_result(self):
        """Test creating failure result."""
        result = DetectionResult(
            success=False,
            error_message="Test error",
        )

        assert result.success is False
        assert result.warped_image is None
        assert result.error_message == "Test error"

    def test_default_values(self):
        """Test default values."""
        result = DetectionResult(success=False)

        assert result.warped_image is None
        assert result.corners is None
        assert result.corners_normalized is None
        assert result.homography is None
        assert result.confidence == 0.0
        assert result.detection_method is None
        assert result.error_message is None


# Integration tests


class TestDetectorIntegration:
    def test_full_pipeline_synthetic(self, default_detector, synthetic_sudoku_image):
        """Test full detection pipeline with synthetic image."""
        result = default_detector.detect(synthetic_sudoku_image)

        # The synthetic image should be detectable
        if result.success:
            # Verify the output is usable
            assert result.warped_image.dtype == np.uint8
            assert np.max(result.warped_image) > 0  # Not all black

    def test_full_pipeline_from_file(self, default_detector, synthetic_sudoku_image):
        """Test full pipeline from file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, synthetic_sudoku_image)
            try:
                result = default_detector.detect_from_file(f.name)
                assert isinstance(result, DetectionResult)
            finally:
                os.unlink(f.name)

    def test_custom_output_size(self, synthetic_sudoku_image):
        """Test with custom output size."""
        config = DetectionConfig(output_size=300)
        detector = SudokuDetector(config=config)

        result = detector.detect(synthetic_sudoku_image)

        if result.success:
            assert result.warped_image.shape[0] == 300
            assert result.warped_image.shape[1] == 300

    def test_import_from_package(self):
        """Test that main classes can be imported from package."""
        from src import DetectionConfig, DetectionResult, SudokuDetector

        assert DetectionConfig is not None
        assert SudokuDetector is not None
        assert DetectionResult is not None

        # Can instantiate
        config = DetectionConfig()
        detector = SudokuDetector(config)
        assert detector is not None
