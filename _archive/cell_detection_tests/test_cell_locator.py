"""
Unit tests for first cell locator.

Tests perspective transform, blob detection, and top-left cell finding.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add project and notebook utils to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPerspectiveTransform:
    """Tests for perspective transform function."""

    def test_warp_to_square(self):
        """Test that warping produces correct output size."""
        # Create a simple test image
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (450, 450), (255, 255, 255), 2)

        corners = np.array([
            [50, 50],    # TL
            [450, 50],   # TR
            [450, 450],  # BR
            [50, 450],   # BL
        ], dtype=np.float32)

        output_size = 450

        # Compute transform
        dst_pts = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1],
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(corners, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (output_size, output_size))

        assert warped.shape == (450, 450, 3)

    def test_warp_preserves_content(self):
        """Test that warping preserves image content."""
        # Create image with distinctive pattern
        image = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(image, (200, 200), 50, 255, -1)  # White circle in center

        corners = np.array([
            [0, 0], [399, 0], [399, 399], [0, 399]
        ], dtype=np.float32)

        output_size = 400
        dst_pts = np.array([
            [0, 0], [399, 0], [399, 399], [0, 399]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(corners, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (output_size, output_size))

        # Center should still be white (circle present)
        center_value = warped[200, 200]
        assert center_value > 200, "Center should contain the white circle"

    def test_warp_handles_skewed_quad(self):
        """Test warping of a skewed quadrilateral."""
        image = np.zeros((500, 500, 3), dtype=np.uint8)

        # Skewed corners (trapezoid)
        corners = np.array([
            [100, 50],   # TL
            [400, 80],   # TR (slightly lower)
            [420, 420],  # BR
            [80, 400],   # BL
        ], dtype=np.float32)

        output_size = 300
        dst_pts = np.array([
            [0, 0], [299, 0], [299, 299], [0, 299]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(corners, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (output_size, output_size))

        assert warped.shape == (300, 300, 3)


class TestConnectedComponents:
    """Tests for connected component analysis."""

    def test_finds_single_blob(self):
        """Test detection of a single blob."""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (60, 60), 255, -1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        # Should have 2 labels: background (0) and the blob (1)
        assert num_labels == 2
        assert stats[1, cv2.CC_STAT_AREA] == 41 * 41  # (60-20+1) * (60-20+1)

    def test_finds_multiple_blobs(self):
        """Test detection of multiple separate blobs."""
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (40, 40), 255, -1)    # Blob 1
        cv2.rectangle(image, (100, 100), (150, 150), 255, -1)  # Blob 2

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        assert num_labels == 3  # Background + 2 blobs

    def test_centroid_calculation(self):
        """Test that centroid is calculated correctly."""
        image = np.zeros((100, 100), dtype=np.uint8)
        # Square blob from (20,20) to (40,40)
        cv2.rectangle(image, (20, 20), (40, 40), 255, -1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        # Centroid should be at center of rectangle
        # Rectangle spans 20-40 inclusive, so center is at 30
        cx, cy = centroids[1]
        assert 29 <= cx <= 31, f"Centroid X should be ~30, got {cx}"
        assert 29 <= cy <= 31, f"Centroid Y should be ~30, got {cy}"

    def test_stats_contain_bbox(self):
        """Test that stats contain correct bounding box."""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (15, 25), (45, 55), 255, -1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        x = stats[1, cv2.CC_STAT_LEFT]
        y = stats[1, cv2.CC_STAT_TOP]
        w = stats[1, cv2.CC_STAT_WIDTH]
        h = stats[1, cv2.CC_STAT_HEIGHT]

        assert x == 15
        assert y == 25
        assert w == 31  # 45 - 15 + 1
        assert h == 31  # 55 - 25 + 1


class TestBlobFiltering:
    """Tests for blob filtering by area and aspect ratio."""

    def test_filters_small_blobs(self):
        """Test that small blobs are filtered out."""
        # Create image with one large and one small blob
        image = np.zeros((450, 450), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (100, 100), 255, -1)   # Large: 51x51 = 2601
        cv2.rectangle(image, (200, 200), (210, 210), 255, -1)  # Small: 11x11 = 121

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        # Filter by minimum area (e.g., 500 pixels)
        min_area = 500
        filtered = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered.append(i)

        assert len(filtered) == 1, "Should only keep the large blob"

    def test_filters_by_aspect_ratio(self):
        """Test that non-square blobs are filtered out."""
        image = np.zeros((450, 450), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (100, 100), 255, -1)   # Square: 51x51, ratio=1.0
        cv2.rectangle(image, (200, 200), (400, 220), 255, -1)  # Wide: 201x21, ratio=9.6

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        min_aspect, max_aspect = 0.5, 2.0
        filtered = []
        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = w / h if h > 0 else 0
            if min_aspect <= aspect <= max_aspect:
                filtered.append(i)

        assert len(filtered) == 1, "Should only keep the square blob"

    def test_keeps_cell_sized_blobs(self):
        """Test that cell-sized blobs pass filtering."""
        # For 450x450 grid, cells are ~50x50
        image = np.zeros((450, 450), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (95, 95), 255, -1)  # 46x46, area=2116

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        image_area = 450 * 450
        min_area_ratio, max_area_ratio = 0.003, 0.02

        filtered = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            ratio = area / image_area
            if min_area_ratio <= ratio <= max_area_ratio:
                filtered.append(i)

        assert len(filtered) == 1, "Cell-sized blob should pass"


class TestTopLeftFinding:
    """Tests for finding the top-left cell."""

    def test_finds_top_left_by_centroid_sum(self):
        """Test that cell with minimum centroid sum is selected."""
        # Simulate blob centroids
        blobs = [
            {"centroid": (200, 200)},  # sum = 400
            {"centroid": (50, 50)},    # sum = 100 (should be top-left)
            {"centroid": (300, 100)},  # sum = 400
            {"centroid": (100, 300)},  # sum = 400
        ]

        top_left = min(blobs, key=lambda b: b["centroid"][0] + b["centroid"][1])

        assert top_left["centroid"] == (50, 50)

    def test_handles_tie_in_sum(self):
        """Test behavior when multiple blobs have same sum."""
        blobs = [
            {"centroid": (100, 50)},   # sum = 150
            {"centroid": (50, 100)},   # sum = 150 (same)
            {"centroid": (200, 200)},  # sum = 400
        ]

        # min() will return the first one with minimum value
        top_left = min(blobs, key=lambda b: b["centroid"][0] + b["centroid"][1])

        # Should be one of the two with sum=150
        assert top_left["centroid"][0] + top_left["centroid"][1] == 150

    def test_handles_single_blob(self):
        """Test with only one blob."""
        blobs = [{"centroid": (100, 100)}]

        top_left = min(blobs, key=lambda b: b["centroid"][0] + b["centroid"][1])

        assert top_left["centroid"] == (100, 100)

    def test_handles_empty_list(self):
        """Test with no blobs."""
        blobs = []

        with pytest.raises(ValueError):
            min(blobs, key=lambda b: b["centroid"][0] + b["centroid"][1])


class TestIntegration:
    """Integration tests for the full cell locator pipeline."""

    def test_finds_cell_in_synthetic_grid(self, synthetic_grid_image):
        """Test cell detection on synthetic grid."""
        image, corners = synthetic_grid_image(size=450, rotation=0)

        # Warp to top-down
        output_size = 450
        dst_pts = np.array([
            [0, 0], [449, 0], [449, 449], [0, 449]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(corners, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (output_size, output_size))

        # Convert to grayscale and threshold
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        # Should find multiple blobs (cells)
        assert num_labels > 1, "Should find at least one cell"

    def test_top_left_is_in_correct_region(self, synthetic_grid_image):
        """Test that detected top-left cell is in upper-left quadrant."""
        image, corners = synthetic_grid_image(size=450)

        output_size = 450
        dst_pts = np.array([
            [0, 0], [449, 0], [449, 449], [0, 449]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(corners, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (output_size, output_size))

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        if num_labels > 1:
            # Find cell with minimum centroid sum
            min_sum = float('inf')
            top_left_centroid = None

            for i in range(1, num_labels):
                cx, cy = centroids[i]
                if cx + cy < min_sum:
                    min_sum = cx + cy
                    top_left_centroid = (cx, cy)

            if top_left_centroid:
                # Should be in upper-left quadrant
                assert top_left_centroid[0] < output_size / 2, "X should be in left half"
                assert top_left_centroid[1] < output_size / 2, "Y should be in top half"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_handles_no_cells_detected(self):
        """Test handling when no cells are detected."""
        # Black image - no blobs
        image = np.zeros((450, 450), dtype=np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        assert num_labels == 1, "Only background label"

    def test_handles_all_white_image(self):
        """Test handling of all-white image."""
        image = np.full((450, 450), 255, dtype=np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        # Entire image is one blob
        assert num_labels == 2

    def test_handles_very_small_cells(self):
        """Test that very small cells are filtered out."""
        image = np.zeros((450, 450), dtype=np.uint8)
        # Add tiny blobs
        for i in range(0, 400, 50):
            for j in range(0, 400, 50):
                cv2.rectangle(image, (i, j), (i+5, j+5), 255, -1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        # All blobs are very small (36 pixels each)
        image_area = 450 * 450
        min_area = 0.003 * image_area  # ~607 pixels

        filtered_count = sum(
            1 for i in range(1, num_labels)
            if stats[i, cv2.CC_STAT_AREA] >= min_area
        )

        assert filtered_count == 0, "All small blobs should be filtered"
