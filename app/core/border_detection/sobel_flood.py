"""
Sobel edge + flood fill border detector.

Uses flood fill from image corners to isolate the Sudoku grid region.
"""

import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..pipeline import PipelineConfig
from .base import (
    BorderDetectionResult,
    BorderDetector,
    is_valid_sudoku_quad,
    order_corners,
)


class SobelFloodDetector(BorderDetector):
    """
    Border detector using Sobel edges and flood fill isolation.

    Algorithm:
    1. Convert to grayscale and blur
    2. Compute Sobel gradient magnitude
    3. Threshold edges
    4. Flood fill from image corners (background becomes marked)
    5. Invert: what's not background is the grid region
    6. Find largest quadrilateral contour

    This method is more robust to partial occlusion and noise
    at the grid edges.
    """

    name = "sobel_flood"
    description = "Sobel edges + flood fill to isolate grid"

    def detect(
        self,
        image: np.ndarray,
        preprocessed: Optional[np.ndarray] = None,
        blur_ksize: int = 5,
        sobel_ksize: int = 3,
        edge_threshold: int = 30,
        flood_tolerance: int = 10,
        epsilon_factor: float = 0.02,
        **kwargs
    ) -> BorderDetectionResult:
        """
        Detect Sudoku grid border using flood fill isolation.

        Args:
            image: BGR input image
            preprocessed: Ignored
            blur_ksize: Gaussian blur kernel size
            sobel_ksize: Sobel operator kernel size
            edge_threshold: Threshold for edge detection
            flood_tolerance: Flood fill tolerance
            epsilon_factor: Contour approximation factor

        Returns:
            BorderDetectionResult with detected corners
        """
        start = time.perf_counter()
        debug_images: Dict[str, np.ndarray] = {}

        # Step 1: Grayscale and blur
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        debug_images["1_blurred"] = blurred

        # Step 2: Compute Sobel edges
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        magnitude = np.sqrt(gx**2 + gy**2)

        if magnitude.max() > 0:
            edges = (magnitude / magnitude.max() * 255).astype(np.uint8)
        else:
            edges = np.zeros_like(gray)
        debug_images["2_sobel_edges"] = edges

        # Step 3: Threshold edges
        _, binary_edges = cv2.threshold(edges, edge_threshold, 255, cv2.THRESH_BINARY)
        debug_images["3_binary_edges"] = binary_edges

        # Step 4: Flood fill from corners
        h, w = binary_edges.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        filled = binary_edges.copy()

        # Fill from all four corners
        fill_value = 128
        corner_points = [
            (0, 0),           # Top-left
            (w - 1, 0),       # Top-right
            (0, h - 1),       # Bottom-left
            (w - 1, h - 1),   # Bottom-right
        ]

        for x, y in corner_points:
            # Only fill if the pixel is dark (not an edge)
            if filled[y, x] < 128:
                cv2.floodFill(
                    filled, mask, (x, y), fill_value,
                    flood_tolerance, flood_tolerance,
                    cv2.FLOODFILL_FIXED_RANGE
                )

        debug_images["4_flood_filled"] = filled

        # Step 5: Create grid mask (everything not filled is grid region)
        grid_mask = np.where(filled == fill_value, 0, 255).astype(np.uint8)
        debug_images["5_grid_mask"] = grid_mask

        # Step 6: Find contours in grid mask
        contours, _ = cv2.findContours(
            grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        corners = None
        confidence = 0.0

        if contours:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 100:  # Minimum size check
                peri = cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, epsilon_factor * peri, True)

                if len(approx) == 4:
                    corners = order_corners(approx.reshape(4, 2))

                    if is_valid_sudoku_quad(corners, image.shape):
                        img_area = h * w
                        confidence = min(1.0, (area / (img_area * 0.15)) * 0.85)
                    else:
                        corners = None

        # Create debug visualization
        debug_images["6_detection"] = self._draw_detection(image, corners, contours)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return BorderDetectionResult(
            success=corners is not None,
            corners=corners,
            confidence=confidence,
            method=self.name,
            debug_images=debug_images,
            metadata={
                "num_contours": len(contours) if contours else 0,
                "params": {
                    "blur_ksize": blur_ksize,
                    "sobel_ksize": sobel_ksize,
                    "edge_threshold": edge_threshold,
                    "flood_tolerance": flood_tolerance,
                    "epsilon_factor": epsilon_factor,
                }
            },
            execution_time_ms=elapsed_ms,
        )

    def _draw_detection(
        self,
        image: np.ndarray,
        corners: Optional[np.ndarray],
        contours: list
    ) -> np.ndarray:
        """Draw detection result on image."""
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        # Draw all contours in blue
        if contours:
            cv2.drawContours(result, contours, -1, (255, 0, 0), 1)

        # Draw detected corners in green
        if corners is not None:
            corners_int = corners.astype(np.int32)
            cv2.polylines(result, [corners_int], True, (0, 255, 0), 3)
            for i, corner in enumerate(corners_int):
                cv2.circle(result, tuple(corner), 8, (0, 0, 255), -1)
                labels = ["TL", "TR", "BR", "BL"]
                cv2.putText(
                    result, labels[i],
                    (corner[0] + 10, corner[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

        return result

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "blur_ksize": 5,
            "sobel_ksize": 3,
            "edge_threshold": 30,
            "flood_tolerance": 10,
            "epsilon_factor": 0.02,
        }

    def get_recommended_pipeline(self) -> Optional[PipelineConfig]:
        return None
