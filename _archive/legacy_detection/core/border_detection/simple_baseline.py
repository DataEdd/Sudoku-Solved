"""
Simple baseline border detector.

Uses gradient thresholding and contour detection as a baseline
for comparing more sophisticated methods.
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


class SimpleBaselineDetector(BorderDetector):
    """
    Simple baseline detector using gradient magnitude and contour finding.

    Algorithm:
    1. Convert to grayscale
    2. Apply Gaussian blur
    3. Compute Sobel gradient magnitude
    4. Threshold to find strong edges
    5. Find contours
    6. Find largest quadrilateral
    7. Return corners

    This serves as a baseline to compare more sophisticated methods against.
    """

    name = "simple_baseline"
    description = "Gradient threshold + contour detection (baseline)"

    def detect(
        self,
        image: np.ndarray,
        preprocessed: Optional[np.ndarray] = None,
        blur_ksize: int = 5,
        sobel_ksize: int = 3,
        threshold: int = 50,
        epsilon_factor: float = 0.02,
        **kwargs
    ) -> BorderDetectionResult:
        """
        Detect Sudoku grid border.

        Args:
            image: BGR input image
            preprocessed: Ignored (detector does own preprocessing)
            blur_ksize: Gaussian blur kernel size
            sobel_ksize: Sobel operator kernel size
            threshold: Edge threshold value (0-255)
            epsilon_factor: Contour approximation factor

        Returns:
            BorderDetectionResult with detected corners
        """
        start = time.perf_counter()
        debug_images: Dict[str, np.ndarray] = {}

        # Step 1: Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        debug_images["1_grayscale"] = gray

        # Step 2: Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        debug_images["2_blurred"] = blurred

        # Step 3: Compute Sobel gradient magnitude
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        magnitude = np.sqrt(gx**2 + gy**2)

        # Normalize to 0-255
        if magnitude.max() > 0:
            magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        else:
            magnitude = magnitude.astype(np.uint8)
        debug_images["3_gradient"] = magnitude

        # Step 4: Threshold
        _, binary = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
        debug_images["4_threshold"] = binary

        # Step 5: Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Step 6: Find largest quadrilateral
        corners = None
        confidence = 0.0
        best_area = 0

        if contours:
            # Sort by area, largest first
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in contours[:10]:  # Check top 10 by area
                area = cv2.contourArea(contour)
                if area < 100:  # Skip tiny contours
                    continue

                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)

                # Looking for quadrilateral
                if len(approx) == 4:
                    candidate_corners = order_corners(approx.reshape(4, 2))

                    # Validate
                    if is_valid_sudoku_quad(candidate_corners, image.shape):
                        if area > best_area:
                            corners = candidate_corners
                            best_area = area

                            # Confidence based on area ratio
                            img_area = image.shape[0] * image.shape[1]
                            confidence = min(1.0, (area / (img_area * 0.3)) * 0.9)

        # Create debug image with detection result
        debug_images["5_detection"] = self._draw_detection(
            image, corners, contours[:5] if contours else []
        )

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
                    "threshold": threshold,
                    "epsilon_factor": epsilon_factor,
                }
            },
            execution_time_ms=elapsed_ms,
        )

    def _draw_detection(
        self,
        image: np.ndarray,
        corners: Optional[np.ndarray],
        top_contours: list
    ) -> np.ndarray:
        """Draw detection result on image."""
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        # Draw top contours in blue
        for contour in top_contours:
            cv2.drawContours(result, [contour], -1, (255, 0, 0), 1)

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
            "threshold": 50,
            "epsilon_factor": 0.02,
        }

    def get_recommended_pipeline(self) -> Optional[PipelineConfig]:
        # This detector does its own preprocessing
        return None
