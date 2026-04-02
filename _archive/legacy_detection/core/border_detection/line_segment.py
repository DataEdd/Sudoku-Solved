"""
Line segment border detector.

Detects grid borders by finding line segments and computing
their intersections.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..pipeline import PipelineConfig
from .base import BorderDetectionResult, BorderDetector, is_valid_sudoku_quad


class LineSegmentDetector(BorderDetector):
    """
    Border detector using line segment detection and intersection.

    Algorithm:
    1. Preprocess with Canny edge detection
    2. Detect line segments using HoughLinesP
    3. Classify lines as horizontal or vertical
    4. Find extremal lines (topmost H, bottommost H, leftmost V, rightmost V)
    5. Compute intersection points as corners

    This method doesn't rely on contour detection and works well
    when the grid has clear straight lines.
    """

    name = "line_segment"
    description = "Line segment detection + intersection finding"

    def detect(
        self,
        image: np.ndarray,
        preprocessed: Optional[np.ndarray] = None,
        blur_ksize: int = 5,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 80,
        min_line_length: int = 50,
        max_line_gap: int = 10,
        angle_threshold: float = 15.0,
        **kwargs
    ) -> BorderDetectionResult:
        """
        Detect Sudoku grid border using line segments.

        Args:
            image: BGR input image
            preprocessed: Ignored
            blur_ksize: Gaussian blur kernel size
            canny_low: Canny low threshold
            canny_high: Canny high threshold
            hough_threshold: Hough accumulator threshold
            min_line_length: Minimum line segment length
            max_line_gap: Maximum gap between line segments
            angle_threshold: Degrees from horizontal/vertical to classify

        Returns:
            BorderDetectionResult with detected corners
        """
        start = time.perf_counter()
        debug_images: Dict[str, np.ndarray] = {}

        # Step 1: Preprocess
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        debug_images["1_blurred"] = blurred

        # Step 2: Canny edge detection
        edges = cv2.Canny(blurred, canny_low, canny_high)
        debug_images["2_canny"] = edges

        # Step 3: Detect line segments
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        if lines is None or len(lines) < 4:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return BorderDetectionResult(
                success=False,
                method=self.name,
                debug_images=debug_images,
                metadata={"error": "Not enough lines detected", "num_lines": 0},
                execution_time_ms=elapsed_ms,
            )

        lines = lines.reshape(-1, 4)
        debug_images["3_all_lines"] = self._draw_lines(image, lines, (128, 128, 128))

        # Step 4: Classify lines
        h_lines, v_lines = self._classify_lines(lines, angle_threshold)
        debug_images["4_classified"] = self._draw_classified_lines(image, h_lines, v_lines)

        if len(h_lines) < 2 or len(v_lines) < 2:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return BorderDetectionResult(
                success=False,
                method=self.name,
                debug_images=debug_images,
                metadata={
                    "error": "Not enough H/V lines",
                    "h_lines": len(h_lines),
                    "v_lines": len(v_lines),
                },
                execution_time_ms=elapsed_ms,
            )

        # Step 5: Find extremal lines
        top_line = min(h_lines, key=lambda l: (l[1] + l[3]) / 2)
        bottom_line = max(h_lines, key=lambda l: (l[1] + l[3]) / 2)
        left_line = min(v_lines, key=lambda l: (l[0] + l[2]) / 2)
        right_line = max(v_lines, key=lambda l: (l[0] + l[2]) / 2)

        # Step 6: Compute corner intersections
        tl = self._line_intersection(top_line, left_line)
        tr = self._line_intersection(top_line, right_line)
        br = self._line_intersection(bottom_line, right_line)
        bl = self._line_intersection(bottom_line, left_line)

        corners = np.array([tl, tr, br, bl], dtype=np.float32)

        # Validate corners are in image bounds
        h, w = image.shape[:2]
        valid = all(
            0 <= c[0] <= w and 0 <= c[1] <= h
            for c in corners
        )

        if valid:
            valid = is_valid_sudoku_quad(corners, image.shape)

        confidence = 0.7 if valid else 0.0

        # Draw final detection
        debug_images["5_extremal"] = self._draw_extremal(
            image, top_line, bottom_line, left_line, right_line, corners if valid else None
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return BorderDetectionResult(
            success=valid,
            corners=corners if valid else None,
            confidence=confidence,
            method=self.name,
            debug_images=debug_images,
            metadata={
                "num_lines": len(lines),
                "h_lines": len(h_lines),
                "v_lines": len(v_lines),
                "params": {
                    "canny_low": canny_low,
                    "canny_high": canny_high,
                    "hough_threshold": hough_threshold,
                    "min_line_length": min_line_length,
                    "angle_threshold": angle_threshold,
                }
            },
            execution_time_ms=elapsed_ms,
        )

    def _classify_lines(
        self,
        lines: np.ndarray,
        angle_threshold: float
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Classify lines as horizontal or vertical."""
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            # Horizontal: close to 0 or 180 degrees
            if angle < angle_threshold or angle > (180 - angle_threshold):
                h_lines.append(line)
            # Vertical: close to 90 degrees
            elif (90 - angle_threshold) < angle < (90 + angle_threshold):
                v_lines.append(line)

        return h_lines, v_lines

    def _line_intersection(
        self,
        line1: np.ndarray,
        line2: np.ndarray
    ) -> Tuple[float, float]:
        """Compute intersection of two lines (extended to infinity)."""
        x1, y1, x2, y2 = line1.astype(float)
        x3, y3, x4, y4 = line2.astype(float)

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-10:
            # Lines are parallel, return midpoint
            return (
                (x1 + x2 + x3 + x4) / 4,
                (y1 + y2 + y3 + y4) / 4
            )

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return (x, y)

    def _draw_lines(
        self,
        image: np.ndarray,
        lines: np.ndarray,
        color: tuple
    ) -> np.ndarray:
        """Draw lines on image."""
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), color, 1)

        return result

    def _draw_classified_lines(
        self,
        image: np.ndarray,
        h_lines: List[np.ndarray],
        v_lines: List[np.ndarray]
    ) -> np.ndarray:
        """Draw classified lines (H=green, V=blue)."""
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        for line in h_lines:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 1)

        for line in v_lines:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 1)

        return result

    def _draw_extremal(
        self,
        image: np.ndarray,
        top: np.ndarray,
        bottom: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        corners: Optional[np.ndarray]
    ) -> np.ndarray:
        """Draw extremal lines and corners."""
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        # Draw extremal lines
        for line, color in [
            (top, (0, 255, 255)),
            (bottom, (0, 255, 255)),
            (left, (255, 0, 255)),
            (right, (255, 0, 255)),
        ]:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), color, 2)

        # Draw corners
        if corners is not None:
            corners_int = corners.astype(np.int32)
            cv2.polylines(result, [corners_int], True, (0, 255, 0), 3)
            labels = ["TL", "TR", "BR", "BL"]
            for i, corner in enumerate(corners_int):
                cv2.circle(result, tuple(corner), 8, (0, 0, 255), -1)
                cv2.putText(
                    result, labels[i],
                    (corner[0] + 10, corner[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

        return result

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "blur_ksize": 5,
            "canny_low": 50,
            "canny_high": 150,
            "hough_threshold": 80,
            "min_line_length": 50,
            "max_line_gap": 10,
            "angle_threshold": 15.0,
        }

    def get_recommended_pipeline(self) -> Optional[PipelineConfig]:
        return PipelineConfig(
            name="canny_for_lines",
            steps=[
                ("grayscale", {}),
                ("gaussian_blur", {"kernel_size": 5}),
                ("canny", {"low_threshold": 50, "high_threshold": 150}),
            ]
        )
