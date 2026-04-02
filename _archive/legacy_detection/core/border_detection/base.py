"""
Base classes for border detection methods.

This module provides the abstract interface that all border detection
methods must implement, plus the result dataclass.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..pipeline import PipelineConfig


@dataclass
class BorderDetectionResult:
    """
    Result from border detection.

    Attributes:
        success: Whether detection was successful
        corners: 4x2 array of corner points [TL, TR, BR, BL] or None
        confidence: Confidence score 0-1
        method: Name of the detection method used
        debug_images: Dict of intermediate images for visualization
        metadata: Additional method-specific info
        execution_time_ms: Time taken for detection
    """

    success: bool
    corners: Optional[np.ndarray] = None  # 4x2 array: TL, TR, BR, BL
    confidence: float = 0.0
    method: str = ""
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0

    def __repr__(self) -> str:
        if self.success and self.corners is not None:
            corners_str = f"corners=[{self.corners[0]}, ...]"
        else:
            corners_str = "corners=None"
        return (
            f"BorderDetectionResult(success={self.success}, "
            f"{corners_str}, confidence={self.confidence:.2f}, "
            f"method={self.method!r}, time={self.execution_time_ms:.1f}ms)"
        )

    def get_corners_ordered(self) -> Optional[np.ndarray]:
        """Get corners in order: TL, TR, BR, BL."""
        return self.corners

    def draw_on_image(
        self,
        image: np.ndarray,
        color: tuple = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detected border on image.

        Args:
            image: Image to draw on (will be copied)
            color: BGR color for border lines
            thickness: Line thickness

        Returns:
            Image with border drawn
        """
        if self.corners is None:
            return image.copy()

        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        corners = self.corners.astype(np.int32)

        # Draw quadrilateral
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(result, pt1, pt2, color, thickness)

        # Draw corner circles
        corner_labels = ["TL", "TR", "BR", "BL"]
        for i, (corner, label) in enumerate(zip(corners, corner_labels)):
            cv2.circle(result, tuple(corner), 5, (0, 0, 255), -1)
            cv2.putText(
                result, label,
                (corner[0] + 10, corner[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        return result


class BorderDetector(ABC):
    """
    Abstract base class for border detection methods.

    All detection methods must inherit from this class and implement
    the required abstract methods.

    Example:
        class MyDetector(BorderDetector):
            name = "my_detector"
            description = "My custom border detection"

            def detect(self, image, preprocessed=None, **kwargs):
                # Detection logic here
                return BorderDetectionResult(...)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this detection method.

        Used in CLI and registry. Should be lowercase with underscores.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of the detection method.

        Shown in reports and CLI help.
        """
        pass

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        preprocessed: Optional[np.ndarray] = None,
        **kwargs
    ) -> BorderDetectionResult:
        """
        Detect the border/corners of a Sudoku grid.

        Args:
            image: Original BGR image
            preprocessed: Pre-processed image (optional, detector may preprocess itself)
            **kwargs: Method-specific parameters

        Returns:
            BorderDetectionResult with corners or failure info
        """
        pass

    def get_default_params(self) -> Dict[str, Any]:
        """
        Return default parameters for this detector.

        Override to specify detector-specific defaults.
        """
        return {}

    def get_recommended_pipeline(self) -> Optional[PipelineConfig]:
        """
        Return recommended preprocessing pipeline for this detector.

        Override if detector works best with specific preprocessing.
        Returns None if no specific pipeline is recommended.
        """
        return None

    def validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validate parameters before detection.

        Override to add parameter validation.

        Raises:
            ValueError: If parameters are invalid
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.

    Uses sum and difference of coordinates to determine position.

    Args:
        pts: Array of 4 points, shape (4, 2)

    Returns:
        Ordered array of shape (4, 2)
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Top-left has smallest sum, bottom-right has largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR

    # Top-right has smallest difference, bottom-left has largest
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL

    return rect


def compute_quad_area(corners: np.ndarray) -> float:
    """
    Compute area of quadrilateral using Shoelace formula.

    Args:
        corners: 4x2 array of corner points

    Returns:
        Area in pixels
    """
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) / 2.0


def is_valid_sudoku_quad(
    corners: np.ndarray,
    image_shape: tuple,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.95,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0
) -> bool:
    """
    Check if detected quadrilateral is a plausible Sudoku grid.

    Args:
        corners: 4x2 array of corner points
        image_shape: (height, width) of image
        min_area_ratio: Minimum quad area / image area
        max_area_ratio: Maximum quad area / image area
        min_aspect_ratio: Minimum width/height ratio
        max_aspect_ratio: Maximum width/height ratio

    Returns:
        True if quad is plausible, False otherwise
    """
    h, w = image_shape[:2]
    image_area = h * w
    quad_area = compute_quad_area(corners)

    # Check area ratio
    area_ratio = quad_area / image_area
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False

    # Check aspect ratio (approximate)
    # Use distance between TL-TR and TL-BL
    width = np.linalg.norm(corners[1] - corners[0])
    height = np.linalg.norm(corners[3] - corners[0])

    if height == 0:
        return False

    aspect = width / height
    if aspect < min_aspect_ratio or aspect > max_aspect_ratio:
        return False

    return True
