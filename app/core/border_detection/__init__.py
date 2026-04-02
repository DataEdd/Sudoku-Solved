"""
Border detection module for Sudoku grid detection.

This package provides multiple methods for detecting the border
of a Sudoku grid in an image.

Example:
    from app.core.border_detection import DetectorRegistry

    # List available detectors
    print(DetectorRegistry.list_all())

    # Get and use a detector
    detector = DetectorRegistry.get("simple_baseline")
    result = detector.detect(image)

    if result.success:
        print(f"Corners: {result.corners}")
        annotated = result.draw_on_image(image)
"""

from .base import (
    BorderDetectionResult,
    BorderDetector,
    compute_quad_area,
    is_valid_sudoku_quad,
    order_corners,
)
from .line_segment import LineSegmentDetector
from .registry import DetectorRegistry
from .simple_baseline import SimpleBaselineDetector
from .sobel_flood import SobelFloodDetector

# List of all built-in detectors
BUILTIN_DETECTORS = [
    SimpleBaselineDetector,
    SobelFloodDetector,
    LineSegmentDetector,
]

# Auto-register all built-in detectors
for detector_class in BUILTIN_DETECTORS:
    DetectorRegistry.register(detector_class)


__all__ = [
    # Base classes
    "BorderDetector",
    "BorderDetectionResult",
    # Registry
    "DetectorRegistry",
    # Detectors
    "SimpleBaselineDetector",
    "SobelFloodDetector",
    "LineSegmentDetector",
    "BUILTIN_DETECTORS",
    # Utilities
    "order_corners",
    "compute_quad_area",
    "is_valid_sudoku_quad",
]
