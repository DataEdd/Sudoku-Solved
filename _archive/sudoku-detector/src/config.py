"""Configuration for sudoku grid detection."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DetectionConfig:
    """Configuration parameters for sudoku grid detection.

    Attributes:
        target_size: Maximum dimension for resized image (memory efficiency)
        min_input_size: Minimum acceptable input image dimension
        clahe_clip_limit: Contrast limit for CLAHE
        clahe_tile_size: Tile grid size for CLAHE
        adaptive_block_size: Block size for adaptive thresholding (must be odd)
        adaptive_c: Constant subtracted from mean in adaptive thresholding
        use_morphological: Whether to apply morphological closing
        morph_kernel_size: Kernel size for morphological operations
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        hough_threshold: Accumulator threshold for Hough line detection
        hough_min_line_length: Minimum line length for Hough detection
        hough_max_line_gap: Maximum gap between line segments
        min_aspect_ratio: Minimum acceptable aspect ratio for grid candidates
        max_aspect_ratio: Maximum acceptable aspect ratio for grid candidates
        min_area_ratio: Minimum grid area as ratio of image area
        max_area_ratio: Maximum grid area as ratio of image area
        min_interior_angle: Minimum acceptable interior angle for quadrilaterals
        max_interior_angle: Maximum acceptable interior angle for quadrilaterals
        min_line_count: Minimum line count for Hough grid validation
        max_line_count: Maximum line count for Hough grid validation
        line_cluster_divisor: Divisor for computing line clustering threshold
        contour_epsilon_factor: Factor for contour approximation epsilon
        output_size: Size of the output perspective-corrected image
    """

    # Image preprocessing
    target_size: int = 1000
    min_input_size: int = 200

    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)

    # Adaptive thresholding parameters
    adaptive_block_size: int = 11
    adaptive_c: int = 2

    # Morphological operations
    use_morphological: bool = False
    morph_kernel_size: int = 3

    # Canny edge detection parameters
    canny_low: int = 50
    canny_high: int = 150

    # Hough line detection parameters
    hough_threshold: int = 60
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10

    # Grid validation parameters
    min_aspect_ratio: float = 0.4
    max_aspect_ratio: float = 2.5
    min_area_ratio: float = 0.05
    max_area_ratio: float = 0.95

    # Angle validation parameters (currently hardcoded as 45-135 in contour_detection.py)
    min_interior_angle: float = 45.0
    max_interior_angle: float = 135.0

    # Line count validation (currently hardcoded as 8-12 in hough_detection.py)
    min_line_count: int = 1
    max_line_count: int = 20

    # Line clustering threshold divisor (image_dimension / divisor)
    # Higher values = more lines kept after clustering
    line_cluster_divisor: float = 36.0

    # Contour approximation epsilon factor (currently hardcoded as 0.02)
    contour_epsilon_factor: float = 0.05

    # Output parameters
    output_size: int = 450
