"""Diagnose why sudoku detection failed on specific images.

This module analyzes the detection pipeline to determine the root cause
of failures, enabling targeted parameter tuning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import DetectionConfig
from ..preprocessing import (
    adaptive_threshold,
    apply_clahe,
    apply_morphological,
    get_edges,
    load_and_validate,
    resize_image,
    to_grayscale,
)
from ..contour_detection import (
    find_quadrilateral_contours,
    validate_quadrilateral,
    detect_contour_path,
)
from ..hough_detection import (
    detect_lines,
    classify_lines,
    cluster_lines,
    validate_grid_lines,
    compute_all_intersections,
    extract_outer_corners,
)


# Diagnostic failure codes
class FailureCode:
    """Failure diagnosis codes."""
    # General
    LOAD_FAILED = "LOAD_FAILED"
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"

    # Contour path failures
    NO_CONTOURS = "NO_CONTOURS"
    AREA_REJECTED_SMALL = "AREA_REJECTED_SMALL"
    AREA_REJECTED_LARGE = "AREA_REJECTED_LARGE"
    NO_QUADRILATERALS = "NO_QUADRILATERALS"
    ASPECT_REJECTED = "ASPECT_REJECTED"
    ANGLE_REJECTED = "ANGLE_REJECTED"
    NOT_CONVEX = "NOT_CONVEX"
    ALL_QUADS_INVALID = "ALL_QUADS_INVALID"

    # Hough path failures
    NO_EDGES = "NO_EDGES"
    HOUGH_NO_LINES = "HOUGH_NO_LINES"
    HOUGH_FEW_HORIZONTAL = "HOUGH_FEW_HORIZONTAL"
    HOUGH_FEW_VERTICAL = "HOUGH_FEW_VERTICAL"
    HOUGH_LINE_COUNT_LOW = "HOUGH_LINE_COUNT_LOW"
    HOUGH_LINE_COUNT_HIGH = "HOUGH_LINE_COUNT_HIGH"
    HOUGH_FEW_INTERSECTIONS = "HOUGH_FEW_INTERSECTIONS"
    HOUGH_CORNER_EXTRACTION_FAILED = "HOUGH_CORNER_EXTRACTION_FAILED"

    # Quality issues (detected but boundary problems)
    BOUNDARY_OVERSHOT = "BOUNDARY_OVERSHOT"
    BOUNDARY_UNDERSHOT = "BOUNDARY_UNDERSHOT"

    # Success
    SUCCESS = "SUCCESS"


@dataclass
class DiagnosticResult:
    """Result of failure diagnosis."""
    image_path: str
    success: bool
    primary_failure: str
    failure_codes: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)

    # Contour path diagnostics
    contour_count: int = 0
    quad_count: int = 0
    valid_quad_count: int = 0
    quad_rejection_reasons: List[str] = field(default_factory=list)

    # Hough path diagnostics
    line_count: int = 0
    horizontal_count: int = 0
    vertical_count: int = 0
    clustered_horizontal_count: int = 0
    clustered_vertical_count: int = 0
    intersection_count: int = 0

    # Area info
    largest_contour_area_ratio: float = 0.0
    smallest_contour_area_ratio: float = 0.0


class FailureDiagnoser:
    """Diagnose why detection failed on specific images."""

    def __init__(self, config: Optional[DetectionConfig] = None):
        """Initialize diagnoser.

        Args:
            config: Detection configuration to use.
        """
        self.config = config or DetectionConfig()

    def diagnose(self, image_path: str) -> DiagnosticResult:
        """Diagnose why detection failed on an image.

        Args:
            image_path: Path to the image.

        Returns:
            DiagnosticResult with failure analysis.
        """
        result = DiagnosticResult(
            image_path=image_path,
            success=False,
            primary_failure=FailureCode.LOAD_FAILED,
        )

        # Step 1: Load image
        image = load_and_validate(image_path)
        if image is None:
            result.failure_codes.append(FailureCode.LOAD_FAILED)
            return result

        # Check minimum size
        if min(image.shape[:2]) < self.config.min_input_size:
            result.failure_codes.append(FailureCode.IMAGE_TOO_SMALL)
            result.primary_failure = FailureCode.IMAGE_TOO_SMALL
            return result

        # Step 2: Preprocess
        resized, scale = resize_image(image, self.config.target_size)
        gray = to_grayscale(resized)
        enhanced = apply_clahe(
            gray,
            self.config.clahe_clip_limit,
            self.config.clahe_tile_size
        )

        # Step 3: Diagnose contour path
        self._diagnose_contour_path(enhanced, resized.shape, result)

        # Step 4: Diagnose Hough path
        self._diagnose_hough_path(enhanced, resized.shape, result)

        # Determine primary failure
        result.primary_failure = self._determine_primary_failure(result)

        # Check if either path succeeded
        if result.valid_quad_count > 0 or (
            result.clustered_horizontal_count >= 8 and
            result.clustered_vertical_count >= 8 and
            result.intersection_count >= 4
        ):
            result.success = True
            result.primary_failure = FailureCode.SUCCESS

        return result

    def _diagnose_contour_path(
        self,
        enhanced: np.ndarray,
        image_shape: Tuple[int, int, int],
        result: DiagnosticResult,
    ):
        """Diagnose the contour detection path.

        Args:
            enhanced: CLAHE-enhanced grayscale image.
            image_shape: Shape of resized image.
            result: DiagnosticResult to populate.
        """
        # Get binary image
        binary = adaptive_threshold(
            enhanced,
            self.config.adaptive_block_size,
            self.config.adaptive_c
        )
        if self.config.use_morphological:
            binary = apply_morphological(binary, self.config.morph_kernel_size)

        image_area = image_shape[0] * image_shape[1]
        min_area = self.config.min_area_ratio * image_area
        max_area = self.config.max_area_ratio * image_area

        # Find all contours (not just quads)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        result.contour_count = len(contours)

        if result.contour_count == 0:
            result.failure_codes.append(FailureCode.NO_CONTOURS)
            return

        # Analyze contours
        contour_areas = [cv2.contourArea(c) for c in contours]
        if contour_areas:
            result.largest_contour_area_ratio = max(contour_areas) / image_area
            result.smallest_contour_area_ratio = min(contour_areas) / image_area

        # Check for area rejections
        area_filtered = []
        has_small_rejection = False
        has_large_rejection = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                has_small_rejection = True
            elif area > max_area:
                has_large_rejection = True
            else:
                area_filtered.append(contour)

        if has_small_rejection:
            result.failure_codes.append(FailureCode.AREA_REJECTED_SMALL)
        if has_large_rejection:
            result.failure_codes.append(FailureCode.AREA_REJECTED_LARGE)

        # Find quadrilaterals
        quads = []
        for contour in area_filtered:
            perimeter = cv2.arcLength(contour, closed=True)
            epsilon = self.config.contour_epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            if len(approx) == 4:
                quads.append(approx.reshape(4, 2))

        result.quad_count = len(quads)

        if result.quad_count == 0:
            result.failure_codes.append(FailureCode.NO_QUADRILATERALS)
            return

        # Validate quadrilaterals
        valid_count = 0
        for quad in quads:
            is_valid, rejection_reason = self._validate_quad_with_reason(quad)
            if is_valid:
                valid_count += 1
            else:
                result.quad_rejection_reasons.append(rejection_reason)
                if rejection_reason not in result.failure_codes:
                    result.failure_codes.append(rejection_reason)

        result.valid_quad_count = valid_count

        if valid_count == 0:
            result.failure_codes.append(FailureCode.ALL_QUADS_INVALID)

    def _validate_quad_with_reason(
        self,
        quad: np.ndarray,
    ) -> Tuple[bool, Optional[str]]:
        """Validate quadrilateral and return rejection reason.

        Args:
            quad: Quadrilateral as (4, 2) array.

        Returns:
            Tuple of (is_valid, rejection_reason or None).
        """
        # Check convexity
        quad_for_cv = quad.reshape(4, 1, 2).astype(np.float32)
        if not cv2.isContourConvex(quad_for_cv):
            return False, FailureCode.NOT_CONVEX

        # Check aspect ratio
        x, y, w, h = cv2.boundingRect(quad_for_cv)
        if h == 0:
            return False, FailureCode.ASPECT_REJECTED
        aspect_ratio = w / h
        if aspect_ratio < self.config.min_aspect_ratio:
            return False, FailureCode.ASPECT_REJECTED
        if aspect_ratio > self.config.max_aspect_ratio:
            return False, FailureCode.ASPECT_REJECTED

        # Check interior angles
        for i in range(4):
            p1 = quad[(i - 1) % 4].astype(np.float64)
            vertex = quad[i].astype(np.float64)
            p2 = quad[(i + 1) % 4].astype(np.float64)

            v1 = p1 - vertex
            v2 = p2 - vertex
            dot = np.dot(v1, v2)
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)

            if mag1 == 0 or mag2 == 0:
                return False, FailureCode.ANGLE_REJECTED

            cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            if angle < self.config.min_interior_angle:
                return False, FailureCode.ANGLE_REJECTED
            if angle > self.config.max_interior_angle:
                return False, FailureCode.ANGLE_REJECTED

        return True, None

    def _diagnose_hough_path(
        self,
        enhanced: np.ndarray,
        image_shape: Tuple[int, int, int],
        result: DiagnosticResult,
    ):
        """Diagnose the Hough line detection path.

        Args:
            enhanced: CLAHE-enhanced grayscale image.
            image_shape: Shape of resized image.
            result: DiagnosticResult to populate.
        """
        # Get edges
        edges = get_edges(
            enhanced,
            self.config.canny_low,
            self.config.canny_high,
            blur_size=5
        )

        if edges is None or np.sum(edges) == 0:
            result.failure_codes.append(FailureCode.NO_EDGES)
            return

        # Detect lines
        lines = detect_lines(edges, self.config)
        if lines is None or len(lines) == 0:
            result.failure_codes.append(FailureCode.HOUGH_NO_LINES)
            return

        result.line_count = len(lines)

        # Classify lines
        h_lines, v_lines = classify_lines(lines)
        result.horizontal_count = len(h_lines)
        result.vertical_count = len(v_lines)

        if len(h_lines) < 3:
            result.failure_codes.append(FailureCode.HOUGH_FEW_HORIZONTAL)
        if len(v_lines) < 3:
            result.failure_codes.append(FailureCode.HOUGH_FEW_VERTICAL)

        if not h_lines or not v_lines:
            return

        # Cluster lines
        height, width = image_shape[:2]
        h_threshold = height / self.config.line_cluster_divisor
        v_threshold = width / self.config.line_cluster_divisor

        h_clustered = cluster_lines(h_lines, h_threshold, is_horizontal=True)
        v_clustered = cluster_lines(v_lines, v_threshold, is_horizontal=False)

        result.clustered_horizontal_count = len(h_clustered)
        result.clustered_vertical_count = len(v_clustered)

        # Check line counts
        if not validate_grid_lines(
            h_clustered, v_clustered,
            self.config.min_line_count,
            self.config.max_line_count
        ):
            if len(h_clustered) < self.config.min_line_count or len(v_clustered) < self.config.min_line_count:
                result.failure_codes.append(FailureCode.HOUGH_LINE_COUNT_LOW)
            else:
                result.failure_codes.append(FailureCode.HOUGH_LINE_COUNT_HIGH)
            return

        # Compute intersections
        intersections = compute_all_intersections(h_clustered, v_clustered, image_shape[:2])
        result.intersection_count = len(intersections)

        if len(intersections) < 4:
            result.failure_codes.append(FailureCode.HOUGH_FEW_INTERSECTIONS)
            return

        # Extract corners
        corners = extract_outer_corners(intersections)
        if corners is None:
            result.failure_codes.append(FailureCode.HOUGH_CORNER_EXTRACTION_FAILED)

    def _determine_primary_failure(self, result: DiagnosticResult) -> str:
        """Determine the primary failure cause.

        Args:
            result: DiagnosticResult with failure codes.

        Returns:
            Primary failure code.
        """
        # Priority order for primary failure
        priority = [
            FailureCode.LOAD_FAILED,
            FailureCode.IMAGE_TOO_SMALL,
            FailureCode.NO_CONTOURS,
            FailureCode.NO_QUADRILATERALS,
            FailureCode.ANGLE_REJECTED,
            FailureCode.ASPECT_REJECTED,
            FailureCode.AREA_REJECTED_SMALL,
            FailureCode.AREA_REJECTED_LARGE,
            FailureCode.HOUGH_NO_LINES,
            FailureCode.HOUGH_LINE_COUNT_LOW,
            FailureCode.HOUGH_LINE_COUNT_HIGH,
            FailureCode.ALL_QUADS_INVALID,
        ]

        for code in priority:
            if code in result.failure_codes:
                return code

        if result.failure_codes:
            return result.failure_codes[0]

        return FailureCode.SUCCESS


def diagnose_image(image_path: str, config: Optional[DetectionConfig] = None) -> DiagnosticResult:
    """Convenience function to diagnose a single image.

    Args:
        image_path: Path to the image.
        config: Optional detection configuration.

    Returns:
        DiagnosticResult with failure analysis.
    """
    diagnoser = FailureDiagnoser(config)
    return diagnoser.diagnose(image_path)


def diagnose_batch(
    image_paths: List[str],
    config: Optional[DetectionConfig] = None,
) -> Dict[str, DiagnosticResult]:
    """Diagnose a batch of images.

    Args:
        image_paths: List of image paths.
        config: Optional detection configuration.

    Returns:
        Dictionary mapping image paths to DiagnosticResults.
    """
    diagnoser = FailureDiagnoser(config)
    results = {}

    for path in image_paths:
        results[path] = diagnoser.diagnose(path)

    return results


def summarize_failures(results: Dict[str, DiagnosticResult]) -> Dict[str, int]:
    """Summarize failure codes across multiple diagnoses.

    Args:
        results: Dictionary of DiagnosticResults.

    Returns:
        Dictionary mapping failure codes to counts.
    """
    failure_counts = {}

    for result in results.values():
        code = result.primary_failure
        failure_counts[code] = failure_counts.get(code, 0) + 1

    return dict(sorted(failure_counts.items(), key=lambda x: -x[1]))
