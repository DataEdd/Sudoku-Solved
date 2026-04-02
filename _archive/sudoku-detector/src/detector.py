"""Main sudoku grid detector orchestrating the full detection pipeline."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import DetectionConfig
from .contour_detection import detect_contour_path
from .geometry import (
    compute_homography,
    order_corners,
    refine_corners,
    scale_corners_to_original,
    warp_perspective,
)
from .preprocessing import (
    load_and_validate,
    preprocess_for_contour,
    resize_image,
    to_grayscale,
)


@dataclass
class DetectionResult:
    """Result of sudoku grid detection.

    Attributes:
        success: Whether detection was successful.
        warped_image: Perspective-corrected sudoku grid image (output_size x output_size).
        corners: Detected corner coordinates in original image coordinates.
        corners_normalized: Detected corner coordinates in resized image coordinates.
        homography: 3x3 homography matrix used for perspective correction.
        confidence: Detection confidence score (0.0 to 1.0, based on centeredness).
        detection_method: Method that produced the result ("contour" or "hough").
        error_message: Error message if detection failed.
    """

    success: bool
    warped_image: Optional[np.ndarray] = None
    corners: Optional[np.ndarray] = None
    corners_normalized: Optional[np.ndarray] = None
    homography: Optional[np.ndarray] = None
    confidence: float = 0.0
    detection_method: Optional[str] = None
    error_message: Optional[str] = None


class SudokuDetector:
    """Detector for extracting sudoku grids from images.

    Uses contour-based detection to find quadrilateral contours that
    represent the sudoku grid boundary. The best result is selected
    based on centeredness score.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        """Initialize the detector.

        Args:
            config: Detection configuration. If None, uses default config.
        """
        self.config = config if config is not None else DetectionConfig()

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect and extract sudoku grid from an image.

        Args:
            image: Input image (BGR color or grayscale).

        Returns:
            DetectionResult with detection outcome and extracted grid.
        """
        # Step a: Validate input image
        is_valid, error_msg = self._validate_input(image)
        if not is_valid:
            return DetectionResult(success=False, error_message=error_msg)

        # Step b: Resize image and store scale factor
        resized, scale_factor = resize_image(image, self.config.target_size)

        # Step c: Store reference to original image for final warp
        original_image = image

        # Step d: Preprocess for contour path
        binary, _, _ = preprocess_for_contour(image, self.config)

        # Step e: Run contour detection
        contour_candidates = detect_contour_path(binary, self.config)

        # Step f: Check if any candidates found
        if not contour_candidates:
            return DetectionResult(
                success=False,
                error_message="No sudoku grid detected",
            )

        # Step g: Select best candidate by highest centeredness score
        best_candidate = self._select_best_candidate(contour_candidates)

        if best_candidate is None:
            return DetectionResult(
                success=False,
                error_message="No valid candidates found",
            )

        # Step k: Order corners
        corners_normalized = order_corners(best_candidate["corners"])

        # Step l: Optionally refine corners with sub-pixel accuracy
        gray_resized = to_grayscale(resized)
        try:
            corners_refined = refine_corners(
                gray_resized, corners_normalized, window_size=5
            )
        except Exception:
            # If refinement fails, use original corners
            corners_refined = corners_normalized

        # Step m: Scale corners back to original image coordinates
        corners_original = scale_corners_to_original(corners_refined, scale_factor)

        # Step n: Compute homography
        homography = compute_homography(corners_original, self.config.output_size)

        # Step o: Warp ORIGINAL image to get final output
        warped = warp_perspective(
            original_image, homography, self.config.output_size
        )

        # Step p: Return complete result
        return DetectionResult(
            success=True,
            warped_image=warped,
            corners=corners_original,
            corners_normalized=corners_refined,
            homography=homography,
            confidence=best_candidate["score"],
            detection_method=best_candidate["method"],
            error_message=None,
        )

    def detect_from_file(self, filepath: str) -> DetectionResult:
        """Detect sudoku grid from an image file.

        Args:
            filepath: Path to the image file.

        Returns:
            DetectionResult with detection outcome and extracted grid.
        """
        # Load and validate image
        image = load_and_validate(filepath, self.config.min_input_size)

        if image is None:
            return DetectionResult(
                success=False,
                error_message="Failed to load image",
            )

        # Run detection
        return self.detect(image)

    def _validate_input(self, image: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Validate input image.

        Args:
            image: Input image to validate.

        Returns:
            Tuple of (is_valid, error_message).
            If valid, returns (True, None).
            If invalid, returns (False, error_message).
        """
        # Check image is not None
        if image is None:
            return False, "Input image is None"

        # Check image has valid shape
        if not isinstance(image, np.ndarray):
            return False, "Input must be a numpy array"

        if len(image.shape) < 2 or len(image.shape) > 3:
            return False, f"Invalid image dimensions: {len(image.shape)}"

        # Check minimum dimensions
        height, width = image.shape[:2]
        min_size = self.config.min_input_size

        if height < min_size or width < min_size:
            return False, f"Image too small: {width}x{height}, minimum is {min_size}x{min_size}"

        return True, None

    def _select_best_candidate(
        self, candidates: List[dict]
    ) -> Optional[dict]:
        """Select the best candidate from a list.

        Args:
            candidates: List of candidate dictionaries with "score" key.

        Returns:
            Best candidate (highest score) or None if list is empty.
        """
        if not candidates:
            return None

        # Sort by score descending
        sorted_candidates = sorted(
            candidates, key=lambda x: x["score"], reverse=True
        )

        return sorted_candidates[0]


def main():
    """CLI entry point for sudoku detection."""
    import sys

    import cv2

    if len(sys.argv) < 2:
        print("Usage: python -m src.detector <image_path> [output_path]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "sudoku_output.jpg"

    detector = SudokuDetector()
    result = detector.detect_from_file(input_path)

    if result.success:
        cv2.imwrite(output_path, result.warped_image)
        print(f"Success! Saved to {output_path}")
        print(f"Detection method: {result.detection_method}")
        print(f"Confidence: {result.confidence:.3f}")
    else:
        print(f"Detection failed: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
