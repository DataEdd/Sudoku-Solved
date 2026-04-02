"""Debug and visualization utilities for sudoku detection."""

from typing import Optional, Tuple

import cv2
import numpy as np

from .detector import DetectionResult


def draw_corners(
    image: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw corner points and connecting lines on an image.

    Args:
        image: Input image (BGR).
        corners: Corner points with shape (4, 2), ordered [TL, TR, BR, BL].
        color: BGR color tuple for drawing.
        thickness: Line and circle thickness.

    Returns:
        Annotated copy of the image.
    """
    result = image.copy()

    # Ensure corners are integers for drawing
    corners = corners.astype(np.int32)

    # Corner labels
    labels = ["TL", "TR", "BR", "BL"]

    # Draw lines connecting corners
    for i in range(4):
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[(i + 1) % 4])
        cv2.line(result, pt1, pt2, color, thickness)

    # Draw circles and labels at each corner
    for i, (corner, label) in enumerate(zip(corners, labels)):
        pt = tuple(corner)

        # Draw filled circle
        cv2.circle(result, pt, 8, color, -1)

        # Draw label with background for visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        label_thickness = 2

        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, label_thickness
        )

        # Position label slightly offset from corner
        label_x = pt[0] + 12
        label_y = pt[1] + 5

        # Draw background rectangle
        cv2.rectangle(
            result,
            (label_x - 2, label_y - text_h - 2),
            (label_x + text_w + 2, label_y + baseline + 2),
            (0, 0, 0),
            -1,
        )

        # Draw text
        cv2.putText(
            result, label, (label_x, label_y),
            font, font_scale, color, label_thickness
        )

    return result


def draw_detection_result(
    image: np.ndarray,
    result: DetectionResult,
) -> np.ndarray:
    """Draw detection result on an image.

    Args:
        image: Input image (BGR).
        result: Detection result from SudokuDetector.

    Returns:
        Annotated copy of the image.
    """
    annotated = image.copy()

    if result.success:
        # Draw corners in green
        annotated = draw_corners(annotated, result.corners, color=(0, 255, 0))

        # Add success info text
        info_text = f"{result.detection_method.upper()} | Conf: {result.confidence:.3f}"
        cv2.putText(
            annotated, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
    else:
        # Add failure text in red
        error_msg = result.error_message or "Detection failed"

        # Draw semi-transparent red overlay at top
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (annotated.shape[1], 50), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

        # Draw error text
        cv2.putText(
            annotated, f"FAILED: {error_msg}", (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

    return annotated


def create_side_by_side(
    original: np.ndarray,
    warped: Optional[np.ndarray],
    max_height: int = 500,
) -> np.ndarray:
    """Create side-by-side comparison of original and warped images.

    Args:
        original: Original input image.
        warped: Warped sudoku grid image, or None if detection failed.
        max_height: Maximum height for both images.

    Returns:
        Combined side-by-side image.
    """
    # Resize original to max_height
    h, w = original.shape[:2]
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        original_resized = cv2.resize(original, (new_w, max_height))
    else:
        original_resized = original.copy()

    # Handle warped image
    if warped is not None:
        # Resize warped to same height
        wh, ww = warped.shape[:2]
        if wh != original_resized.shape[0]:
            scale = original_resized.shape[0] / wh
            new_ww = int(ww * scale)
            warped_resized = cv2.resize(
                warped, (new_ww, original_resized.shape[0])
            )
        else:
            warped_resized = warped.copy()

        # Ensure both have same number of channels
        if len(original_resized.shape) == 2:
            original_resized = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)
        if len(warped_resized.shape) == 2:
            warped_resized = cv2.cvtColor(warped_resized, cv2.COLOR_GRAY2BGR)
    else:
        # Create gray placeholder
        placeholder_w = original_resized.shape[0]  # Square placeholder
        warped_resized = np.full(
            (original_resized.shape[0], placeholder_w, 3),
            128,
            dtype=np.uint8,
        )

        # Add "Detection Failed" text
        text = "Detection Failed"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (placeholder_w - text_w) // 2
        text_y = (original_resized.shape[0] + text_h) // 2

        cv2.putText(
            warped_resized, text, (text_x, text_y),
            font, font_scale, (0, 0, 255), thickness
        )

    # Add labels
    cv2.putText(
        original_resized, "Original", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    cv2.putText(
        warped_resized, "Extracted", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    # Create separator line
    separator = np.full(
        (original_resized.shape[0], 5, 3), 255, dtype=np.uint8
    )

    # Combine horizontally
    combined = np.hstack([original_resized, separator, warped_resized])

    return combined
