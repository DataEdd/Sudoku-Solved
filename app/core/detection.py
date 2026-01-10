"""
Unified grid detection interface with step-by-step visualization.

Provides a consistent interface for comparing different detection methods:
- Contour-based detection (from main.py)
- Standard Hough Transform (HoughLinesP)
- Polar Hough Transform (HoughLines)

Each method returns all intermediate preprocessing steps for debugging.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.preprocessing import (
    to_grayscale,
    apply_blur,
    adaptive_threshold,
    preprocess_for_hough_polar_full,
)
from app.core.hough_standard import (
    detect_lines,
    classify_lines,
    cluster_lines,
    find_grid_lines,
    compute_intersections,
    compute_confidence,
    draw_lines_on_image,
    draw_grid_overlay,
    draw_intersections,
    detect_lines_polar,
    filter_similar_lines,
    classify_lines_by_theta,
    filter_by_dominant_angles,
    draw_polar_lines,
)
from app.config import get_settings


@dataclass
class DetectionResult:
    """Result from any detection method with all intermediate steps."""

    method: str
    success: bool
    confidence: float
    steps: Dict[str, np.ndarray] = field(default_factory=dict)
    stats: Dict[str, any] = field(default_factory=dict)

    @property
    def result_image(self) -> Optional[np.ndarray]:
        """Get the final result image."""
        # Find the highest numbered step (the result)
        if not self.steps:
            return None
        last_key = sorted(self.steps.keys())[-1]
        return self.steps[last_key]


def detect_contour(image: np.ndarray) -> DetectionResult:
    """
    Detect grid using contour-based method (find largest quadrilateral).

    Pipeline:
    1. Grayscale conversion
    2. Gaussian blur
    3. Adaptive threshold
    4. Find contours
    5. Select largest quadrilateral

    Args:
        image: BGR input image

    Returns:
        DetectionResult with all preprocessing steps
    """
    steps = {}
    stats = {}

    # Step 1: Grayscale
    gray = to_grayscale(image)
    steps["01_grayscale"] = gray

    # Step 2: Blur
    blurred = apply_blur(gray, kernel_size=5)
    steps["02_blurred"] = blurred

    # Step 3: Threshold
    thresh = adaptive_threshold(blurred, block_size=11, c=2)
    steps["03_threshold"] = thresh

    # Step 4: Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats["total_contours"] = len(contours)

    # Draw all contours on original
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

    # Find the largest quadrilateral
    grid_contour = None
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                grid_contour = approx
                stats["contour_area"] = int(cv2.contourArea(approx))
                stats["vertices"] = 4
                # Highlight the detected grid contour
                cv2.drawContours(contour_img, [approx], -1, (0, 0, 255), 4)
                break

    steps["04_contours"] = contour_img

    # Step 5: Result with perspective transform visualization
    result_img = image.copy()
    success = grid_contour is not None

    if success:
        # Draw the detected quadrilateral
        cv2.drawContours(result_img, [grid_contour], -1, (0, 255, 0), 3)
        # Draw corner points
        for point in grid_contour:
            x, y = point[0]
            cv2.circle(result_img, (x, y), 8, (0, 0, 255), -1)

        # Calculate confidence based on area ratio
        img_area = image.shape[0] * image.shape[1]
        contour_area = cv2.contourArea(grid_contour)
        area_ratio = contour_area / img_area
        # Good detection: grid covers 20-80% of image
        if 0.2 <= area_ratio <= 0.8:
            confidence = 0.8 + 0.2 * (1 - abs(0.5 - area_ratio) / 0.3)
        else:
            confidence = max(0.1, 1 - abs(0.5 - area_ratio))
    else:
        confidence = 0.0
        # Draw "No grid found" text
        cv2.putText(result_img, "No grid found", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    steps["05_result"] = result_img
    stats["success"] = success

    return DetectionResult(
        method="contour",
        success=success,
        confidence=confidence,
        steps=steps,
        stats=stats
    )


def detect_hough_standard(image: np.ndarray) -> DetectionResult:
    """
    Detect grid using Standard Hough Transform (HoughLinesP).

    Pipeline:
    1. Grayscale conversion
    2. Gaussian blur
    3. Adaptive threshold
    4. Detect all lines (HoughLinesP)
    5. Classify lines (horizontal/vertical)
    6. Cluster and interpolate to 10 lines each

    Args:
        image: BGR input image

    Returns:
        DetectionResult with all preprocessing steps
    """
    settings = get_settings()
    steps = {}
    stats = {}

    # Step 1: Grayscale
    gray = to_grayscale(image)
    steps["01_grayscale"] = gray

    # Step 2: Blur
    blurred = apply_blur(gray, kernel_size=5)
    steps["02_blurred"] = blurred

    # Step 3: Threshold
    thresh = adaptive_threshold(blurred, block_size=11, c=2)
    steps["03_threshold"] = thresh

    # Step 4: Detect all lines
    all_lines = detect_lines(
        thresh,
        threshold=settings.hough_threshold,
        min_line_length=settings.hough_min_line_length,
        max_line_gap=settings.hough_max_line_gap
    )
    stats["total_lines"] = len(all_lines)

    all_lines_img = draw_lines_on_image(image, all_lines, color=(0, 0, 255), thickness=1)
    steps["04_all_lines"] = all_lines_img

    # Step 5: Classify lines
    h_lines, v_lines = classify_lines(all_lines, settings.line_angle_threshold)
    stats["h_lines"] = len(h_lines)
    stats["v_lines"] = len(v_lines)

    classified_img = image.copy()
    classified_img = draw_lines_on_image(classified_img, h_lines, color=(255, 0, 0), thickness=2)
    classified_img = draw_lines_on_image(classified_img, v_lines, color=(0, 255, 0), thickness=2)
    steps["05_classified"] = classified_img

    # Step 6: Cluster and create result
    success = False
    confidence = 0.0
    result_img = image.copy()

    if len(h_lines) >= 2 and len(v_lines) >= 2:
        h_clustered = cluster_lines(h_lines, is_horizontal=True,
                                     distance_threshold=settings.line_cluster_distance)
        v_clustered = cluster_lines(v_lines, is_horizontal=False,
                                     distance_threshold=settings.line_cluster_distance)

        stats["h_clustered"] = len(h_clustered)
        stats["v_clustered"] = len(v_clustered)

        try:
            h_final, v_final = find_grid_lines(h_clustered, v_clustered, target_count=10)
            intersections = compute_intersections(h_final, v_final)
            confidence = compute_confidence(h_final, v_final)

            result_img = draw_grid_overlay(image, h_final, v_final)
            result_img = draw_intersections(result_img, intersections)
            success = True
        except ValueError:
            cv2.putText(result_img, "Could not find 10 lines", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(result_img, f"Not enough lines (H:{len(h_lines)}, V:{len(v_lines)})",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    steps["06_result"] = result_img
    stats["success"] = success

    return DetectionResult(
        method="hough",
        success=success,
        confidence=confidence,
        steps=steps,
        stats=stats
    )


def detect_hough_polar(image: np.ndarray) -> DetectionResult:
    """
    Detect grid using Polar Hough Transform (HoughLines).

    Pipeline:
    1. Grayscale conversion
    2. Canny edge detection
    3. Morphological dilation (connect gaps)
    4. Morphological erosion (remove noise)
    5. Detect all lines (HoughLines in polar form)
    6. Filter similar lines
    7. Final result

    Args:
        image: BGR input image

    Returns:
        DetectionResult with all preprocessing steps
    """
    settings = get_settings()
    steps = {}
    stats = {}

    # Steps 1-4: Preprocessing with morphological operations
    preprocess = preprocess_for_hough_polar_full(
        image,
        canny_low=settings.canny_low,
        canny_high=settings.canny_high,
        dilate_kernel_size=settings.dilate_kernel_size,
        erode_kernel_size=settings.erode_kernel_size
    )

    steps["01_grayscale"] = preprocess.grayscale
    steps["02_canny"] = preprocess.edges
    steps["03_dilated"] = preprocess.dilated
    steps["04_eroded"] = preprocess.eroded

    # Step 5: Detect all lines
    all_lines = detect_lines_polar(preprocess.eroded, threshold=settings.hough_polar_threshold)
    n_all = len(all_lines) if all_lines is not None else 0
    stats["total_lines"] = n_all

    if all_lines is not None and len(all_lines) > 0:
        all_lines_img = draw_polar_lines(
            image,
            [tuple(line[0]) for line in all_lines],
            color=(0, 0, 255),
            thickness=1
        )
    else:
        all_lines_img = image.copy()
        cv2.putText(all_lines_img, "No lines detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    steps["05_all_lines"] = all_lines_img

    # Step 6: Filter similar lines
    success = False
    confidence = 0.0
    result_img = image.copy()

    if all_lines is not None and len(all_lines) > 0:
        # First filter by dominant angles
        angle_filtered = filter_by_dominant_angles(all_lines, angle_tolerance=0.2)
        lines_for_similarity = np.array([[[r, t]] for r, t in angle_filtered])

        # Then filter similar lines
        if len(lines_for_similarity) > 0:
            filtered_lines = filter_similar_lines(
                lines_for_similarity,
                rho_threshold=settings.rho_threshold,
                theta_threshold=settings.theta_threshold
            )
        else:
            filtered_lines = []

        stats["filtered_lines"] = len(filtered_lines)

        if filtered_lines:
            filtered_img = draw_polar_lines(image, filtered_lines, color=(0, 255, 0), thickness=2)
            steps["06_filtered"] = filtered_img

            # Classify into H/V
            h_lines, v_lines = classify_lines_by_theta(filtered_lines)
            stats["h_lines"] = len(h_lines)
            stats["v_lines"] = len(v_lines)

            # Draw result with H=blue, V=green
            result_img = image.copy()
            result_img = draw_polar_lines(result_img, h_lines, color=(255, 0, 0), thickness=2)
            result_img = draw_polar_lines(result_img, v_lines, color=(0, 255, 0), thickness=2)

            # Confidence based on line count (expecting ~20 lines for Sudoku)
            expected_lines = 20
            confidence = min(1.0, len(filtered_lines) / expected_lines)
            success = len(filtered_lines) >= 10
        else:
            steps["06_filtered"] = image.copy()
            cv2.putText(result_img, "No lines after filtering", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        steps["06_filtered"] = image.copy()

    steps["07_result"] = result_img
    stats["success"] = success

    return DetectionResult(
        method="polar",
        success=success,
        confidence=confidence,
        steps=steps,
        stats=stats
    )


DETECTION_METHODS = {
    "contour": detect_contour,
    "hough": detect_hough_standard,
    "polar": detect_hough_polar,
}


def detect_all(image: np.ndarray) -> Dict[str, DetectionResult]:
    """
    Run all detection methods on an image.

    Args:
        image: BGR input image

    Returns:
        Dict mapping method name to DetectionResult
    """
    results = {}
    for name, detect_fn in DETECTION_METHODS.items():
        results[name] = detect_fn(image)
    return results


def create_comparison_image(
    results: Dict[str, DetectionResult],
    image_height: int = 400
) -> np.ndarray:
    """
    Create a side-by-side comparison of final results from all methods.

    Args:
        results: Dict of DetectionResult from each method
        image_height: Height to resize images to

    Returns:
        Combined image showing all results side-by-side
    """
    images = []
    labels = []

    for method_name, result in results.items():
        img = result.result_image
        if img is None:
            continue

        # Resize to consistent height
        h, w = img.shape[:2]
        new_w = int(w * image_height / h)
        img_resized = cv2.resize(img, (new_w, image_height))

        # Add label at top
        labeled = np.zeros((image_height + 40, new_w, 3), dtype=np.uint8)
        labeled[40:, :] = img_resized

        # Draw label background and text
        label_text = f"{method_name.upper()} (conf: {result.confidence:.2f})"
        if result.success:
            color = (0, 255, 0)  # Green for success
        else:
            color = (0, 0, 255)  # Red for failure

        cv2.rectangle(labeled, (0, 0), (new_w, 40), (50, 50, 50), -1)
        cv2.putText(labeled, label_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        images.append(labeled)

    if not images:
        return np.zeros((image_height, 400, 3), dtype=np.uint8)

    return np.hstack(images)


def create_preprocessing_grid(
    results: Dict[str, DetectionResult],
    cell_height: int = 200
) -> np.ndarray:
    """
    Create a grid showing key preprocessing steps from each method.

    Layout:
    - Rows: preprocessing steps (grayscale, threshold/canny, result)
    - Columns: methods (contour, hough, polar)

    Args:
        results: Dict of DetectionResult from each method
        cell_height: Height of each cell

    Returns:
        Grid image comparing preprocessing steps
    """
    # Key steps to show for each method
    key_steps = {
        "contour": ["01_grayscale", "03_threshold", "05_result"],
        "hough": ["01_grayscale", "03_threshold", "06_result"],
        "polar": ["01_grayscale", "02_canny", "07_result"],
    }

    row_labels = ["Grayscale", "Edge/Threshold", "Result"]
    methods = ["contour", "hough", "polar"]

    # Calculate cell width based on first image
    sample_img = None
    for result in results.values():
        if result.steps:
            sample_img = list(result.steps.values())[0]
            break

    if sample_img is None:
        return np.zeros((cell_height * 3, 400, 3), dtype=np.uint8)

    h, w = sample_img.shape[:2]
    cell_width = int(w * cell_height / h)

    # Create grid
    label_width = 120
    grid_width = label_width + cell_width * len(methods)
    grid_height = 30 + cell_height * len(row_labels)

    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    grid[:] = (40, 40, 40)  # Dark gray background

    # Draw column headers
    for i, method in enumerate(methods):
        x = label_width + i * cell_width + cell_width // 2 - 30
        cv2.putText(grid, method.upper(), (x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Fill in cells
    for row_idx, (_, step_keys) in enumerate(zip(row_labels, zip(*[key_steps[m] for m in methods]))):
        y_start = 30 + row_idx * cell_height

        # Draw row label
        cv2.putText(grid, row_labels[row_idx], (5, y_start + cell_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for col_idx, method in enumerate(methods):
            step_key = key_steps[method][row_idx]
            result = results.get(method)

            if result and step_key in result.steps:
                img = result.steps[step_key]

                # Convert grayscale to BGR for display
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # Resize
                img_resized = cv2.resize(img, (cell_width, cell_height))

                # Place in grid
                x_start = label_width + col_idx * cell_width
                grid[y_start:y_start + cell_height, x_start:x_start + cell_width] = img_resized

    return grid
