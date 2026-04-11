"""
Visualization utilities for cell detection notebooks.

Provides functions for displaying images, drawing detected features,
and creating debug montages for step-by-step analysis.
"""

from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_image(
    image: np.ndarray,
    title: str = "",
    figsize: Tuple[int, int] = (8, 8),
    cmap: Optional[str] = None,
) -> None:
    """
    Display a single image with matplotlib.

    Args:
        image: Image to display (BGR or grayscale)
        title: Title for the plot
        figsize: Figure size as (width, height)
        cmap: Colormap for grayscale images (default: 'gray')
    """
    plt.figure(figsize=figsize)

    if len(image.shape) == 2:
        # Grayscale
        plt.imshow(image, cmap=cmap or "gray")
    else:
        # BGR to RGB for display
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_images_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    cols: int = 4,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Display multiple images in a grid layout.

    Args:
        images: List of images to display
        titles: Optional list of titles for each image
        cols: Number of columns in the grid
        figsize: Figure size (auto-calculated if None)
    """
    n = len(images)
    rows = (n + cols - 1) // cols

    if figsize is None:
        figsize = (4 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, img in enumerate(images):
        ax = axes[i]

        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10)
        ax.axis("off")

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def draw_corners(
    image: np.ndarray,
    corners: np.ndarray,
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 0, 255),
    radius: int = 8,
    thickness: int = -1,
) -> np.ndarray:
    """
    Draw corner points on an image.

    Args:
        image: Input image (will be copied)
        corners: Array of corner points, shape (N, 2)
        labels: Optional labels for each corner
        color: BGR color for the points
        radius: Radius of the corner circles
        thickness: -1 for filled circles

    Returns:
        Image with corners drawn
    """
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    if labels is None:
        labels = ["TL", "TR", "BR", "BL"]

    corners = corners.astype(np.int32)

    for i, corner in enumerate(corners):
        pt = tuple(corner)
        cv2.circle(result, pt, radius, color, thickness)

        if i < len(labels):
            cv2.putText(
                result,
                labels[i],
                (pt[0] + 12, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    return result


def draw_quadrilateral(
    image: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    draw_corners: bool = True,
) -> np.ndarray:
    """
    Draw a quadrilateral on an image.

    Args:
        image: Input image (will be copied)
        corners: 4x2 array of corner points [TL, TR, BR, BL]
        color: BGR color for the lines
        thickness: Line thickness
        draw_corners: Whether to also draw corner points

    Returns:
        Image with quadrilateral drawn
    """
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    corners = corners.astype(np.int32)

    # Draw the four sides
    for i in range(4):
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[(i + 1) % 4])
        cv2.line(result, pt1, pt2, color, thickness)

    # Optionally draw corners
    if draw_corners:
        for i, corner in enumerate(corners):
            cv2.circle(result, tuple(corner), 6, (0, 0, 255), -1)

    return result


def draw_cell_highlight(
    image: np.ndarray,
    cell_corners: np.ndarray,
    fill_color: Tuple[int, int, int] = (0, 255, 255),
    fill_alpha: float = 0.3,
    border_color: Tuple[int, int, int] = (0, 255, 0),
    border_thickness: int = 2,
) -> np.ndarray:
    """
    Highlight a cell with semi-transparent fill and border.

    Args:
        image: Input image (will be copied)
        cell_corners: 4x2 array of cell corner points
        fill_color: BGR color for the fill
        fill_alpha: Transparency of fill (0-1)
        border_color: BGR color for the border
        border_thickness: Border line thickness

    Returns:
        Image with cell highlighted
    """
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    corners = cell_corners.astype(np.int32)

    # Create overlay for semi-transparent fill
    overlay = result.copy()
    cv2.fillPoly(overlay, [corners], fill_color)

    # Blend overlay with original
    result = cv2.addWeighted(overlay, fill_alpha, result, 1 - fill_alpha, 0)

    # Draw border
    cv2.polylines(result, [corners], True, border_color, border_thickness)

    return result


def create_debug_montage(
    steps: Dict[str, np.ndarray],
    title: str = "Processing Steps",
    cols: int = 4,
) -> np.ndarray:
    """
    Create a montage image showing all processing steps.

    Args:
        steps: Dictionary mapping step names to images
        title: Overall title for the montage
        cols: Number of columns

    Returns:
        Combined montage image
    """
    images = list(steps.values())
    titles = list(steps.keys())

    # Normalize all images to same size and color space
    target_h, target_w = 300, 300
    normalized = []

    for img in images:
        # Convert to BGR if grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Resize maintaining aspect ratio
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # Pad to target size
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        padded[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # Add title
        cv2.putText(
            padded,
            titles[len(normalized)] if len(normalized) < len(titles) else "",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        normalized.append(padded)

    # Arrange in grid
    n = len(normalized)
    rows = (n + cols - 1) // cols

    # Pad to fill grid
    while len(normalized) < rows * cols:
        normalized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    # Create montage
    row_images = []
    for r in range(rows):
        row_imgs = normalized[r * cols:(r + 1) * cols]
        row_images.append(np.hstack(row_imgs))

    montage = np.vstack(row_images)

    return montage
