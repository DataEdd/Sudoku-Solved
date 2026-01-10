"""
Pytest configuration and fixtures for visual testing.

This module provides fixtures for:
- Loading sample Sudoku images
- Displaying images interactively with matplotlib
- Generating synthetic test images
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============== Display Fixtures ==============

@pytest.fixture
def display_images() -> Callable:
    """
    Fixture that returns a function to display multiple images side-by-side.

    Usage:
        def test_something(display_images):
            display_images({
                "Original": image1,
                "Processed": image2,
            }, title="My Test")
    """
    import matplotlib.pyplot as plt

    def _display(
        images: Dict[str, np.ndarray],
        title: str = "",
        figsize_per_image: float = 4.0
    ):
        """
        Display multiple images in a row.

        Args:
            images: Dict mapping name to image array
            title: Overall figure title
            figsize_per_image: Width per image in inches
        """
        n = len(images)
        if n == 0:
            return

        fig, axes = plt.subplots(1, n, figsize=(figsize_per_image * n, figsize_per_image))

        if n == 1:
            axes = [axes]

        for ax, (name, img) in zip(axes, images.items()):
            # Handle different image formats
            if img is None:
                ax.text(0.5, 0.5, "No image", ha='center', va='center')
            elif len(img.shape) == 2:
                # Grayscale
                ax.imshow(img, cmap='gray')
            elif img.shape[2] == 3:
                # BGR to RGB for matplotlib
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif img.shape[2] == 4:
                # BGRA to RGBA
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))

            ax.set_title(name)
            ax.axis('off')

        if title:
            plt.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()

    return _display


@pytest.fixture
def display_with_histogram() -> Callable:
    """
    Display images with their histograms.

    Useful for analyzing preprocessing effects.
    """
    import matplotlib.pyplot as plt

    def _display(
        images: Dict[str, np.ndarray],
        title: str = ""
    ):
        n = len(images)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 6))

        if n == 1:
            axes = axes.reshape(2, 1)

        for i, (name, img) in enumerate(images.items()):
            # Image
            if len(img.shape) == 2:
                axes[0, i].imshow(img, cmap='gray')
            else:
                axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(name)
            axes[0, i].axis('off')

            # Histogram
            if len(img.shape) == 2:
                axes[1, i].hist(img.ravel(), bins=256, range=(0, 256), color='gray')
            else:
                for c, color in enumerate(['b', 'g', 'r']):
                    axes[1, i].hist(img[:, :, c].ravel(), bins=256, range=(0, 256),
                                   color=color, alpha=0.5)
            axes[1, i].set_xlim([0, 256])

        if title:
            plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    return _display


# ============== Sample Image Fixtures ==============

@pytest.fixture
def sample_sudoku_image() -> np.ndarray:
    """
    Load a sample Sudoku image for testing.

    First tries to load from fixtures directory.
    If not found, generates a synthetic image.
    """
    fixtures_dir = Path(__file__).parent / "fixtures" / "sudoku_samples"

    # Try to load real image
    for pattern in ["*.jpg", "*.png", "*.jpeg"]:
        images = list(fixtures_dir.glob(pattern))
        if images:
            return cv2.imread(str(images[0]))

    # Generate synthetic if no real image
    return generate_synthetic_sudoku(size=400)


@pytest.fixture
def synthetic_sudoku() -> Callable:
    """
    Fixture that returns a function to generate synthetic Sudoku grids.

    Usage:
        def test_something(synthetic_sudoku):
            img = synthetic_sudoku(size=500, rotation=15)
    """
    def _generate(
        size: int = 400,
        rotation: float = 0,
        noise_level: float = 0,
        perspective: bool = False
    ) -> np.ndarray:
        return generate_synthetic_sudoku(
            size=size,
            rotation=rotation,
            noise_level=noise_level,
            perspective=perspective
        )

    return _generate


def generate_synthetic_sudoku(
    size: int = 400,
    rotation: float = 0,
    noise_level: float = 0,
    perspective: bool = False,
    fill_digits: bool = True
) -> np.ndarray:
    """
    Generate a synthetic Sudoku grid image for testing.

    Args:
        size: Output image size (square)
        rotation: Rotation angle in degrees
        noise_level: Amount of Gaussian noise (0-1)
        perspective: If True, apply perspective distortion
        fill_digits: If True, add some random digits

    Returns:
        BGR image of synthetic Sudoku
    """
    # Create white background
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    margin = int(size * 0.1)
    grid_size = size - 2 * margin
    cell_size = grid_size // 9

    # Draw grid lines
    for i in range(10):
        # Horizontal lines
        y = margin + i * cell_size
        thickness = 3 if i % 3 == 0 else 1
        cv2.line(img, (margin, y), (margin + grid_size, y), (0, 0, 0), thickness)

        # Vertical lines
        x = margin + i * cell_size
        cv2.line(img, (x, margin), (x, margin + grid_size), (0, 0, 0), thickness)

    # Add some digits
    if fill_digits:
        sample_grid = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = cell_size / 50
        font_thickness = max(1, int(cell_size / 25))

        for row in range(9):
            for col in range(9):
                digit = sample_grid[row][col]
                if digit != 0:
                    x = margin + col * cell_size + cell_size // 3
                    y = margin + row * cell_size + 2 * cell_size // 3
                    cv2.putText(img, str(digit), (x, y), font, font_scale,
                               (0, 0, 0), font_thickness)

    # Apply rotation
    if rotation != 0:
        center = (size // 2, size // 2)
        matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        img = cv2.warpAffine(img, matrix, (size, size),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))

    # Apply perspective distortion
    if perspective:
        pts1 = np.float32([
            [margin, margin],
            [size - margin, margin],
            [size - margin, size - margin],
            [margin, size - margin]
        ])
        # Random perspective shift
        shift = size * 0.05
        pts2 = np.float32([
            [margin + np.random.uniform(-shift, shift),
             margin + np.random.uniform(-shift, shift)],
            [size - margin + np.random.uniform(-shift, shift),
             margin + np.random.uniform(-shift, shift)],
            [size - margin + np.random.uniform(-shift, shift),
             size - margin + np.random.uniform(-shift, shift)],
            [margin + np.random.uniform(-shift, shift),
             size - margin + np.random.uniform(-shift, shift)]
        ])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, matrix, (size, size),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))

    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 50, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


# ============== Test Image Collection Fixtures ==============

@pytest.fixture
def test_image_collection(synthetic_sudoku) -> Dict[str, np.ndarray]:
    """
    Collection of test images with various characteristics.

    Returns dict with:
    - clean: Perfect grid
    - rotated: 15° rotation
    - perspective: Perspective distortion
    - noisy: With Gaussian noise
    """
    return {
        "clean": synthetic_sudoku(size=400),
        "rotated": synthetic_sudoku(size=400, rotation=15),
        "perspective": synthetic_sudoku(size=400, perspective=True),
        "noisy": synthetic_sudoku(size=400, noise_level=0.3),
    }


# ============== Utility Fixtures ==============

@pytest.fixture
def save_test_image() -> Callable:
    """
    Fixture to save test images for inspection.

    Usage:
        def test_something(save_test_image):
            save_test_image(result_image, "my_test_result.png")
    """
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    def _save(image: np.ndarray, filename: str):
        path = output_dir / filename
        cv2.imwrite(str(path), image)
        print(f"Saved: {path}")

    return _save
