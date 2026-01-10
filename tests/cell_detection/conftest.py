"""
Pytest fixtures for cell detection tests.

Provides synthetic test images and sample loading utilities.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def synthetic_grid_image():
    """
    Generate a synthetic 9x9 Sudoku grid image.

    Returns a function that creates grids with configurable parameters.
    """
    def _generate(
        size: int = 450,
        border_thickness: int = 2,
        cell_border_thickness: int = 1,
        background: int = 255,
        foreground: int = 0,
        rotation: float = 0,
        add_noise: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic grid.

        Args:
            size: Image size (square)
            border_thickness: Outer border thickness
            cell_border_thickness: Inner cell border thickness
            background: Background color (0-255)
            foreground: Line color (0-255)
            rotation: Rotation angle in degrees
            add_noise: Whether to add Gaussian noise

        Returns:
            Tuple of (image, corners) where corners is 4x2 array [TL, TR, BR, BL]
        """
        # Create base image
        image = np.full((size, size), background, dtype=np.uint8)

        # Grid parameters
        margin = size // 10
        grid_size = size - 2 * margin
        cell_size = grid_size // 9

        # Draw outer border
        cv2.rectangle(
            image,
            (margin, margin),
            (margin + grid_size, margin + grid_size),
            foreground,
            border_thickness,
        )

        # Draw inner grid lines
        for i in range(1, 9):
            # Horizontal lines
            y = margin + i * cell_size
            thickness = border_thickness if i % 3 == 0 else cell_border_thickness
            cv2.line(image, (margin, y), (margin + grid_size, y), foreground, thickness)

            # Vertical lines
            x = margin + i * cell_size
            cv2.line(image, (x, margin), (x, margin + grid_size), foreground, thickness)

        # Corners before rotation
        corners = np.array([
            [margin, margin],  # TL
            [margin + grid_size, margin],  # TR
            [margin + grid_size, margin + grid_size],  # BR
            [margin, margin + grid_size],  # BL
        ], dtype=np.float32)

        # Apply rotation if specified
        if rotation != 0:
            center = (size // 2, size // 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            image = cv2.warpAffine(image, M, (size, size), borderValue=background)

            # Rotate corners
            ones = np.ones((4, 1))
            corners_h = np.hstack([corners, ones])
            corners = (M @ corners_h.T).T

        # Add noise if specified
        if add_noise:
            noise = np.random.normal(0, 10, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Convert to BGR for consistency
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image_bgr, corners.astype(np.float32)

    return _generate


@pytest.fixture
def sample_sudoku_images():
    """
    Load sample images from Examples/aug directory.

    Returns a function that loads N sample images.
    """
    def _load(
        n: int = 5,
        seed: Optional[int] = None,
        aug_levels: Optional[List[int]] = None,
    ) -> List[Tuple[Path, np.ndarray]]:
        """
        Load sample images.

        Args:
            n: Number of images to load
            seed: Random seed for reproducibility
            aug_levels: Filter by augmentation levels (e.g., [0, 50, 100])

        Returns:
            List of (path, image) tuples
        """
        examples_dir = PROJECT_ROOT / "Examples" / "aug"

        if not examples_dir.exists():
            return []

        # Get all images
        patterns = ["*.jpeg", "*.jpg", "*.png"]
        all_images = []
        for pattern in patterns:
            all_images.extend(examples_dir.glob(pattern))

        # Filter by augmentation level if specified
        if aug_levels is not None:
            filtered = []
            for path in all_images:
                # Filename format: _LEVEL_rest.jpeg
                parts = path.stem.split("_")
                if len(parts) >= 2:
                    try:
                        level = int(parts[1])
                        if level in aug_levels:
                            filtered.append(path)
                    except ValueError:
                        continue
            all_images = filtered

        # Random sample
        if seed is not None:
            np.random.seed(seed)

        if len(all_images) > n:
            indices = np.random.choice(len(all_images), n, replace=False)
            selected = [all_images[i] for i in indices]
        else:
            selected = all_images[:n]

        # Load images
        results = []
        for path in selected:
            img = cv2.imread(str(path))
            if img is not None:
                results.append((path, img))

        return results

    return _load


@pytest.fixture
def display_images(request):
    """
    Display images during test (only when --show-images flag is used).

    Usage in tests:
        def test_something(display_images):
            display_images([img1, img2], ["Title 1", "Title 2"])
    """
    import matplotlib.pyplot as plt

    show = request.config.getoption("--show-images", default=False)

    def _display(images: List[np.ndarray], titles: Optional[List[str]] = None):
        if not show:
            return

        n = len(images)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for i, img in enumerate(images):
            ax = axes[i]
            if len(img.shape) == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if titles and i < len(titles):
                ax.set_title(titles[i])
            ax.axis("off")

        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    return _display


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--show-images",
        action="store_true",
        default=False,
        help="Show images during tests using matplotlib",
    )
