"""
Image sampling utilities for border detection tests.

Provides functions to sample images from the Examples directory.
"""

import random
import re
from pathlib import Path
from typing import List, Optional, Set

import cv2
import numpy as np


def get_examples_dir() -> Path:
    """Get path to Examples directory."""
    # Try relative to this file
    module_dir = Path(__file__).parent.parent.parent
    examples_dir = module_dir / "Examples"

    if examples_dir.exists():
        return examples_dir

    raise FileNotFoundError(f"Examples directory not found at {examples_dir}")


def get_aug_dir() -> Path:
    """Get path to Examples/aug directory."""
    return get_examples_dir() / "aug"


def list_aug_images(
    aug_levels: Optional[List[int]] = None
) -> List[Path]:
    """
    List all augmented images.

    Args:
        aug_levels: Filter by specific augmentation levels (e.g., [0, 100])

    Returns:
        List of image paths
    """
    aug_dir = get_aug_dir()

    if not aug_dir.exists():
        return []

    all_images = list(aug_dir.glob("*.jpeg")) + list(aug_dir.glob("*.jpg"))

    if aug_levels is None:
        return sorted(all_images)

    # Filter by augmentation level
    # Filename format: _[LEVEL]_[ID].jpeg
    filtered = []
    level_set = set(aug_levels)

    for img_path in all_images:
        match = re.match(r"_(\d+)_", img_path.name)
        if match:
            level = int(match.group(1))
            if level in level_set:
                filtered.append(img_path)

    return sorted(filtered)


def list_unsolved_images() -> List[Path]:
    """List images in Examples/unsolved directory."""
    unsolved_dir = get_examples_dir() / "unsolved"

    if not unsolved_dir.exists():
        return []

    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        images.extend(unsolved_dir.glob(ext))

    return sorted(images)


def get_aug_levels() -> Set[int]:
    """Get set of all augmentation levels in the dataset."""
    aug_dir = get_aug_dir()

    if not aug_dir.exists():
        return set()

    levels = set()
    for img_path in aug_dir.glob("*.jpeg"):
        match = re.match(r"_(\d+)_", img_path.name)
        if match:
            levels.add(int(match.group(1)))

    return levels


def sample_images(
    n: int,
    aug_levels: Optional[List[int]] = None,
    include_unsolved: bool = False,
    seed: Optional[int] = None
) -> List[Path]:
    """
    Randomly sample images.

    Args:
        n: Number of images to sample
        aug_levels: Filter by augmentation levels (None = all)
        include_unsolved: Include images from unsolved directory
        seed: Random seed for reproducibility

    Returns:
        List of sampled image paths
    """
    if seed is not None:
        random.seed(seed)

    # Collect candidate images
    candidates = list_aug_images(aug_levels)

    if include_unsolved:
        candidates.extend(list_unsolved_images())

    if not candidates:
        return []

    # Sample
    n = min(n, len(candidates))
    return random.sample(candidates, n)


def load_image(path: Path) -> Optional[np.ndarray]:
    """
    Load image from path.

    Args:
        path: Path to image file

    Returns:
        BGR image array or None if load failed
    """
    image = cv2.imread(str(path))
    return image


def get_image_info(path: Path) -> dict:
    """
    Get metadata about an image.

    Args:
        path: Path to image file

    Returns:
        Dict with name, aug_level, size, etc.
    """
    info = {
        "path": str(path),
        "name": path.name,
        "aug_level": None,
        "source": "unknown",
    }

    # Parse augmentation level from filename
    match = re.match(r"_(\d+)_", path.name)
    if match:
        info["aug_level"] = int(match.group(1))
        info["source"] = "aug"
    elif "unsolved" in str(path):
        info["source"] = "unsolved"

    # Get file size
    if path.exists():
        info["file_size_kb"] = path.stat().st_size / 1024

    return info
