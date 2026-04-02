"""SIFT (Scale-Invariant Feature Transform) implementation.

This package provides a pure NumPy implementation of the SIFT algorithm
for educational and experimental purposes.

Modules:
    scale_space: Gaussian scale-space pyramid and DoG computation
"""

from .scale_space import (
    gaussian_kernel_1d,
    gaussian_blur,
    build_gaussian_pyramid,
    build_dog_pyramid,
)

__all__ = [
    "gaussian_kernel_1d",
    "gaussian_blur",
    "build_gaussian_pyramid",
    "build_dog_pyramid",
]
