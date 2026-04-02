# Sudoku Detection Utilities
# All algorithms implemented from scratch using numpy

from .convolution import convolve2d, gaussian_kernel, gaussian_blur
from .edges import sobel_gradients, edge_magnitude, edge_direction
from .threshold import adaptive_threshold, otsu_threshold
from .contours import find_contours, approximate_polygon
from .geometry import order_corners, compute_quad_area, point_distance
from .homography import compute_homography, apply_homography, warp_perspective, bilinear_interpolate

__all__ = [
    # Convolution
    'convolve2d', 'gaussian_kernel', 'gaussian_blur',
    # Edges
    'sobel_gradients', 'edge_magnitude', 'edge_direction',
    # Threshold
    'adaptive_threshold', 'otsu_threshold',
    # Contours
    'find_contours', 'approximate_polygon',
    # Geometry
    'order_corners', 'compute_quad_area', 'point_distance',
    # Homography
    'compute_homography', 'apply_homography', 'warp_perspective', 'bilinear_interpolate',
]
