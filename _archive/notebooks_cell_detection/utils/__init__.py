"""
Utility modules for cell detection notebooks.

This package provides helper functions for:
- visualization: Plotting and debugging visual output
- geometry: Coordinate transforms and shape calculations
"""

from .visualization import (
    show_image,
    show_images_grid,
    draw_corners,
    draw_quadrilateral,
    draw_cell_highlight,
    create_debug_montage,
)

from .geometry import (
    order_corners,
    compute_quad_area,
    is_valid_quadrilateral,
    line_angle,
    line_intersection,
    point_distance,
)

__all__ = [
    # Visualization
    "show_image",
    "show_images_grid",
    "draw_corners",
    "draw_quadrilateral",
    "draw_cell_highlight",
    "create_debug_montage",
    # Geometry
    "order_corners",
    "compute_quad_area",
    "is_valid_quadrilateral",
    "line_angle",
    "line_intersection",
    "point_distance",
]
