"""
Generalized Hough Transform (GHT) for Sudoku grid detection.

Mathematical foundation:
- GHT can detect arbitrary shapes, not just lines
- Uses an R-table (reference table) that encodes shape structure
- For each edge pixel in template, store vector to reference point (center)

Algorithm:
1. Build R-table from template:
   - For each edge pixel, compute gradient direction
   - Store vector from edge to center, indexed by gradient direction

2. Detection:
   - For each edge pixel in image, compute gradient direction
   - Look up R-table for that direction
   - Vote in accumulator for possible centers
   - Peak in accumulator = detected shape center

3. Scale and rotation handling:
   - Can search over multiple scales
   - Can search over rotations (rotate R-table entries)
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class GHTResult:
    """Result of Generalized Hough Transform detection."""

    # Detection results
    center: Tuple[float, float]        # Detected grid center (x, y)
    scale: float                        # Detected scale relative to template
    rotation: float                     # Detected rotation in degrees
    corners: np.ndarray                 # 4 corner points [[x,y], ...]

    # Confidence score
    confidence: float

    # Visualization data
    accumulator: Optional[np.ndarray] = None
    annotated_image: Optional[np.ndarray] = None


class GridTemplate:
    """
    Sudoku grid template for Generalized Hough Transform.

    Creates a template of a Sudoku grid and builds the R-table
    for shape matching.
    """

    def __init__(self, size: int = 200, line_thickness: int = 2):
        """
        Initialize grid template.

        Args:
            size: Template size in pixels (square)
            line_thickness: Thickness of grid lines
        """
        self.size = size
        self.line_thickness = line_thickness
        self.center = (size // 2, size // 2)

        # Create template and extract features
        self.image = self._create_grid_template()
        self.edges = self._detect_edges()
        self.r_table = self._build_r_table()

    def _create_grid_template(self) -> np.ndarray:
        """
        Create a template image of a Sudoku grid.

        The grid has:
        - Outer border
        - 9x9 internal divisions (10 lines each direction)
        """
        template = np.zeros((self.size, self.size), dtype=np.uint8)

        cell_size = self.size / 9

        # Draw horizontal lines (10 lines: 0 to 9)
        for i in range(10):
            y = int(i * cell_size)
            # Thicker lines at box boundaries
            thickness = self.line_thickness * 2 if i % 3 == 0 else self.line_thickness
            cv2.line(template, (0, y), (self.size - 1, y), 255, thickness)

        # Draw vertical lines (10 lines: 0 to 9)
        for j in range(10):
            x = int(j * cell_size)
            thickness = self.line_thickness * 2 if j % 3 == 0 else self.line_thickness
            cv2.line(template, (x, 0), (x, self.size - 1), 255, thickness)

        return template

    def _detect_edges(self) -> np.ndarray:
        """Detect edges in template using Canny."""
        return cv2.Canny(self.image, 50, 150)

    def _build_r_table(self, n_bins: int = 360) -> Dict[int, List[Tuple[float, float]]]:
        """
        Build R-table for the template.

        The R-table maps gradient direction (quantized) to a list of
        vectors from edge pixels to the reference point (center).

        For each edge pixel:
        1. Compute gradient direction θ at that pixel
        2. Compute vector r = (center - pixel)
        3. Store r in R-table[θ]

        Args:
            n_bins: Number of angle bins (360 = 1° resolution)

        Returns:
            R-table: Dict mapping angle bin to list of (dx, dy) vectors
        """
        # Compute gradients
        gx = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=3)

        r_table = defaultdict(list)

        # For each edge pixel
        edge_pixels = np.where(self.edges > 0)

        for y, x in zip(*edge_pixels):
            # Compute gradient direction
            angle = np.arctan2(gy[y, x], gx[y, x])

            # Quantize to bin
            angle_bin = int((angle + np.pi) / (2 * np.pi) * n_bins) % n_bins

            # Vector from edge to center
            dx = self.center[0] - x
            dy = self.center[1] - y

            r_table[angle_bin].append((dx, dy))

        return dict(r_table)

    def get_r_table_stats(self) -> Dict:
        """Get statistics about the R-table for visualization."""
        n_entries = sum(len(v) for v in self.r_table.values())
        n_bins_used = len(self.r_table)
        max_per_bin = max(len(v) for v in self.r_table.values()) if self.r_table else 0

        return {
            "total_entries": n_entries,
            "bins_used": n_bins_used,
            "max_per_bin": max_per_bin,
            "avg_per_bin": n_entries / n_bins_used if n_bins_used > 0 else 0
        }


def detect_grid_ght(
    image: np.ndarray,
    template: Optional[GridTemplate] = None,
    scales: List[float] = None,
    return_accumulator: bool = True
) -> GHTResult:
    """
    Detect Sudoku grid using Generalized Hough Transform.

    Algorithm:
    1. Preprocess image to get edges
    2. Compute gradient direction at each edge pixel
    3. For each scale:
       - Create accumulator
       - For each edge pixel, look up R-table and vote
    4. Find peak in accumulator = grid center
    5. Extract corners based on center and scale

    Args:
        image: BGR input image
        template: Pre-built GridTemplate (will create if None)
        scales: List of scales to try (default: [0.5, 0.75, 1.0, 1.25, 1.5])
        return_accumulator: If True, include accumulator in result

    Returns:
        GHTResult with detection data
    """
    from app.core.preprocessing import preprocess_for_ght

    if template is None:
        template = GridTemplate(size=200)

    if scales is None:
        # Try different scales based on image size
        min_dim = min(image.shape[:2])
        base_scale = min_dim / template.size
        scales = [base_scale * s for s in [0.5, 0.75, 1.0, 1.25, 1.5]]

    # Preprocess
    edges = preprocess_for_ght(image)

    # Compute gradients for direction
    gx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)

    h, w = image.shape[:2]

    best_result = None
    best_score = 0
    best_accumulator = None

    # Try each scale
    for scale in scales:
        # Create accumulator
        accumulator = np.zeros((h, w), dtype=np.float64)

        # Get edge pixels
        edge_pixels = np.where(edges > 0)

        # Vote
        for y, x in zip(*edge_pixels):
            # Compute gradient direction
            angle = np.arctan2(gy[y, x], gx[y, x])
            angle_bin = int((angle + np.pi) / (2 * np.pi) * 360) % 360

            # Look up R-table
            if angle_bin in template.r_table:
                for dx, dy in template.r_table[angle_bin]:
                    # Scale the displacement
                    cx = int(x + dx * scale)
                    cy = int(y + dy * scale)

                    # Vote if within bounds
                    if 0 <= cx < w and 0 <= cy < h:
                        accumulator[cy, cx] += 1

        # Apply Gaussian blur to smooth accumulator
        accumulator = cv2.GaussianBlur(accumulator, (15, 15), 0)

        # Find peak
        max_val = accumulator.max()
        if max_val > best_score:
            best_score = max_val
            max_loc = np.unravel_index(accumulator.argmax(), accumulator.shape)
            center = (max_loc[1], max_loc[0])  # (x, y)

            # Estimate grid size at this scale
            grid_size = template.size * scale

            # Compute corners
            half_size = grid_size / 2
            corners = np.array([
                [center[0] - half_size, center[1] - half_size],  # top-left
                [center[0] + half_size, center[1] - half_size],  # top-right
                [center[0] + half_size, center[1] + half_size],  # bottom-right
                [center[0] - half_size, center[1] + half_size],  # bottom-left
            ])

            best_result = {
                "center": center,
                "scale": scale,
                "corners": corners
            }
            best_accumulator = accumulator.copy()

    if best_result is None:
        # No detection
        return GHTResult(
            center=(0, 0),
            scale=1.0,
            rotation=0.0,
            corners=np.array([]),
            confidence=0.0
        )

    # Normalize confidence
    confidence = min(1.0, best_score / 1000)  # Normalize based on expected votes

    # Create visualization
    annotated = None
    if return_accumulator:
        annotated = draw_ght_result(image, best_result["corners"])

        # Normalize accumulator for visualization
        if best_accumulator is not None:
            best_accumulator = (best_accumulator / best_accumulator.max() * 255).astype(np.uint8)
            best_accumulator = cv2.applyColorMap(best_accumulator, cv2.COLORMAP_JET)

    return GHTResult(
        center=best_result["center"],
        scale=best_result["scale"],
        rotation=0.0,  # TODO: Implement rotation detection
        corners=best_result["corners"],
        confidence=confidence,
        accumulator=best_accumulator,
        annotated_image=annotated
    )


def draw_ght_result(
    image: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3
) -> np.ndarray:
    """
    Draw detected grid boundary on image.

    Args:
        image: BGR image
        corners: 4 corner points [[x,y], ...]
        color: Line color
        thickness: Line thickness

    Returns:
        Annotated image
    """
    result = image.copy()

    if len(corners) == 4:
        pts = corners.astype(np.int32)

        # Draw quadrilateral
        for i in range(4):
            pt1 = tuple(pts[i])
            pt2 = tuple(pts[(i + 1) % 4])
            cv2.line(result, pt1, pt2, color, thickness)

        # Draw center
        center = np.mean(corners, axis=0).astype(int)
        cv2.circle(result, tuple(center), 5, (0, 0, 255), -1)

    return result


def visualize_r_table(
    template: GridTemplate,
    n_samples: int = 50
) -> np.ndarray:
    """
    Create visualization of R-table as vectors on template.

    Shows a sample of R-table entries as arrows from edge
    pixels to the center.

    Args:
        template: GridTemplate with R-table
        n_samples: Number of vectors to draw

    Returns:
        Visualization image
    """
    # Create color image from template
    vis = cv2.cvtColor(template.image, cv2.COLOR_GRAY2BGR)

    # Sample entries from R-table
    all_entries = []
    for angle_bin, vectors in template.r_table.items():
        # Find a representative edge pixel for this angle
        angle = (angle_bin / 360 * 2 * np.pi) - np.pi

        for dx, dy in vectors:
            # Compute edge pixel position
            ex = template.center[0] - dx
            ey = template.center[1] - dy

            if 0 <= ex < template.size and 0 <= ey < template.size:
                all_entries.append((int(ex), int(ey), int(dx), int(dy)))

    # Sample and draw
    if len(all_entries) > n_samples:
        indices = np.random.choice(len(all_entries), n_samples, replace=False)
        entries = [all_entries[i] for i in indices]
    else:
        entries = all_entries

    for ex, ey, dx, dy in entries:
        # Draw arrow from edge to center
        cv2.arrowedLine(
            vis,
            (ex, ey),
            (ex + dx // 2, ey + dy // 2),
            (0, 255, 0),
            1,
            tipLength=0.3
        )

    # Mark center
    cv2.circle(vis, template.center, 5, (0, 0, 255), -1)

    return vis


def visualize_accumulator(accumulator: np.ndarray) -> np.ndarray:
    """
    Create heatmap visualization of accumulator.

    Args:
        accumulator: 2D accumulator array

    Returns:
        Colored heatmap image
    """
    # Normalize to 0-255
    normalized = (accumulator / accumulator.max() * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    return heatmap
