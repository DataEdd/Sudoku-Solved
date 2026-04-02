# Sudoku Corner Detection - Implementation Instructions

This document contains complete instructions to build a sudoku grid corner detector from scratch. The detector identifies the 4 outer corners of a sudoku puzzle in an image.

---

## Project Overview

**Goal:** Given an image containing a sudoku puzzle, detect the 4 corner points of the grid (top-left, top-right, bottom-right, bottom-left).

**Method:** Adaptive thresholding + contour detection with multiple preprocessing configurations and scoring-based selection.

**Success Metric:** >95% IoU (Intersection over Union) between detected and ground truth quadrilaterals.

---

## Step 1: Project Setup

### 1.1 Create Project Directory

```bash
mkdir sudoku-detector
cd sudoku-detector
```

### 1.2 Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1.3 Install Dependencies

Create `requirements.txt`:

```
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.4.0
shapely>=1.8.0
```

Install:

```bash
pip install -r requirements.txt
```

### 1.4 Create Project Structure

```bash
mkdir -p src data
touch src/__init__.py
touch src/preprocessing.py
touch src/geometry.py
touch src/detection.py
```

Final structure:
```
sudoku-detector/
├── venv/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Image preprocessing utilities
│   ├── geometry.py         # Geometric utilities (corner ordering, IoU)
│   └── detection.py        # Main detection pipeline
├── data/
│   └── ground_truth_corners.json  # Optional: for testing
├── requirements.txt
└── detect.py               # CLI entry point
```

---

## Step 2: Implement Preprocessing Module

Create `src/preprocessing.py`:

```python
"""Image preprocessing utilities."""

import cv2
import numpy as np
from typing import Tuple


def resize_image(image: np.ndarray, max_size: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Resize image so longest edge equals max_size.

    Args:
        image: Input image (BGR or grayscale)
        max_size: Target size for longest edge

    Returns:
        resized: Resized image
        scale: Scale factor applied (resized = original * scale)
    """
    h, w = image.shape[:2]

    if max(h, w) <= max_size:
        return image.copy(), 1.0

    if h > w:
        scale = max_size / h
    else:
        scale = max_size / w

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized, scale


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if needed.

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        Grayscale image
    """
    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

---

## Step 3: Implement Geometry Module

Create `src/geometry.py`:

```python
"""Geometric utilities for corner detection."""

import numpy as np
from typing import Tuple

try:
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Order corners as [top-left, top-right, bottom-right, bottom-left].

    Args:
        corners: (4, 2) array of corner points in any order

    Returns:
        (4, 2) array with corners ordered [TL, TR, BR, BL]
    """
    corners = np.array(corners, dtype=np.float32)

    # Sort by sum (x + y): TL has smallest, BR has largest
    s = corners.sum(axis=1)
    tl_idx = np.argmin(s)
    br_idx = np.argmax(s)

    # Sort by difference (y - x): TR has smallest, BL has largest
    d = np.diff(corners, axis=1).flatten()
    tr_idx = np.argmin(d)
    bl_idx = np.argmax(d)

    return np.array([
        corners[tl_idx],
        corners[tr_idx],
        corners[br_idx],
        corners[bl_idx]
    ], dtype=np.float32)


def compute_iou(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """
    Compute Intersection over Union of two quadrilaterals.

    Args:
        corners1: (4, 2) array of first quad corners
        corners2: (4, 2) array of second quad corners

    Returns:
        IoU value between 0.0 and 1.0
    """
    corners1 = np.array(corners1, dtype=np.float64)
    corners2 = np.array(corners2, dtype=np.float64)

    if SHAPELY_AVAILABLE:
        poly1 = Polygon(corners1)
        poly2 = Polygon(corners2)

        if not poly1.is_valid or not poly2.is_valid:
            return 0.0

        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area

        return intersection_area / union_area if union_area > 0 else 0.0

    # Fallback: rasterize and compute IoU on binary masks
    all_points = np.vstack([corners1, corners2])
    min_xy = np.floor(all_points.min(axis=0)).astype(int)
    max_xy = np.ceil(all_points.max(axis=0)).astype(int)

    # Shift to origin
    c1_shifted = (corners1 - min_xy).astype(np.int32)
    c2_shifted = (corners2 - min_xy).astype(np.int32)

    h = max_xy[1] - min_xy[1] + 1
    w = max_xy[0] - min_xy[0] + 1

    import cv2
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(mask1, [c1_shifted], 1)
    cv2.fillPoly(mask2, [c2_shifted], 1)

    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)

    return intersection / union if union > 0 else 0.0
```

---

## Step 4: Implement Detection Module

Create `src/detection.py`:

```python
"""
Sudoku grid corner detection using adaptive thresholding and contour detection.

Algorithm:
1. Resize image to max 1000px, convert to grayscale
2. Try multiple preprocessing configurations (blur, threshold, dilation, CLAHE)
3. For each config: binarize → find contours → approximate to quadrilateral
4. Score each result by geometric properties (area, aspect ratio, angles)
5. Return best scoring result with corners ordered [TL, TR, BR, BL]
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List

from .preprocessing import resize_image, to_grayscale
from .geometry import order_corners


# Processing target size
TARGET_SIZE = 1000

# Detection configurations: (blur_size, block_size, c_value, dilation, use_clahe)
# These were empirically optimized for various image conditions
DETECTION_CONFIGS = [
    # Standard configs - work well for most well-lit images
    (7, 7, 5, 0, False),
    (7, 11, 5, 0, False),
    (5, 11, 2, 0, False),
    (5, 11, 2, 1, False),

    # CLAHE configs - for low contrast / challenging lighting
    (5, 5, 10, 1, True),
    (7, 5, 8, 2, True),
    (5, 5, 10, 2, True),

    # Small block configs - for tilted / small grids
    (3, 7, 2, 0, False),
    (3, 7, 1, 0, False),
]


def _find_quadrilateral(
    binary: np.ndarray,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.95
) -> Optional[np.ndarray]:
    """
    Find the largest valid quadrilateral in a binary image.

    Args:
        binary: Binary image (white = foreground)
        min_area_ratio: Minimum area as ratio of image area
        max_area_ratio: Maximum area as ratio of image area

    Returns:
        (4, 2) array of corner points, or None if not found
    """
    h, w = binary.shape
    img_area = h * w
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(contour, True)

        # Try increasing epsilon values until we get 4 points
        for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
            approx = cv2.approxPolyDP(contour, eps * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
                quad_area = cv2.contourArea(approx)

                if quad_area > best_area:
                    best_area = quad_area
                    best_quad = quad
                break

    return best_quad


def _try_adaptive_detection(
    gray: np.ndarray,
    blur_size: int,
    block_size: int,
    c_val: int,
    dilation: int,
    use_clahe: bool = False
) -> Optional[np.ndarray]:
    """
    Try adaptive thresholding with specific parameters.

    Args:
        gray: Grayscale image
        blur_size: Gaussian blur kernel size (must be odd)
        block_size: Adaptive threshold block size (must be odd)
        c_val: Constant subtracted from mean in adaptive threshold
        dilation: Number of dilation iterations (0 = none)
        use_clahe: Whether to apply CLAHE preprocessing

    Returns:
        (4, 2) array of corner points, or None if not found
    """
    processed = gray.copy()

    # Optional CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)

    # Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(processed, (blur_size, blur_size), 0)

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, c_val
    )

    # Optional dilation to connect broken lines
    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=dilation)

    return _find_quadrilateral(binary)


def _score_quadrilateral(
    quad: np.ndarray,
    img_shape: Tuple[int, int],
    method: str = ""
) -> float:
    """
    Score a quadrilateral based on geometric properties.

    Higher score = better candidate for sudoku grid.

    Args:
        quad: (4, 2) array of corner points
        img_shape: (height, width) of image
        method: Detection method name (for bonus scoring)

    Returns:
        Score between 0 and 1
    """
    h, w = img_shape
    img_area = h * w

    # Compute quadrilateral area
    quad_area = cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.int32))
    area_ratio = quad_area / img_area

    # Area score: moderate preference for larger, don't over-penalize smaller
    if area_ratio < 0.05:
        area_score = 0.2
    elif area_ratio > 0.9:
        area_score = 0.5
    elif area_ratio < 0.15:
        area_score = 0.6 + (area_ratio - 0.05) * 2
    else:
        area_score = 0.8 + min(0.2, (area_ratio - 0.15) * 0.5)

    # Order corners for consistent measurements
    ordered = order_corners(quad)

    # Compute widths and heights
    widths = [
        np.linalg.norm(ordered[1] - ordered[0]),  # Top edge
        np.linalg.norm(ordered[2] - ordered[3])   # Bottom edge
    ]
    heights = [
        np.linalg.norm(ordered[3] - ordered[0]),  # Left edge
        np.linalg.norm(ordered[2] - ordered[1])   # Right edge
    ]

    avg_width = np.mean(widths)
    avg_height = np.mean(heights)

    # Aspect ratio score: prefer square-ish shapes
    if avg_height > 0 and avg_width > 0:
        aspect = min(avg_width, avg_height) / max(avg_width, avg_height)
        aspect_score = aspect
    else:
        aspect_score = 0.1

    # Angle score: prefer near-90-degree corners
    def angle_at_corner(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    angles = []
    for i in range(4):
        p1 = ordered[(i - 1) % 4]
        p2 = ordered[i]
        p3 = ordered[(i + 1) % 4]
        angles.append(angle_at_corner(p1, p2, p3))

    angle_deviations = [abs(90 - a) for a in angles]
    avg_deviation = np.mean(angle_deviations)
    angle_score = max(0.1, 1.0 - avg_deviation / 45)

    # Combined score
    score = area_score * 0.35 + aspect_score * 0.35 + angle_score * 0.30

    # Small bonus for adaptive methods (empirically more accurate)
    if "adaptive" in method:
        score += 0.02

    return score


def detect_corners(image: np.ndarray) -> Tuple[Optional[np.ndarray], dict]:
    """
    Detect sudoku grid corners in an image.

    Args:
        image: Input image (BGR format from cv2.imread)

    Returns:
        corners: (4, 2) array of corners [TL, TR, BR, BL] in original
                 image coordinates, or None if detection failed
        debug: Dictionary with debug information:
               - 'method': Detection method that succeeded
               - 'score': Score of selected result
               - 'scale': Scale factor applied during processing
               - 'num_candidates': Number of valid candidates found
               - 'error': Error message if detection failed
    """
    debug = {}

    # Preprocess
    resized, scale = resize_image(image, TARGET_SIZE)
    gray = to_grayscale(resized)
    debug['scale'] = scale

    # Try all configurations and collect candidates
    candidates = []

    for blur_size, block_size, c_val, dilation, use_clahe in DETECTION_CONFIGS:
        quad = _try_adaptive_detection(
            gray, blur_size, block_size, c_val, dilation, use_clahe
        )

        if quad is not None:
            clahe_str = "_clahe" if use_clahe else ""
            method = f"adaptive_b{blur_size}_block{block_size}_c{c_val}_d{dilation}{clahe_str}"
            score = _score_quadrilateral(quad, gray.shape, method)
            candidates.append((quad, score, method))

    # Also try morphological gradient as fallback
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for dilation in [1, 2]:
        dilated = cv2.dilate(binary, kernel, iterations=dilation)
        quad = _find_quadrilateral(dilated)

        if quad is not None:
            method = f"morph_d{dilation}"
            score = _score_quadrilateral(quad, gray.shape, method)
            candidates.append((quad, score, method))

    debug['num_candidates'] = len(candidates)

    if not candidates:
        debug['error'] = 'No quadrilateral found'
        return None, debug

    # Select best candidate by score
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_quad, best_score, best_method = candidates[0]

    debug['method'] = best_method
    debug['score'] = best_score

    # Order corners and scale back to original image coordinates
    corners = order_corners(best_quad)
    corners_original = corners / scale

    return corners_original, debug
```

---

## Step 5: Create CLI Entry Point

Create `detect.py` in project root:

```python
#!/usr/bin/env python3
"""
Command-line interface for sudoku corner detection.

Usage:
    python detect.py <image_path>
    python detect.py <image_path> --visualize
    python detect.py <image_path> --output corners.json
"""

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.detection import detect_corners


def main():
    parser = argparse.ArgumentParser(description='Detect sudoku grid corners in an image')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--visualize', '-v', action='store_true', help='Show visualization')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file for corners')

    args = parser.parse_args()

    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        sys.exit(1)

    # Detect corners
    corners, debug = detect_corners(image)

    if corners is None:
        print(f"Detection failed: {debug.get('error', 'Unknown error')}")
        sys.exit(1)

    # Print results
    print(f"Detection successful!")
    print(f"  Method: {debug['method']}")
    print(f"  Score: {debug['score']:.4f}")
    print(f"  Corners (TL, TR, BR, BL):")
    labels = ['TL', 'TR', 'BR', 'BL']
    for label, corner in zip(labels, corners):
        print(f"    {label}: ({corner[0]:.1f}, {corner[1]:.1f})")

    # Save to JSON if requested
    if args.output:
        output_data = {
            'image': str(image_path),
            'corners': corners.tolist(),
            'method': debug['method'],
            'score': debug['score']
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"  Saved to: {args.output}")

    # Visualize if requested
    if args.visualize:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)

        # Draw quadrilateral
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=2)

        # Draw corner points with labels
        ax.scatter(corners[:, 0], corners[:, 1], c='green', s=100, zorder=5)
        for label, corner in zip(labels, corners):
            ax.annotate(label, corner, fontsize=12, color='white',
                       bbox=dict(boxstyle='round', facecolor='green', alpha=0.8),
                       xytext=(10, 10), textcoords='offset points')

        ax.set_title(f"{image_path.name}\nMethod: {debug['method']}, Score: {debug['score']:.4f}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
```

---

## Step 6: Test the Implementation

### 6.1 Basic Test

```bash
python detect.py path/to/sudoku_image.jpg --visualize
```

### 6.2 Create Test Script

Create `test_detection.py`:

```python
#!/usr/bin/env python3
"""Test detection on sample images."""

import json
from pathlib import Path

import cv2
import numpy as np

from src.detection import detect_corners
from src.geometry import compute_iou


def test_with_ground_truth(images_dir: Path, ground_truth_file: Path):
    """Test detection against ground truth."""

    # Load ground truth
    with open(ground_truth_file) as f:
        ground_truth = json.load(f)

    results = []

    for name, gt_corners in ground_truth.items():
        image_path = images_dir / name
        if not image_path.exists():
            continue

        image = cv2.imread(str(image_path))
        gt_corners = np.array(gt_corners, dtype=np.float32)

        corners, debug = detect_corners(image)

        if corners is not None:
            iou = compute_iou(corners, gt_corners)
            status = 'PASS' if iou >= 0.95 else 'FAIL'
        else:
            iou = 0.0
            status = 'FAIL'

        results.append({
            'name': name,
            'iou': iou,
            'status': status,
            'method': debug.get('method')
        })

        print(f"{name}: IoU={iou:.4f} [{status}] via {debug.get('method', 'N/A')}")

    # Summary
    passed = sum(1 for r in results if r['status'] == 'PASS')
    total = len(results)
    avg_iou = np.mean([r['iou'] for r in results])

    print(f"\nPassed: {passed}/{total} ({100*passed/total:.0f}%)")
    print(f"Average IoU: {avg_iou:.4f}")


if __name__ == '__main__':
    # Update these paths as needed
    images_dir = Path('examples')
    ground_truth_file = Path('data/ground_truth_corners.json')

    if ground_truth_file.exists():
        test_with_ground_truth(images_dir, ground_truth_file)
    else:
        print("No ground truth file found. Run detection on individual images:")
        print("  python detect.py <image_path> --visualize")
```

---

## Algorithm Summary

### Why This Approach Works

1. **Adaptive thresholding** handles uneven lighting (shadows, glare, different exposures)

2. **Multiple configurations** handle variety in images:
   - Standard configs for well-lit images
   - CLAHE configs for low contrast
   - Small block configs for tilted/small grids

3. **Polygon approximation with variable epsilon** handles both smooth and jagged contours

4. **Scoring function** selects geometrically valid sudoku grids:
   - Area: 35% weight (reasonable size, not too small/large)
   - Aspect ratio: 35% weight (square-ish shape)
   - Angles: 30% weight (corners close to 90°)

### Key Parameters

| Parameter | Purpose | Values Used |
|-----------|---------|-------------|
| blur_size | Noise reduction | 3, 5, 7 |
| block_size | Local neighborhood for threshold | 5, 7, 11 |
| c_value | Threshold sensitivity | 1, 2, 5, 8, 10 |
| dilation | Connect broken lines | 0, 1, 2 |
| CLAHE | Contrast enhancement | On/Off |
| epsilon | Polygon simplification | 0.01 - 0.06 |

### Corner Ordering

Corners are ordered using sum and difference of coordinates:
- **Top-Left**: Smallest (x + y)
- **Bottom-Right**: Largest (x + y)
- **Top-Right**: Smallest (y - x)
- **Bottom-Left**: Largest (y - x)

---

## Ground Truth Format

If creating ground truth for testing, use this JSON format:

```json
{
  "image1.jpg": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "image2.jpg": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
}
```

Where corners are ordered: [TL, TR, BR, BL]

---

## Troubleshooting

### Detection Fails

1. **Image too dark/bright**: CLAHE configs should help
2. **Grid too small**: Try lowering `min_area_ratio` in `_find_quadrilateral`
3. **Grid very tilted**: Small block configs should help
4. **Broken grid lines**: Increase dilation

### Low IoU

1. **Check corner ordering**: Ensure ground truth uses [TL, TR, BR, BL] order
2. **Check image orientation**: Detector assumes standard orientation
3. **Partial grid visible**: Detector expects full grid boundary visible

### Performance

- Processing is ~50-200ms per image depending on size
- Resize to smaller `TARGET_SIZE` for faster processing (may reduce accuracy)
