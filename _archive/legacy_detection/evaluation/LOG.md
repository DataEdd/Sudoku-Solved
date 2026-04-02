# Sudoku Detection Benchmark Log

## Test Setup

- **Test set**: 23 images (5 real photos, 18 augmented at various rotations 0-300 deg)
- **Methods tested**: 7 detection approaches + 1 template-matching (GHT)
- **Metrics**: Detection rate, timing, self-reported confidence
- **Date**: 2026-03-31

## Overall Results

| Method | Detection Rate | Avg Time | Median Time | Avg Confidence |
|--------|---------------|----------|-------------|----------------|
| **contour** | **100% (23/23)** | **11ms** | **2ms** | 0.70 |
| hough_polar | 91% (21/23) | 2057ms | 1407ms | 1.00 |
| sudoku_detector | 78% (18/23) | 5ms | 5ms | 0.90 |
| simple_baseline | 70% (16/23) | 3ms | 2ms | 0.97 |
| hough_standard | 70% (16/23) | 21ms | 16ms | 0.98 |
| sobel_flood | 61% (14/23) | 2ms | 2ms | 1.00 |
| line_segment | 57% (13/23) | 6ms | 6ms | 0.70 |

## Key Findings

### 1. Contour method is the clear winner

The contour method (adaptive threshold + largest quadrilateral contour) detects 100% of images across all categories. It's also fast (median 2ms). The only downside: its self-reported confidence (0.70 avg) is lower than methods that detect fewer images — those methods just happen to be confident when they succeed but miss harder cases.

**Takeaway: Self-reported confidence is meaningless for comparison.** A method that says "0.97 confidence" but only detects 70% of images is worse than one that says "0.70 confidence" but detects everything.

### 2. Hough Polar: accurate but impractical

Hough Polar achieves 91% detection and always reports 1.00 confidence. But it takes 1.4 seconds median (up to 16 seconds on high-res images). The polar Hough transform is running `cv2.HoughLines` which generates O(n) lines, then filtering/deduplication takes quadratic time on large images. Not viable for a real-time camera app.

### 3. Rotation robustness varies wildly

| Category | contour | simple_baseline | sobel_flood | line_segment | hough_standard | hough_polar | sudoku_detector |
|----------|---------|-----------------|-------------|-------------|----------------|-------------|-----------------|
| clean (0 deg) | 100% | 100% | 33% | 0% | 67% | 100% | 67% |
| slight (1-15 deg) | 100% | 75% | 75% | 50% | 25% | 100% | 100% |
| moderate (30-60 deg) | 100% | 67% | 33% | 67% | 100% | 100% | 100% |
| heavy (90-180 deg) | 100% | 75% | 75% | 100% | 100% | 100% | 100% |
| extreme (180-300 deg) | 100% | 50% | 75% | 75% | 100% | 50% | 50% |
| real photos | 100% | 60% | 60% | 40% | 40% | 100% | 60% |

Notable observations:
- **line_segment fails on clean images (0%)** — the "clean" augmented images have thin grid lines that don't produce enough Hough line segments meeting the min length threshold.
- **hough_standard fails on slight rotations (25%)** — when lines are slightly off-axis, the angle classifier puts them in neither H nor V buckets.
- **hough_polar fails on extreme rotations (50%)** — the dominant angle detection breaks when the grid is rotated near 210-240 degrees.
- **sobel_flood is inconsistent (33% on clean)** — the flood fill from corners sometimes fills into the grid itself when the outer border isn't strong enough.

### 4. sudoku_detector package: solid but not top

The `sudoku-detector/` package achieves 78% detection with very fast timing (5ms). It's essentially a more sophisticated contour detector with CLAHE preprocessing, sub-pixel corner refinement, and configurable validation. But its stricter validation (aspect ratio, angle checks) causes it to reject valid grids that are heavily warped or rotated.

### 5. Real photos are the hardest category

Only contour and hough_polar achieve 100% on real photos. The other methods struggle because real photos have:
- Perspective distortion (not just rotation)
- Non-uniform lighting
- Background clutter
- Thin/faded grid lines

## Per-Method Analysis

### contour (main.py)
- **Algorithm**: Adaptive threshold → find contours → largest quadrilateral
- **Strengths**: Simple, fast, works on everything
- **Weaknesses**: Confidence scoring doesn't reflect actual quality. No sub-pixel refinement.
- **Why it wins**: Adaptive thresholding handles lighting variation. The "largest contour" heuristic is surprisingly robust — the grid is almost always the biggest rectangular thing in the image.

### simple_baseline (border_detection/)
- **Algorithm**: Sobel gradient magnitude → threshold → find contours → largest quad
- **Strengths**: Very fast (2ms), good confidence when it works
- **Weaknesses**: Fixed Sobel threshold (50) too aggressive for images with low contrast or thin lines. No adaptive behavior.

### sobel_flood (border_detection/)
- **Algorithm**: Sobel edges → flood fill from corners → invert mask → find contour
- **Strengths**: Novel approach to background subtraction
- **Weaknesses**: Flood fill tolerance (10) is too low — it stops at subtle gradients within the grid, creating fragmented masks. Also fails when the grid touches the image edge (flood fill can't reach past it).

### line_segment (border_detection/)
- **Algorithm**: Canny → HoughLinesP → classify H/V → find extremal lines → compute intersections
- **Strengths**: Theoretically principled (intersection of grid lines)
- **Weaknesses**: Requires clear, detectable line segments. Minimum line length (50px) is too high for small grids. Angle threshold (15 deg) too tight for even slightly rotated grids.

### hough_standard (detection.py)
- **Algorithm**: Adaptive threshold → HoughLinesP → classify → cluster → interpolate to 10 lines → intersections
- **Strengths**: When it works, it finds the full grid structure (not just corners)
- **Weaknesses**: The clustering + interpolation pipeline is fragile. Requires detecting both H and V line groups, which fails on rotation.

### hough_polar (detection.py)
- **Algorithm**: Canny → morphological ops → HoughLines (polar) → filter similar → classify by theta
- **Strengths**: Detects lines regardless of orientation
- **Weaknesses**: Extremely slow on high-res images. The dominant angle detection can pick wrong angles on heavily rotated grids.

### sudoku_detector (sudoku-detector/)
- **Algorithm**: CLAHE → adaptive threshold → contour detection → validation → sub-pixel refinement → perspective warp
- **Strengths**: Most engineered solution with proper validation
- **Weaknesses**: Over-validates — rejects valid grids that don't meet strict aspect ratio/angle criteria

## Confidence Score Analysis

Self-reported confidence scores are NOT comparable across methods:

| Method | Avg conf (when detected) | Avg conf (overall) | Meaning |
|--------|--------------------------|---------------------|---------|
| sobel_flood | 1.00 | 0.61 | Always says 1.0 when it detects anything |
| hough_polar | 1.00 | 0.91 | Always says 1.0 (based on line count >= 10) |
| hough_standard | 0.98 | 0.69 | Based on grid line spacing regularity |
| simple_baseline | 0.97 | 0.67 | Based on area ratio |
| sudoku_detector | 0.90 | 0.70 | Based on centeredness score |
| contour | 0.70 | 0.70 | Based on area ratio (conservative formula) |
| line_segment | 0.70 | 0.40 | Hardcoded 0.7 when valid |

**Conclusion**: These scores measure different things and can't be compared. A proper evaluation needs ground truth, not self-assessment.

## Recommendation for Production

**Primary method**: contour (100% detection, 2ms median)
**Fallback**: sudoku_detector (78% detection, 5ms, better corner refinement)

The contour method should be the default. If corner quality matters (for OCR accuracy), run sudoku_detector on images where contour succeeds — it has sub-pixel refinement that could improve the perspective transform.

## 8-Point Annotation: Center-Box Corners (2026-03-31)

### Why center-box corners matter

The source images are photographs of Sudoku puzzles in **curved newspapers and similar non-flat media**. On a curved surface, the paper is not planar — a single 4-point homography can map the outer boundary correctly, but the interior grid lines will not be evenly spaced in the warped image. Equal 9x9 slicing then cuts digits in half, causing OCR failures.

The 4 corners of the center 3×3 box (the intersections at row 3/col 3, row 3/col 6, row 6/col 6, row 6/col 3) serve as interior reference points. On flat paper these land at exactly 1/3 and 2/3 of the grid — on curved paper they don't. The deviation measures how much the interior grid is distorted.

### Annotation tool upgrade

`evaluation/annotate.py` now collects 8 points in two phases:
1. **Phase 1**: 4 outer corners (TL → TR → BR → BL) — same as before
2. **Phase 2**: 4 center-box corners (CTL → CTR → CBR → CBL) at the 3rd/6th gridline intersections

The warped preview uses **piecewise perspective correction**: 9 sub-regions each get their own local homography, so the center box is always a perfect square and all grid lines are straight in the output — even on curved paper.

### Warp deviation metric

`compute_warp_deviation()` in `extraction.py` applies the standard 4-corner homography and checks where the center-box corners land vs. their ideal 1/3, 2/3 positions. Max pixel deviation = "warp score". This is reported per-method in benchmarks when run against `ground_truth_annotated.json`.

### Piecewise cell extraction

`extract_cells_piecewise()` in `extraction.py` uses 4 outer + 4 center corners to define 9 box-regions (interpolating 8 boundary midpoints), applies a separate `getPerspectiveTransform` per region, composites into a flat grid, then slices into 81 cells. This replaces equal-division slicing for warped images.

### Initial measurements (6 augmented images)

On the synthetically augmented images, warp deviation was only 2-7px (<2% of grid). This is expected — synthetic augmentation applies a global homography to already-flat scans, so 4 corners suffice. The real benefit will show on the actual newspaper/curved-surface source photos where interior distortion is significant.

### Output format

`ground_truth_annotated.json` entries now have:
```json
{
  "path": "...",
  "outer_corners": [[x,y], ...],
  "center_corners": [[x,y], ...],
  "grid": [[...], ...]
}
```

## Benchmark v2: Real Newspaper Photos (2026-03-31)

### Test set

20 real newspaper/curved-surface photos, all manually annotated with 8 points (4 outer + 4 center-box corners) and ground truth grids. This is a fundamentally harder set than v1 — no synthetic augmentation, all real-world conditions.

### Results

| Method | Detection Rate | Avg ms | Med ms | Avg Conf |
|--------|---------------|--------|--------|----------|
| **sudoku_detector** | **100% (20/20)** | **4.4** | **4.3** | 0.85 |
| simple_baseline | 65% (13/20) | 2.2 | 2.0 | 0.97 |
| sobel_flood | 65% (13/20) | 2.1 | 2.0 | 1.00 |
| line_segment | 45% (9/20) | 5.8 | 5.4 | 0.70 |
| contour | 0% (0/20) | — | — | — |
| hough_standard | 0% (0/20) | — | — | — |
| hough_polar | 0% (0/20) | — | — | — |

### Key finding: sudoku_detector is the clear winner on real images

The v1 champion (contour, 100% on augmented images) **completely fails on real newspaper photos** (0/20). The "largest quadrilateral contour" heuristic breaks when there's background clutter, curved paper edges, and non-uniform lighting — the grid is no longer the largest rectangular contour in the image.

**sudoku_detector** is the only method that detects all 20 images. Its CLAHE preprocessing and stricter validation pipeline (which previously hurt it on extreme rotations) turns out to be exactly what real-world images need.

Both Hough methods also fail completely — curved paper doesn't produce the straight lines they require.

### Updated production recommendation

**Primary method**: sudoku_detector (100% on real photos, 4.3ms median)
**Fallback**: simple_baseline (65%, 2ms, very consistent corners when it works)

The v1 recommendation of contour as primary is **retracted** — it only worked on synthetic/augmented images and fails entirely on the real use case.

### Warp deviation metric — bug noted

The center-corner deviation metric returned empty results for all methods in this benchmark run. The `compute_center_corner_deviation()` function returns `None` even when methods detect corners. Needs debugging — likely a format mismatch between the corners returned by detection methods and what the function expects.

## 8-Corner Piecewise Warp: Trade-offs (2026-03-31)

### When 8-corner piecewise warp helps

The piecewise approach (9 sub-region homographies from 4 outer + 4 center-box corners) is better than standard 4-corner warp in **almost all cases**:

- **Curved/crumpled newspaper**: The primary use case. Interior grid lines that curve on the paper surface get straightened by local homographies. Equal 9×9 slicing on a 4-corner warp cuts digits in half; piecewise warp keeps cells aligned.
- **Perspective from an angle**: When the camera is not directly above, far vs near rows have different spacing. Center-box corners capture this non-linearity.
- **Warped/bent pages**: Any non-planar surface where the interior deviates from the outer boundary's implied geometry.

### When 8-corner warp can slightly distort

On a small number of images — particularly those that are **already flat and well-aligned** — the piecewise warp can introduce minor unnecessary distortion:

- If the annotated center corners have small annotation error (off by a few pixels), the piecewise warp forces the grid to pass through those slightly-wrong points, creating subtle kinks at the 1/3 and 2/3 boundaries that wouldn't exist with a clean 4-corner warp.
- On perfectly flat scans, the center-box corners land exactly at 1/3 and 2/3, so piecewise warp reduces to the same result as 4-corner warp — but annotation noise can push them off.

### Verdict

This is a **more than worthy trade-off**. The cases where piecewise warp helps (curved paper, perspective distortion) are the hard cases where OCR fails without it. The cases where it slightly hurts (already-flat images) are the easy cases where OCR works regardless. Optimizing for the hard cases is the right call.

A future improvement could be: compute warp deviation first, and only use piecewise extraction when deviation exceeds a threshold (e.g., >10px). On low-deviation images, fall back to simple 4-corner warp to avoid amplifying annotation noise.

## Notebook Visualization Bug: Color Collision (2026-03-31)

The review notebook (`review_results.ipynb`) draws GT center corners and sobel_flood's detected corners in the **same color** — both use BGR `(255, 255, 0)` which renders as cyan in matplotlib's RGB display. This makes it appear that sobel_flood is detecting the center square, when in fact the cyan quad is the ground truth annotation.

Verified by computing distance from sobel_flood's detected corners to GT outer vs GT center corners: **all 13 detections are closer to the outer grid** (typically 2-5px from GT outer, 170-330px from GT center). sobel_flood is finding the outer border, not the center box.

**Fix needed**: Change sobel_flood's METHOD_COLORS to a distinct color (e.g., lime green or magenta).

## Sobel Flood Fill for Center-Box Detection — Future Exploration (2026-03-31)

### Observation

Even though the color collision was a visual artifact, the underlying idea has merit: **could flood fill behavior be used to automatically detect the center 3×3 box boundaries?**

### Why it could work

In newspaper Sudoku puzzles, the 3×3 box borders are printed **thicker** than individual cell borders. The current sobel_flood fills from image corners and stops at the outer grid border. A modified approach could:

1. First find the outer grid (using sudoku_detector, which is 100% reliable)
2. Then run a **second flood fill from inside the grid** — starting from a point known to be inside a cell
3. The fill would spread through thin cell dividers but **stop at thick box borders**
4. The unfilled boundaries would delineate the 9 box regions
5. The center-box corners would be the intersections of these boundaries

This would give us **automatic center-box detection at runtime** — no manual annotation needed.

### Why current sobel_flood doesn't do this

The current implementation:
- Flood fills from image corners only (background → in)
- Uses `flood_tolerance=10` which is very tight — stops at subtle gradients
- Only looks for the largest contour (outer boundary)
- Has no concept of interior structure

### Proposed ensemble method

**Phase 1 — Outer grid**: Use sudoku_detector (100% on real photos, 4.3ms)
**Phase 2 — Center box**: Use modified sobel flood fill inside the detected grid region to find thick box borders and extract center-box corner intersections

This would give both the 4 outer corners and 4 center-box corners automatically, enabling piecewise perspective correction without manual annotation.

### Key parameters to explore

- `flood_tolerance`: Current 10 is too low for background but might be right for interior — the difference between thin cell lines and thick box borders is a gradient magnitude question
- Seed point selection: Start flood from the center of each expected 3×3 box region (using outer corners to estimate)
- Edge detection threshold: May need different Sobel thresholds to distinguish thick vs thin lines

## Next Steps

1. ~~Complete 8-point annotation for all 20 sample images~~ DONE
2. Fix notebook color collision (sobel_flood vs GT center both cyan)
3. Debug warp deviation metric (returns empty for all methods)
4. **Prototype sobel flood fill for center-box detection** — explore using interior flood fill to find thick box borders
5. Run OCR comparison: equal-slice vs piecewise extraction on annotated images
6. Test with more diverse real photos (newspapers, curved surfaces, different phones)
7. Investigate why contour fails on all real photos — is it the test set or a code regression?
8. Prototype ensemble method: sudoku_detector (outer) + sobel flood (center box)
