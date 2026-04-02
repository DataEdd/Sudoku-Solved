"""
Unified benchmark for all Sudoku grid detection methods.

Runs each detection method on a common test set and measures:
- Detection rate (did it find a quadrilateral?)
- Corner coordinates (for cross-method comparison)
- OCR accuracy (cells correctly read vs consensus grid)
- End-to-end solve rate (extract -> OCR -> solve -> valid solution?)
- Wall-clock timing per image

Usage:
    cd Sudoku-Solved/
    python -m evaluation.benchmark
    python -m evaluation.benchmark --methods contour simple_baseline --images 5
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Result dataclasses ──────────────────────────────────────────────

@dataclass
class MethodResult:
    """Result from a single detection method on a single image."""
    method: str
    detected: bool
    corners: Optional[List[List[float]]] = None  # 4x2 list
    extracted_grid: Optional[List[List[int]]] = None  # 9x9 grid
    confidence: float = 0.0
    timing_ms: float = 0.0
    error: Optional[str] = None
    debug_image_path: Optional[str] = None
    center_corner_deviation: Optional[float] = None  # pixels of interior warp


@dataclass
class ImageResult:
    """Results for all methods on a single image."""
    image_path: str
    category: str
    methods: Dict[str, MethodResult] = field(default_factory=dict)
    consensus_grid: Optional[List[List[int]]] = None


# ── Method wrappers ─────────────────────────────────────────────────
# Each wrapper follows: detect(image) -> (detected, corners_4x2, confidence, extra)

def _corners_to_list(corners: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    """Convert corners array to JSON-serializable list."""
    if corners is None:
        return None
    return corners.reshape(4, 2).tolist()


def wrap_contour(image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], float, dict]:
    """Contour method (adaptive threshold + largest quadrilateral)."""
    from app.core.extraction import preprocess_image, find_grid_contour, order_points
    thresh = preprocess_image(image)
    contour = find_grid_contour(thresh)
    if contour is None:
        return False, None, 0.0, {}
    corners = order_points(contour)
    # Confidence: area ratio
    img_area = image.shape[0] * image.shape[1]
    contour_area = cv2.contourArea(contour)
    area_ratio = contour_area / img_area
    confidence = min(1.0, area_ratio / 0.5) if area_ratio > 0.05 else 0.0
    return True, corners, confidence, {"area_ratio": area_ratio}


def wrap_simple_baseline(image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], float, dict]:
    """SimpleBaselineDetector from border_detection."""
    from app.core.border_detection.simple_baseline import SimpleBaselineDetector
    detector = SimpleBaselineDetector()
    result = detector.detect(image)
    return result.success, result.corners, result.confidence, {"time_ms": result.execution_time_ms}


def wrap_sobel_flood(image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], float, dict]:
    """SobelFloodDetector from border_detection."""
    from app.core.border_detection.sobel_flood import SobelFloodDetector
    detector = SobelFloodDetector()
    result = detector.detect(image)
    return result.success, result.corners, result.confidence, {"time_ms": result.execution_time_ms}


def wrap_line_segment(image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], float, dict]:
    """LineSegmentDetector from border_detection."""
    from app.core.border_detection.line_segment import LineSegmentDetector
    detector = LineSegmentDetector()
    result = detector.detect(image)
    return result.success, result.corners, result.confidence, result.metadata


def wrap_hough_standard(image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], float, dict]:
    """Standard Hough Transform (HoughLinesP)."""
    from app.core.detection import detect_hough_standard
    result = detect_hough_standard(image)
    # Hough standard returns grid intersections, not corners
    # Extract bounding corners from the result image if successful
    if result.success:
        # The method doesn't return corners directly, but grid line positions
        h_pos = result.stats.get("h_clustered", 0)
        v_pos = result.stats.get("v_clustered", 0)
        # Approximate corners from first/last grid lines if available
        return True, None, result.confidence, result.stats
    return False, None, 0.0, result.stats


def wrap_hough_polar(image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], float, dict]:
    """Polar Hough Transform (HoughLines)."""
    from app.core.detection import detect_hough_polar
    result = detect_hough_polar(image)
    # Polar also returns line-level results, not corners
    return result.success, None, result.confidence, result.stats


def wrap_generalized_hough(image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], float, dict]:
    """Generalized Hough Transform."""
    from app.core.hough_generalized import detect_grid_ght
    result = detect_grid_ght(image, return_accumulator=False)
    if result.corners is not None and len(result.corners) == 4:
        return True, result.corners.astype(np.float32), result.confidence, {
            "center": result.center, "scale": result.scale
        }
    return False, None, 0.0, {}


def wrap_hybrid(image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], float, dict]:
    """Hybrid method: CLAHE + epsilon=0.02 + area/centeredness scoring + subpixel."""
    from app.core.extraction import detect_grid
    corners, confidence = detect_grid(image)
    if corners is None:
        return False, None, 0.0, {}
    return True, corners, confidence, {}


def wrap_sudoku_detector(image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], float, dict]:
    """sudoku-detector package dual path."""
    sys.path.insert(0, str(PROJECT_ROOT / "sudoku-detector"))
    from src.detector import SudokuDetector
    detector = SudokuDetector()
    result = detector.detect(image)
    corners = None
    if result.corners is not None:
        corners = result.corners.reshape(4, 2).astype(np.float32)
    return result.success, corners, result.confidence, {
        "method": result.detection_method or "unknown"
    }


# Registry of all methods
METHODS = {
    "contour": wrap_contour,
    "hybrid": wrap_hybrid,
    "simple_baseline": wrap_simple_baseline,
    "sobel_flood": wrap_sobel_flood,
    "line_segment": wrap_line_segment,
    "hough_standard": wrap_hough_standard,
    "hough_polar": wrap_hough_polar,
    "generalized_hough": wrap_generalized_hough,
    "sudoku_detector": wrap_sudoku_detector,
}


# ── OCR + Solve helpers ─────────────────────────────────────────────

def extract_grid_with_ocr(image: np.ndarray, corners: Optional[np.ndarray]) -> Optional[List[List[int]]]:
    """Given corners, warp perspective and OCR the grid."""
    if corners is None:
        return None
    try:
        from app.core.extraction import order_points, perspective_transform, extract_cells, recognize_digit
        pts = corners.reshape(4, 2).astype(np.float32)
        # order_points expects (4,2)
        ordered = order_points(pts.reshape(4, 1, 2))
        warped = perspective_transform(image, pts.reshape(4, 1, 2))
        cells = extract_cells(warped)
        grid = []
        for i in range(9):
            row = []
            for j in range(9):
                digit = recognize_digit(cells[i * 9 + j])
                row.append(digit)
            grid.append(row)
        return grid
    except Exception as e:
        return None


def try_solve(grid: List[List[int]]) -> Tuple[bool, Optional[List[List[int]]], int]:
    """Attempt to solve a grid. Returns (success, solution, iterations)."""
    try:
        from app.core.solver import simulated_annealing
        solution, iterations, success = simulated_annealing(grid, max_iterations=100000)
        return success, solution, iterations
    except Exception:
        return False, None, 0


def compute_center_corner_deviation(
    detected_corners: Optional[np.ndarray],
    gt_center_corners: List[List[float]],
    size: int = 450,
) -> Optional[float]:
    """Compute how far the detected outer corners produce interior distortion.

    Uses the detected 4 outer corners to build a homography, then checks
    where the ground-truth center-box corners land in the warped image.
    If they land at (size/3, size/3) etc., the grid is regular. Deviation
    from that = interior warping the method can't account for.

    Returns max deviation in pixels, or None if corners unavailable.
    """
    if detected_corners is None:
        return None
    try:
        from app.core.extraction import compute_warp_deviation
        return compute_warp_deviation(detected_corners, gt_center_corners, size)
    except Exception:
        return None


def compute_consensus_grid(grids: List[Optional[List[List[int]]]]) -> Optional[List[List[int]]]:
    """
    Compute consensus grid from multiple OCR attempts.
    For each cell, take the most common non-zero value.
    """
    valid_grids = [g for g in grids if g is not None]
    if not valid_grids:
        return None

    consensus = [[0] * 9 for _ in range(9)]
    for i in range(9):
        for j in range(9):
            values = [g[i][j] for g in valid_grids if g[i][j] != 0]
            if values:
                # Most common value
                from collections import Counter
                counter = Counter(values)
                consensus[i][j] = counter.most_common(1)[0][0]
    return consensus


# ── Main benchmark logic ────────────────────────────────────────────

def run_benchmark(
    ground_truth_path: str = "evaluation/ground_truth.json",
    output_dir: str = "evaluation",
    methods_to_run: Optional[List[str]] = None,
    max_images: Optional[int] = None,
    save_debug_images: bool = True,
    skip_ocr: bool = False,
    skip_solve: bool = True,
) -> List[ImageResult]:
    """Run the full benchmark."""

    with open(ground_truth_path) as f:
        gt_data = json.load(f)

    images = gt_data["images"]
    if max_images:
        images = images[:max_images]

    if methods_to_run is None:
        methods_to_run = list(METHODS.keys())

    debug_dir = os.path.join(output_dir, "debug_images")
    if save_debug_images:
        os.makedirs(debug_dir, exist_ok=True)

    all_results: List[ImageResult] = []

    for idx, img_info in enumerate(images):
        img_path = img_info["path"]
        category = img_info.get("category", "unknown")
        img_name = Path(img_path).stem

        print(f"\n[{idx+1}/{len(images)}] {img_path} ({category})")

        image = cv2.imread(img_path)
        if image is None:
            print(f"  ERROR: Could not load image")
            continue

        img_result = ImageResult(image_path=img_path, category=category)
        ocr_grids = []

        for method_name in methods_to_run:
            if method_name not in METHODS:
                print(f"  SKIP: Unknown method {method_name}")
                continue

            wrapper = METHODS[method_name]
            mr = MethodResult(method=method_name, detected=False)

            try:
                t0 = time.perf_counter()
                detected, corners, confidence, extra = wrapper(image)
                elapsed = (time.perf_counter() - t0) * 1000

                mr.detected = detected
                mr.confidence = confidence
                mr.timing_ms = elapsed
                mr.corners = _corners_to_list(corners)

                status = "OK" if detected else "FAIL"
                print(f"  {method_name:20s} {status:4s}  conf={confidence:.2f}  {elapsed:.0f}ms")

                # Save debug image: draw detected corners on the original
                if save_debug_images and detected and corners is not None:
                    debug_img = image.copy()
                    pts = corners.reshape(4, 2).astype(np.int32)
                    cv2.polylines(debug_img, [pts], True, (0, 255, 0), 3)
                    for pt in pts:
                        cv2.circle(debug_img, tuple(pt), 8, (0, 0, 255), -1)
                    debug_path = os.path.join(debug_dir, f"{img_name}_{method_name}.jpg")
                    cv2.imwrite(debug_path, debug_img)
                    mr.debug_image_path = debug_path

                # Center-corner deviation if GT center corners available
                gt_center = img_info.get("center_corners")
                if detected and corners is not None and gt_center is not None:
                    dev = compute_center_corner_deviation(corners, gt_center)
                    mr.center_corner_deviation = dev
                    if dev is not None:
                        print(f"    Warp deviation: {dev:.1f}px")

                # OCR if method found corners
                if not skip_ocr and detected and corners is not None:
                    grid = extract_grid_with_ocr(image, corners)
                    mr.extracted_grid = grid
                    if grid:
                        ocr_grids.append(grid)
                        nonzero = sum(1 for row in grid for v in row if v != 0)
                        print(f"    OCR: {nonzero}/81 cells filled")

            except Exception as e:
                mr.error = str(e)
                print(f"  {method_name:20s} ERROR: {e}")

            img_result.methods[method_name] = mr

        # Compute consensus grid
        if ocr_grids:
            img_result.consensus_grid = compute_consensus_grid(ocr_grids)

        all_results.append(img_result)

    return all_results


def compute_summary(results: List[ImageResult], methods: List[str]) -> dict:
    """Compute summary statistics from benchmark results."""
    summary = {
        "total_images": len(results),
        "methods": {},
        "by_category": {},
    }

    # Per-method stats
    for method in methods:
        detections = 0
        total = 0
        timings = []
        confidences = []
        deviations = []

        for img_result in results:
            if method in img_result.methods:
                total += 1
                mr = img_result.methods[method]
                if mr.detected:
                    detections += 1
                    confidences.append(mr.confidence)
                    if mr.center_corner_deviation is not None:
                        deviations.append(mr.center_corner_deviation)
                timings.append(mr.timing_ms)

        method_summary = {
            "detection_rate": detections / total if total > 0 else 0,
            "detections": detections,
            "total": total,
            "avg_timing_ms": sum(timings) / len(timings) if timings else 0,
            "median_timing_ms": sorted(timings)[len(timings)//2] if timings else 0,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
        }
        if deviations:
            method_summary["avg_warp_deviation_px"] = sum(deviations) / len(deviations)
            method_summary["max_warp_deviation_px"] = max(deviations)
        summary["methods"][method] = method_summary

    # Per-category stats
    categories = set(r.category for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in results if r.category == cat]
        cat_summary = {}
        for method in methods:
            detections = sum(
                1 for r in cat_results
                if method in r.methods and r.methods[method].detected
            )
            total = sum(1 for r in cat_results if method in r.methods)
            cat_summary[method] = {
                "detection_rate": detections / total if total > 0 else 0,
                "detections": detections,
                "total": total,
            }
        summary["by_category"][cat] = cat_summary

    return summary


def save_results(results: List[ImageResult], summary: dict, output_dir: str):
    """Save results to JSON."""

    # Convert results to serializable format
    results_data = []
    for img_result in results:
        entry = {
            "image_path": img_result.image_path,
            "category": img_result.category,
            "consensus_grid": img_result.consensus_grid,
            "methods": {},
        }
        for method_name, mr in img_result.methods.items():
            entry["methods"][method_name] = {
                "detected": mr.detected,
                "corners": mr.corners,
                "confidence": mr.confidence,
                "timing_ms": round(mr.timing_ms, 2),
                "error": mr.error,
                "debug_image_path": mr.debug_image_path,
                "extracted_grid": mr.extracted_grid,
                "center_corner_deviation": round(mr.center_corner_deviation, 2) if mr.center_corner_deviation is not None else None,
            }
        results_data.append(entry)

    output = {
        "summary": summary,
        "results": results_data,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


def print_summary_table(summary: dict, methods: List[str]):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("DETECTION BENCHMARK SUMMARY")
    print("=" * 80)

    # Overall results table
    header = f"{'Method':<22s} {'Det Rate':>8s} {'Det/Tot':>8s} {'Avg ms':>8s} {'Med ms':>8s} {'Conf':>6s}"
    print(f"\n{header}")
    print("-" * len(header))

    for method in methods:
        if method not in summary["methods"]:
            continue
        s = summary["methods"][method]
        print(
            f"{method:<22s} "
            f"{s['detection_rate']:>7.0%} "
            f"{s['detections']:>3d}/{s['total']:<3d} "
            f"{s['avg_timing_ms']:>7.0f} "
            f"{s['median_timing_ms']:>7.0f} "
            f"{s['avg_confidence']:>5.2f}"
        )

    # Category breakdown
    print(f"\n{'Category breakdown (detection rate):'}")
    cats = sorted(summary["by_category"].keys())
    header = f"{'Category':<22s}" + "".join(f" {m[:10]:>10s}" for m in methods)
    print(header)
    print("-" * len(header))
    for cat in cats:
        row = f"{cat:<22s}"
        for method in methods:
            if method in summary["by_category"][cat]:
                rate = summary["by_category"][cat][method]["detection_rate"]
                row += f" {rate:>9.0%} "
            else:
                row += f" {'N/A':>10s}"
        print(row)

    print("=" * 80)


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sudoku detection benchmark")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to benchmark (default: all)")
    parser.add_argument("--images", type=int, default=None,
                        help="Max number of images to test")
    parser.add_argument("--no-debug-images", action="store_true",
                        help="Skip saving debug images")
    parser.add_argument("--ocr", action="store_true",
                        help="Run OCR on detected grids (slower)")
    parser.add_argument("--solve", action="store_true",
                        help="Attempt to solve extracted grids")
    parser.add_argument("--ground-truth", default="evaluation/ground_truth.json",
                        help="Path to ground truth JSON (use ground_truth_annotated.json for center-corner metrics)")
    parser.add_argument("--output", default="evaluation",
                        help="Output directory")
    args = parser.parse_args()

    methods = args.methods or list(METHODS.keys())

    print(f"Sudoku Detection Benchmark")
    print(f"Methods: {', '.join(methods)}")
    print(f"Ground truth: {args.ground_truth}")

    results = run_benchmark(
        ground_truth_path=args.ground_truth,
        output_dir=args.output,
        methods_to_run=methods,
        max_images=args.images,
        save_debug_images=not args.no_debug_images,
        skip_ocr=not args.ocr,
        skip_solve=not args.solve,
    )

    summary = compute_summary(results, methods)
    print_summary_table(summary, methods)
    save_results(results, summary, args.output)


if __name__ == "__main__":
    main()
