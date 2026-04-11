"""
OCR evaluation script — benchmark digit recognition against ground truth.

Evaluates the full pipeline (detect → warp → extract → OCR) against
annotated ground truth grids. Reports per-cell accuracy, confusion matrix,
confidence calibration, and error patterns.

Usage:
    python -m evaluation.evaluate_ocr              # Full evaluation
    python -m evaluation.evaluate_ocr --verbose     # Per-image breakdown
    python -m evaluation.evaluate_ocr --piecewise   # Compare piecewise vs simple warp
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.extraction import (
    detect_grid,
    extract_cells,
    extract_cells_piecewise,
    order_points,
    perspective_transform,
    recognize_cells,
)

GT_PATH = Path(__file__).parent / "ground_truth.json"
IMAGES_DIR = PROJECT_ROOT / "Examples" / "Ground Example"

# corners_16 layout:
#   P0  P1  P2  P3     (top row)
#   P4  P5  P6  P7
#   P8  P9  P10 P11
#   P12 P13 P14 P15    (bottom row)
GT_OUTER_INDICES = [0, 3, 15, 12]  # TL, TR, BR, BL
GT_CENTER_INDICES = [5, 6, 10, 9]  # CTL, CTR, CBR, CBL


def load_ground_truth() -> List[dict]:
    """Load ground truth data."""
    with open(GT_PATH) as f:
        return json.load(f)["images"]


def gt_corners_outer(entry: dict) -> np.ndarray:
    """Extract 4 outer corners from corners_16 ground truth."""
    c16 = entry["corners_16"]
    return np.array([c16[i] for i in GT_OUTER_INDICES], dtype=np.float32)


def gt_corners_center(entry: dict) -> np.ndarray:
    """Extract 4 center-box corners from corners_16 ground truth."""
    c16 = entry["corners_16"]
    return np.array([c16[i] for i in GT_CENTER_INDICES], dtype=np.float32)


def match_gt(predicted: int, gt_val) -> bool:
    """Check if predicted digit matches GT (supports multi-value cells)."""
    if isinstance(gt_val, list):
        return predicted in gt_val
    return predicted == gt_val


def is_gt_filled(gt_val) -> bool:
    """Check if a GT cell is filled (non-empty)."""
    if isinstance(gt_val, list):
        return any(v != 0 for v in gt_val)
    return gt_val != 0


def gt_digit(gt_val) -> int:
    """Get the primary digit from a GT cell (first non-zero, or 0)."""
    if isinstance(gt_val, list):
        for v in gt_val:
            if v != 0:
                return v
        return 0
    return gt_val


def evaluate_single(
    image: np.ndarray,
    gt_grid: List[List],
    corners: Optional[np.ndarray] = None,
    use_piecewise: bool = False,
    gt_center_corners: Optional[np.ndarray] = None,
) -> dict:
    """Evaluate OCR on a single image.

    Args:
        image: BGR image
        gt_grid: 9x9 ground truth grid
        corners: 4 outer corners (if None, runs detect_grid)
        use_piecewise: Use piecewise warp with center corners
        gt_center_corners: 4 center-box corners for piecewise warp

    Returns dict with per-cell results.
    """
    result = {
        "detected": False,
        "used_gt_corners": corners is not None,
        "cells": [],  # list of {row, col, predicted, gt, confidence, correct}
    }

    if corners is None:
        detected_corners, confidence = detect_grid(image)
        if detected_corners is None:
            return result
        corners = detected_corners
        result["detection_confidence"] = float(confidence)

    result["detected"] = True

    if use_piecewise and gt_center_corners is not None:
        outer = order_points(corners.reshape(4, 1, 2))
        cells = extract_cells_piecewise(image, outer, gt_center_corners, size=450)
    else:
        contour = corners.reshape(4, 1, 2).astype(np.float32)
        warped = perspective_transform(image, contour)
        cells = extract_cells(warped)

    grid, confidence_map = recognize_cells(cells)

    for i in range(9):
        for j in range(9):
            predicted = grid[i][j]
            gt_val = gt_grid[i][j]
            conf = confidence_map[i][j]
            correct = match_gt(predicted, gt_val)

            result["cells"].append({
                "row": i,
                "col": j,
                "predicted": predicted,
                "gt": gt_val,
                "gt_digit": gt_digit(gt_val),
                "confidence": conf,
                "correct": correct,
                "gt_filled": is_gt_filled(gt_val),
                "predicted_filled": predicted != 0,
            })

    return result


def run_evaluation(
    verbose: bool = False,
    use_piecewise: bool = False,
    use_gt_corners: bool = False,
) -> List[dict]:
    """Run OCR evaluation across all GT images."""
    gt_data = load_ground_truth()
    results = []

    for entry in gt_data:
        filename = Path(entry["path"]).name
        image_path = IMAGES_DIR / filename
        if not image_path.exists():
            print(f"  SKIP {filename} — file not found")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  SKIP {filename} — failed to load")
            continue

        corners = None
        center_corners = None

        if use_gt_corners or use_piecewise:
            corners = gt_corners_outer(entry)
            if use_piecewise:
                center_corners = gt_corners_center(entry)

        result = evaluate_single(
            image, entry["grid"], corners,
            use_piecewise=use_piecewise,
            gt_center_corners=center_corners,
        )
        result["filename"] = filename

        if verbose and result["detected"]:
            cells = result["cells"]
            correct = sum(1 for c in cells if c["correct"])
            filled_correct = sum(
                1 for c in cells if c["gt_filled"] and c["correct"]
            )
            filled_total = sum(1 for c in cells if c["gt_filled"])
            print(
                f"  {filename}: {correct}/81 all, "
                f"{filled_correct}/{filled_total} filled"
            )
        elif verbose:
            print(f"  {filename}: NOT DETECTED")

        results.append(result)

    return results


def print_summary(results: List[dict], label: str = ""):
    """Print evaluation summary."""
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

    detected = [r for r in results if r["detected"]]
    not_detected = [r for r in results if not r["detected"]]

    print(f"\nDetection: {len(detected)}/{len(results)}")
    if not_detected:
        print(f"  Failed: {', '.join(r['filename'] for r in not_detected)}")

    if not detected:
        print("No images detected — cannot evaluate OCR.")
        return

    # Aggregate all cells
    all_cells = [c for r in detected for c in r["cells"]]
    filled_cells = [c for c in all_cells if c["gt_filled"]]
    empty_cells = [c for c in all_cells if not c["gt_filled"]]

    # Overall accuracy
    all_correct = sum(1 for c in all_cells if c["correct"])
    filled_correct = sum(1 for c in filled_cells if c["correct"])
    empty_correct = sum(1 for c in empty_cells if c["correct"])

    print(f"\nOverall Accuracy: {all_correct}/{len(all_cells)} "
          f"({100 * all_correct / len(all_cells):.1f}%)")
    print(f"  Filled cells: {filled_correct}/{len(filled_cells)} "
          f"({100 * filled_correct / len(filled_cells):.1f}%)")
    print(f"  Empty cells:  {empty_correct}/{len(empty_cells)} "
          f"({100 * empty_correct / len(empty_cells):.1f}%)")

    # --- Error breakdown ---
    print(f"\nError Breakdown:")
    # Filled GT cell → predicted empty (missed digit)
    missed = sum(
        1 for c in filled_cells
        if not c["predicted_filled"]
    )
    # Filled GT cell → predicted wrong digit
    wrong_digit = sum(
        1 for c in filled_cells
        if c["predicted_filled"] and not c["correct"]
    )
    # Empty GT cell → predicted filled (hallucinated digit)
    hallucinated = sum(
        1 for c in empty_cells
        if c["predicted_filled"]
    )
    print(f"  Missed digits (filled→0):     {missed}")
    print(f"  Wrong digits (filled→wrong):   {wrong_digit}")
    print(f"  Hallucinated (empty→digit):    {hallucinated}")

    # --- Confusion matrix for filled cells ---
    print(f"\nConfusion Matrix (filled cells, GT rows → Predicted cols):")
    confusion = np.zeros((10, 10), dtype=int)  # 0-9 x 0-9
    for c in filled_cells:
        gt_d = c["gt_digit"]
        pred = c["predicted"]
        confusion[gt_d][pred] += 1

    # Print header
    print(f"     {'  '.join(str(d) for d in range(10))}")
    for gt_d in range(1, 10):  # skip 0 row (empty GT)
        row_str = f"  {gt_d}: "
        for pred_d in range(10):
            val = confusion[gt_d][pred_d]
            if val == 0:
                row_str += " . "
            elif gt_d == pred_d:
                row_str += f"{val:2d}*"
            else:
                row_str += f"{val:2d} "
        total = sum(confusion[gt_d])
        correct_pct = 100 * confusion[gt_d][gt_d] / total if total > 0 else 0
        row_str += f"  | {correct_pct:.0f}%"
        print(row_str)

    # --- Confidence calibration ---
    print(f"\nConfidence Calibration:")
    bins = [(0.0, 0.5, "Low  <0.5"), (0.5, 0.8, "Med  0.5-0.8"),
            (0.8, 1.01, "High >=0.8")]
    for lo, hi, label in bins:
        bin_cells = [c for c in all_cells if lo <= c["confidence"] < hi]
        if bin_cells:
            acc = sum(1 for c in bin_cells if c["correct"]) / len(bin_cells)
            print(f"  {label}: {len(bin_cells):4d} cells, "
                  f"{100 * acc:.1f}% accurate")
        else:
            print(f"  {label}:    0 cells")

    # --- Edge vs interior ---
    print(f"\nPosition Analysis:")
    edge_cells = [c for c in filled_cells if c["row"] in (0, 8) or c["col"] in (0, 8)]
    interior_cells = [c for c in filled_cells if c["row"] not in (0, 8) and c["col"] not in (0, 8)]
    if edge_cells:
        edge_acc = sum(1 for c in edge_cells if c["correct"]) / len(edge_cells)
        print(f"  Edge cells:     {100 * edge_acc:.1f}% ({len(edge_cells)} cells)")
    if interior_cells:
        int_acc = sum(1 for c in interior_cells if c["correct"]) / len(interior_cells)
        print(f"  Interior cells: {100 * int_acc:.1f}% ({len(interior_cells)} cells)")

    # --- Per-image summary ---
    print(f"\nPer-Image Accuracy (filled cells):")
    image_stats = []
    for r in detected:
        fc = [c for c in r["cells"] if c["gt_filled"]]
        if fc:
            acc = sum(1 for c in fc if c["correct"]) / len(fc)
            image_stats.append((r["filename"], acc, len(fc)))

    image_stats.sort(key=lambda x: x[1])
    for fname, acc, n in image_stats[:5]:
        print(f"  WORST: {fname}: {100 * acc:.0f}% ({n} filled)")
    print(f"  ...")
    for fname, acc, n in image_stats[-5:]:
        print(f"  BEST:  {fname}: {100 * acc:.0f}% ({n} filled)")

    median_acc = np.median([s[1] for s in image_stats])
    mean_acc = np.mean([s[1] for s in image_stats])
    print(f"\n  Median per-image accuracy: {100 * median_acc:.1f}%")
    print(f"  Mean per-image accuracy:   {100 * mean_acc:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="OCR evaluation against ground truth")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-image results")
    parser.add_argument("--piecewise", action="store_true",
                        help="Compare piecewise vs simple warp (uses GT corners)")
    parser.add_argument("--gt-corners", action="store_true",
                        help="Use GT corners instead of detect_grid")
    args = parser.parse_args()

    if args.piecewise:
        # Compare simple vs piecewise, both using GT corners
        print("Running simple warp with GT corners...")
        simple_results = run_evaluation(
            verbose=args.verbose, use_gt_corners=True,
        )
        print_summary(simple_results, "Simple Warp (GT Corners)")

        print("\nRunning piecewise warp with GT corners...")
        piecewise_results = run_evaluation(
            verbose=args.verbose, use_piecewise=True,
        )
        print_summary(piecewise_results, "Piecewise Warp (GT Corners)")
    else:
        mode = "GT corners" if args.gt_corners else "detect_grid"
        print(f"Running OCR evaluation ({mode})...")
        results = run_evaluation(
            verbose=args.verbose, use_gt_corners=args.gt_corners,
        )
        print_summary(
            results,
            f"OCR Evaluation — {mode}",
        )


if __name__ == "__main__":
    main()
