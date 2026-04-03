"""
Detection evaluation script for hyperparameter tuning.

Evaluates grid detection accuracy against ground truth corner annotations.
Supports single-run evaluation and parameter sweep modes.

Usage:
    # Single run with current defaults
    python -m evaluation.evaluate_detection

    # Single run with custom params
    python -m evaluation.evaluate_detection --blur-k 7 --block-size 13

    # Full parameter sweep
    python -m evaluation.evaluate_detection --sweep

    # Sweep with specific ranges
    python -m evaluation.evaluate_detection --sweep --blur-k 3 5 7 --block-size 9 11 13 15
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Project root for imports and file resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.extraction import order_points

# Ground truth corner indices: outer corners from corners_16
# Layout:  P0  P1  P2  P3     (top)
#          P4  P5  P6  P7
#          P8  P9  P10 P11
#          P12 P13 P14 P15    (bottom)
GT_OUTER_INDICES = [0, 3, 15, 12]  # TL, TR, BR, BL

# Current production defaults (from detect_grid / preprocess_image)
DEFAULTS = {
    "clahe_clip": 2.0,
    "clahe_tile": (8, 8),
    "blur_k": 5,
    "block_size": 11,
    "thresh_c": 2,
    "epsilon": 0.02,
    "min_area_ratio": 0.01,
    "morph_dilate": 0,
    "morph_erode": 0,
    "area_weight": 0.4,
    "squareness_weight": 0.3,
    "center_weight": 0.3,
    "squareness_min": 0.0,
    "area_max_ratio": 0.99,
    "contour_mode": "external",
}

# Default sweep ranges (single-element lists for new params = no sweep by default)
SWEEP_DEFAULTS = {
    "clahe_clip": [1.0, 2.0, 3.0, 4.0],
    "clahe_tile": [(4, 4), (8, 8), (16, 16)],
    "blur_k": [3, 5, 7, 9],
    "block_size": [7, 9, 11, 13, 15],
    "thresh_c": [1, 2, 3, 5, 7],
    "epsilon": [0.01, 0.015, 0.02, 0.03, 0.04],
    "min_area_ratio": [0.005, 0.01, 0.02, 0.05],
    "morph_dilate": [0],
    "morph_erode": [0],
    "area_weight": [0.4],
    "squareness_weight": [0.3],
    "center_weight": [0.3],
    "squareness_min": [0.0],
    "area_max_ratio": [0.99],
    "contour_mode": ["external"],
}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def compute_iou(quad_a: np.ndarray, quad_b: np.ndarray) -> float:
    """Compute IoU of two convex quadrilaterals using cv2.intersectConvexConvex."""
    a = cv2.convexHull(quad_a.reshape(-1, 1, 2).astype(np.float32))
    b = cv2.convexHull(quad_b.reshape(-1, 1, 2).astype(np.float32))

    inter_area, _ = cv2.intersectConvexConvex(a, b)
    if inter_area <= 0:
        return 0.0

    area_a = cv2.contourArea(a)
    area_b = cv2.contourArea(b)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Parameterized detection
# ---------------------------------------------------------------------------

def detect_with_params(
    image: np.ndarray,
    clahe_clip: float = 2.0,
    clahe_tile: Tuple[int, int] = (8, 8),
    blur_k: int = 5,
    block_size: int = 11,
    thresh_c: int = 2,
    epsilon: float = 0.02,
    min_area_ratio: float = 0.01,
    morph_dilate: int = 0,
    morph_erode: int = 0,
    area_weight: float = 0.4,
    squareness_weight: float = 0.3,
    center_weight: float = 0.3,
    squareness_min: float = 0.0,
    area_max_ratio: float = 0.99,
    contour_mode: str = "external",
) -> Tuple[Optional[np.ndarray], float]:
    """Parameterized detection -- same logic as detect_grid() but with tunable params.

    Does NOT include the fallback path (no-CLAHE retry) so that each parameter
    set is evaluated in isolation.

    Returns:
        (corners_4x2 [TL, TR, BR, BL] or None, confidence).
    """
    h, w = image.shape[:2]
    image_area = h * w
    img_center = np.array([w / 2.0, h / 2.0])
    max_dist = np.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocessing: CLAHE + blur + adaptive threshold
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (blur_k, blur_k), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block_size, thresh_c,
    )

    # Morphological operations (connect broken lines / remove noise)
    if morph_dilate > 0:
        thresh = cv2.dilate(
            thresh, np.ones((morph_dilate, morph_dilate), np.uint8), iterations=1,
        )
    if morph_erode > 0:
        thresh = cv2.erode(
            thresh, np.ones((morph_erode, morph_erode), np.uint8), iterations=1,
        )

    # Find contours
    cv_mode = cv2.RETR_TREE if contour_mode == "tree" else cv2.RETR_EXTERNAL
    contours, _ = cv2.findContours(thresh, cv_mode, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    # Filter to quad candidates
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_ratio * image_area or area > area_max_ratio * image_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        # Squareness filter
        if squareness_min > 0:
            x, y, bw, bh = cv2.boundingRect(approx)
            sq = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
            if sq < squareness_min:
                continue
        candidates.append((approx, area))

    if not candidates:
        return None, 0.0

    max_area = max(c[1] for c in candidates)

    # Score candidates (inline scoring with configurable weights)
    best_contour = None
    best_score = -1.0
    for approx, area in candidates:
        quad = approx.reshape(4, 2).astype(np.float32)
        # Area normalized to largest candidate
        area_norm = area / max_area if max_area > 0 else 0.0
        # Squareness from bounding rect
        x, y, bw, bh = cv2.boundingRect(approx)
        squareness = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0.0
        # Centeredness
        centroid = np.mean(quad, axis=0)
        dist = np.linalg.norm(centroid - img_center)
        centeredness = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
        # Weighted score
        s = (area_norm * area_weight) + (squareness * squareness_weight) + (centeredness * center_weight)
        if s > best_score:
            best_score = s
            best_contour = approx

    if best_contour is None:
        return None, 0.0

    # Order corners: TL, TR, BR, BL
    quad = best_contour.reshape(4, 2).astype(np.float32)
    corners = order_points(quad.reshape(4, 1, 2))

    # Sub-pixel refinement
    corners_sp = corners.reshape(-1, 1, 2).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(
        gray, corners_sp, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria,
    )
    corners = refined.reshape(4, 2).astype(np.float32)

    # Confidence based on area ratio
    best_area = cv2.contourArea(best_contour)
    ar = best_area / image_area
    confidence = min(1.0, ar / 0.5) if ar > 0.05 else 0.0

    return corners, confidence


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_image(
    image_path: str,
    gt_corners: np.ndarray,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate detection on a single image against GT corners.

    Returns a dict with per-corner errors, mean error, IoU, and confidence.
    """
    image = cv2.imread(image_path)
    if image is None:
        return {
            "path": image_path,
            "detected": False,
            "error": "Could not load image",
            "corner_errors": [None, None, None, None],
            "mean_error": None,
            "iou": 0.0,
            "confidence": 0.0,
        }

    detected_corners, confidence = detect_with_params(image, **params)

    if detected_corners is None:
        return {
            "path": image_path,
            "detected": False,
            "corner_errors": [None, None, None, None],
            "mean_error": None,
            "iou": 0.0,
            "confidence": 0.0,
        }

    # Per-corner Euclidean distance (TL, TR, BR, BL)
    corner_errors = [
        round(float(np.linalg.norm(detected_corners[i] - gt_corners[i])), 2)
        for i in range(4)
    ]

    mean_error = round(float(np.mean(corner_errors)), 2)
    iou = round(compute_iou(detected_corners, gt_corners), 4)

    return {
        "path": image_path,
        "detected": True,
        "corner_errors": corner_errors,
        "mean_error": mean_error,
        "iou": iou,
        "confidence": round(confidence, 4),
    }


def load_ground_truth() -> List[Dict]:
    """Load ground truth and extract outer corners for each image."""
    gt_path = PROJECT_ROOT / "evaluation" / "ground_truth.json"
    with open(gt_path) as f:
        data = json.load(f)

    images = []
    for entry in data["images"]:
        corners_16 = np.array(entry["corners_16"], dtype=np.float32)
        outer = corners_16[GT_OUTER_INDICES]  # TL, TR, BR, BL

        image_path = str(PROJECT_ROOT / entry["path"])
        images.append({
            "path": image_path,
            "display_path": entry["path"],
            "gt_corners": outer,
        })

    return images


def evaluate_run(
    images: List[Dict],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Run evaluation on all images with given parameters."""
    results = []
    for img_info in images:
        result = evaluate_image(img_info["path"], img_info["gt_corners"], params)
        result["path"] = img_info["display_path"]
        results.append(result)

    # Summary statistics
    detected_count = sum(1 for r in results if r["detected"])
    total = len(results)
    errors = [r["mean_error"] for r in results if r["mean_error"] is not None]
    ious = [r["iou"] for r in results if r["detected"]]

    summary = {
        "detection_rate": f"{detected_count}/{total}",
        "detected": detected_count,
        "total": total,
        "mean_error_px": round(float(np.mean(errors)), 2) if errors else None,
        "median_error_px": round(float(np.median(errors)), 2) if errors else None,
        "max_error_px": round(float(np.max(errors)), 2) if errors else None,
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
    }

    return {"params": params, "summary": summary, "per_image": results}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_single_results(run: Dict[str, Any]) -> None:
    """Print per-image results and summary for a single run."""
    params = run["params"]
    summary = run["summary"]

    print(f"\n{'=' * 80}")
    print("Detection Evaluation Results")
    print(f"{'=' * 80}")
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    print(f"Params: {param_str}\n")

    # Per-image table
    hdr = (
        f"{'Image':<45} {'Det':>3} {'TL':>6} {'TR':>6} "
        f"{'BR':>6} {'BL':>6} {'Mean':>7} {'IoU':>6} {'Conf':>5}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in run["per_image"]:
        name = Path(r["path"]).name
        if len(name) > 44:
            name = name[:41] + "..."

        if r["detected"]:
            e = r["corner_errors"]
            print(
                f"{name:<45} {'Y':>3} {e[0]:>6.1f} {e[1]:>6.1f} "
                f"{e[2]:>6.1f} {e[3]:>6.1f} {r['mean_error']:>7.1f} "
                f"{r['iou']:>6.3f} {r['confidence']:>5.2f}"
            )
        else:
            print(
                f"{name:<45} {'N':>3} {'--':>6} {'--':>6} "
                f"{'--':>6} {'--':>6} {'--':>7} {'--':>6} {'--':>5}"
            )

    print("-" * len(hdr))
    print(f"\nSummary:")
    print(f"  Detection rate: {summary['detection_rate']}")
    if summary["mean_error_px"] is not None:
        print(f"  Mean error:     {summary['mean_error_px']:.1f}px")
        print(f"  Median error:   {summary['median_error_px']:.1f}px")
        print(f"  Max error:      {summary['max_error_px']:.1f}px")
        print(f"  Mean IoU:       {summary['mean_iou']:.4f}")
    print()


def is_current_defaults(params: Dict[str, Any]) -> bool:
    """Check if params match current production defaults."""
    for key, val in DEFAULTS.items():
        pval = params.get(key)
        # Handle tuple/list comparison
        if isinstance(val, tuple):
            if tuple(pval) != val:
                return False
        elif pval != val:
            return False
    return True


def _has_non_default_param(sweep_results: List[Dict], key: str) -> bool:
    """Check if any result has a non-default value for a parameter."""
    default = DEFAULTS[key]
    for r in sweep_results:
        val = r["params"].get(key)
        if isinstance(default, tuple):
            if tuple(val) if isinstance(val, list) else val != default:
                return True
        elif val != default:
            return True
    return False


def print_sweep_results(sweep_results: List[Dict[str, Any]]) -> None:
    """Print ranked sweep results table."""
    # Sort: most detections first, then lowest mean error
    def sort_key(r):
        s = r["summary"]
        err = s["mean_error_px"] if s["mean_error_px"] is not None else float("inf")
        return (-s["detected"], err)

    sweep_results.sort(key=sort_key)

    # Determine which extra columns to show (only when they vary)
    extra_cols = []
    extra_keys = [
        ("morph_dilate", "dil", 3),
        ("morph_erode", "ero", 3),
        ("area_weight", "a_wt", 4),
        ("squareness_weight", "s_wt", 4),
        ("center_weight", "c_wt", 4),
        ("squareness_min", "sq_m", 4),
        ("area_max_ratio", "a_mx", 4),
        ("contour_mode", "mode", 8),
    ]
    for key, label, width in extra_keys:
        if _has_non_default_param(sweep_results, key):
            extra_cols.append((key, label, width))

    print(f"\n{'=' * 120}")
    print(f"Parameter Sweep Results ({len(sweep_results)} combinations)")
    print(f"{'=' * 120}")

    hdr = (
        f"{'Rank':>4} | {'Detection':>9} | {'Mean Err':>8} | "
        f"{'Med Err':>7} | {'Mean IoU':>8} | "
        f"{'clip':>5} | {'tile':>4} | {'blur':>4} | "
        f"{'block':>5} | {'thr_c':>5} | {'eps':>5} | {'min_a':>5}"
    )
    for key, label, width in extra_cols:
        hdr += f" | {label:>{width}}"
    print(hdr)
    print("-" * len(hdr))

    for rank, run in enumerate(sweep_results, 1):
        s = run["summary"]
        p = run["params"]

        mean_err = f"{s['mean_error_px']:.1f}px" if s["mean_error_px"] is not None else "  N/A"
        med_err = f"{s['median_error_px']:.1f}px" if s["median_error_px"] is not None else " N/A"
        mean_iou = f"{s['mean_iou']:.4f}" if s["mean_iou"] is not None else "   N/A"

        tile = p["clahe_tile"]
        tile_val = tile[0] if isinstance(tile, (list, tuple)) else tile

        tag = " (current)" if is_current_defaults(p) else ""

        line = (
            f"{rank:>4} | {s['detection_rate']:>9} | {mean_err:>8} | "
            f"{med_err:>7} | {mean_iou:>8} | "
            f"{p['clahe_clip']:>5.1f} | {tile_val:>4} | {p['blur_k']:>4} | "
            f"{p['block_size']:>5} | {p['thresh_c']:>5} | "
            f"{p['epsilon']:>5.3f} | {p['min_area_ratio']:>5.3f}"
        )
        for key, label, width in extra_cols:
            val = p.get(key, DEFAULTS[key])
            if isinstance(val, float):
                line += f" | {val:>{width}.2f}"
            elif isinstance(val, str):
                line += f" | {val:>{width}}"
            else:
                line += f" | {val:>{width}}"
        line += tag
        print(line)

    print()


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    images: List[Dict],
    sweep_params: Dict[str, List],
) -> List[Dict[str, Any]]:
    """Run evaluation across all parameter combinations."""
    keys = list(sweep_params.keys())
    values = list(sweep_params.values())
    combinations = list(itertools.product(*values))
    total = len(combinations)

    print(f"\nSweeping {total} parameter combinations across {len(images)} images")
    print(f"Total detections: {total * len(images)}")

    results = []
    start = time.time()

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        run = evaluate_run(images, params)

        # Keep only params + summary for sweep (per_image would be too large)
        results.append({"params": run["params"], "summary": run["summary"]})

        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == total:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i + 1}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining",
                end="\r",
            )

    elapsed = time.time() - start
    print(f"\n  Completed in {elapsed:.1f}s ({total / elapsed:.1f} combos/sec)")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tuples to lists for JSON serialization."""
    out = {}
    for k, v in params.items():
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate grid detection against ground truth annotations.",
    )
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")

    parser.add_argument("--clahe-clip", type=float, nargs="+", default=None)
    parser.add_argument("--clahe-tile", type=int, nargs="+", default=None,
                        help="Tile sizes (e.g. 4 8 16 → (4,4) (8,8) (16,16))")
    parser.add_argument("--blur-k", type=int, nargs="+", default=None)
    parser.add_argument("--block-size", type=int, nargs="+", default=None)
    parser.add_argument("--thresh-c", type=int, nargs="+", default=None)
    parser.add_argument("--epsilon", type=float, nargs="+", default=None)
    parser.add_argument("--min-area-ratio", type=float, nargs="+", default=None)
    parser.add_argument("--morph-dilate", type=int, nargs="+", default=None,
                        help="Dilation kernel size (0 = skip)")
    parser.add_argument("--morph-erode", type=int, nargs="+", default=None,
                        help="Erosion kernel size (0 = skip)")
    parser.add_argument("--area-weight", type=float, nargs="+", default=None)
    parser.add_argument("--squareness-weight", type=float, nargs="+", default=None)
    parser.add_argument("--center-weight", type=float, nargs="+", default=None)
    parser.add_argument("--squareness-min", type=float, nargs="+", default=None,
                        help="Hard squareness cutoff (0 = disabled)")
    parser.add_argument("--area-max-ratio", type=float, nargs="+", default=None,
                        help="Max area ratio filter")
    parser.add_argument("--contour-mode", type=str, nargs="+", default=None,
                        choices=["external", "tree"])

    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON path (default: auto in evaluation/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    images = load_ground_truth()
    print(f"Loaded {len(images)} ground truth images")

    if args.sweep:
        sweep_params = {
            "clahe_clip": args.clahe_clip or SWEEP_DEFAULTS["clahe_clip"],
            "clahe_tile": (
                [(t, t) for t in args.clahe_tile]
                if args.clahe_tile
                else SWEEP_DEFAULTS["clahe_tile"]
            ),
            "blur_k": args.blur_k or SWEEP_DEFAULTS["blur_k"],
            "block_size": args.block_size or SWEEP_DEFAULTS["block_size"],
            "thresh_c": args.thresh_c or SWEEP_DEFAULTS["thresh_c"],
            "epsilon": args.epsilon or SWEEP_DEFAULTS["epsilon"],
            "min_area_ratio": args.min_area_ratio or SWEEP_DEFAULTS["min_area_ratio"],
            "morph_dilate": args.morph_dilate or SWEEP_DEFAULTS["morph_dilate"],
            "morph_erode": args.morph_erode or SWEEP_DEFAULTS["morph_erode"],
            "area_weight": args.area_weight or SWEEP_DEFAULTS["area_weight"],
            "squareness_weight": args.squareness_weight or SWEEP_DEFAULTS["squareness_weight"],
            "center_weight": args.center_weight or SWEEP_DEFAULTS["center_weight"],
            "squareness_min": args.squareness_min or SWEEP_DEFAULTS["squareness_min"],
            "area_max_ratio": args.area_max_ratio or SWEEP_DEFAULTS["area_max_ratio"],
            "contour_mode": args.contour_mode or SWEEP_DEFAULTS["contour_mode"],
        }

        results = run_sweep(images, sweep_params)
        print_sweep_results(results)

        output_path = args.output or str(
            PROJECT_ROOT / "evaluation" / "param_sweep_results.json"
        )
        serializable = [
            {"params": _serialize_params(r["params"]), "summary": r["summary"]}
            for r in results
        ]
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Sweep results saved to {output_path}")

    else:
        # Single run — use first value from each CLI list, or production default
        params = {
            "clahe_clip": (args.clahe_clip or [DEFAULTS["clahe_clip"]])[0],
            "clahe_tile": (
                (args.clahe_tile[0], args.clahe_tile[0])
                if args.clahe_tile
                else DEFAULTS["clahe_tile"]
            ),
            "blur_k": (args.blur_k or [DEFAULTS["blur_k"]])[0],
            "block_size": (args.block_size or [DEFAULTS["block_size"]])[0],
            "thresh_c": (args.thresh_c or [DEFAULTS["thresh_c"]])[0],
            "epsilon": (args.epsilon or [DEFAULTS["epsilon"]])[0],
            "min_area_ratio": (args.min_area_ratio or [DEFAULTS["min_area_ratio"]])[0],
            "morph_dilate": (args.morph_dilate or [DEFAULTS["morph_dilate"]])[0],
            "morph_erode": (args.morph_erode or [DEFAULTS["morph_erode"]])[0],
            "area_weight": (args.area_weight or [DEFAULTS["area_weight"]])[0],
            "squareness_weight": (args.squareness_weight or [DEFAULTS["squareness_weight"]])[0],
            "center_weight": (args.center_weight or [DEFAULTS["center_weight"]])[0],
            "squareness_min": (args.squareness_min or [DEFAULTS["squareness_min"]])[0],
            "area_max_ratio": (args.area_max_ratio or [DEFAULTS["area_max_ratio"]])[0],
            "contour_mode": (args.contour_mode or [DEFAULTS["contour_mode"]])[0],
        }

        run = evaluate_run(images, params)
        print_single_results(run)

        output_path = args.output or str(
            PROJECT_ROOT / "evaluation" / "detection_results.json"
        )
        save_run = {
            "params": _serialize_params(run["params"]),
            "summary": run["summary"],
            "per_image": run["per_image"],
        }
        with open(output_path, "w") as f:
            json.dump(save_run, f, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
