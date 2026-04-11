"""
Production detector benchmark.

Runs ``app.core.extraction.detect_grid_v2`` — the deterministic 4-step
fallback chain actually used by ``/api/extract`` — against the 38 newspaper
photos in ``Examples/Ground Example/`` and compares the detected outer
corners against the annotated 16-point ground truth.

This is intentionally separate from ``evaluate_detection.py``, which wraps
a parameterised copy of the legacy single-stage detector and does NOT
exercise the production fallback chain.

Single-run only — ``detect_grid_v2`` has no tunable knobs.

Usage:
    python -m evaluation.evaluate_detection_v2
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.extraction import detect_grid_v2  # noqa: E402

GT_PATH = Path(__file__).parent / "ground_truth.json"
OUTPUT_PATH = Path(__file__).parent / "detection_v2_results.json"

# corners_16 layout (rows = 0..3, cols = 0..3):
#   P0  P1  P2  P3     (top row of intersections)
#   P4  P5  P6  P7
#   P8  P9  P10 P11
#   P12 P13 P14 P15    (bottom row of intersections)
GT_OUTER_INDICES = [0, 3, 15, 12]  # TL, TR, BR, BL


def compute_iou(quad_a: np.ndarray, quad_b: np.ndarray) -> float:
    """Convex-hull IoU of two 4-point quads, adapted from evaluate_detection.py:88."""
    a = cv2.convexHull(quad_a.reshape(-1, 1, 2).astype(np.float32))
    b = cv2.convexHull(quad_b.reshape(-1, 1, 2).astype(np.float32))

    inter_area, _ = cv2.intersectConvexConvex(a, b)
    if inter_area <= 0:
        return 0.0

    area_a = cv2.contourArea(a)
    area_b = cv2.contourArea(b)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def load_ground_truth() -> List[Dict[str, Any]]:
    with open(GT_PATH) as f:
        data = json.load(f)

    images: List[Dict[str, Any]] = []
    for entry in data["images"]:
        corners_16 = np.array(entry["corners_16"], dtype=np.float32)
        outer = corners_16[GT_OUTER_INDICES]  # TL, TR, BR, BL
        images.append(
            {
                "display_path": entry["path"],
                "abs_path": str(PROJECT_ROOT / entry["path"]),
                "gt_outer": outer,
            }
        )
    return images


def evaluate_image(img_info: Dict[str, Any]) -> Dict[str, Any]:
    image = cv2.imread(img_info["abs_path"])
    if image is None:
        return {
            "path": img_info["display_path"],
            "detected": False,
            "error": "Could not load image",
            "corner_errors_px": [None, None, None, None],
            "mean_error_px": None,
            "iou": 0.0,
            "confidence": 0.0,
        }

    detected_corners, confidence = detect_grid_v2(image)

    if detected_corners is None:
        return {
            "path": img_info["display_path"],
            "detected": False,
            "corner_errors_px": [None, None, None, None],
            "mean_error_px": None,
            "iou": 0.0,
            "confidence": round(float(confidence), 4),
        }

    gt = img_info["gt_outer"]
    corner_errors = [
        round(float(np.linalg.norm(detected_corners[i] - gt[i])), 3)
        for i in range(4)
    ]
    mean_error = round(float(np.mean(corner_errors)), 3)
    iou = round(compute_iou(detected_corners, gt), 4)

    return {
        "path": img_info["display_path"],
        "detected": True,
        "corner_errors_px": corner_errors,
        "mean_error_px": mean_error,
        "iou": iou,
        "confidence": round(float(confidence), 4),
    }


def summarize(per_image: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(per_image)
    detected = [r for r in per_image if r["detected"]]
    failed_filenames = [Path(r["path"]).name for r in per_image if not r["detected"]]

    mean_errors = [r["mean_error_px"] for r in detected if r["mean_error_px"] is not None]
    ious = [r["iou"] for r in detected]

    return {
        "total_images": total,
        "detected": len(detected),
        "detection_rate": round(len(detected) / total, 4) if total else 0.0,
        "failed_filenames": failed_filenames,
        "mean_corner_error_px": round(float(np.mean(mean_errors)), 3) if mean_errors else None,
        "median_corner_error_px": round(float(statistics.median(mean_errors)), 3) if mean_errors else None,
        "max_corner_error_px": round(float(np.max(mean_errors)), 3) if mean_errors else None,
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "median_iou": round(float(statistics.median(ious)), 4) if ious else None,
    }


def print_report(per_image: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    print()
    print("=" * 96)
    print("detect_grid_v2 — production detector vs 38-image GT")
    print("=" * 96)

    hdr = (
        f"{'Image':<40} {'Det':>3} {'TL':>7} {'TR':>7} "
        f"{'BR':>7} {'BL':>7} {'Mean':>7} {'IoU':>6} {'Conf':>5}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in per_image:
        name = Path(r["path"]).name
        if len(name) > 39:
            name = name[:36] + "..."
        if r["detected"]:
            e = r["corner_errors_px"]
            print(
                f"{name:<40} {'Y':>3} {e[0]:>7.2f} {e[1]:>7.2f} "
                f"{e[2]:>7.2f} {e[3]:>7.2f} {r['mean_error_px']:>7.2f} "
                f"{r['iou']:>6.3f} {r['confidence']:>5.2f}"
            )
        else:
            print(
                f"{name:<40} {'N':>3} {'--':>7} {'--':>7} "
                f"{'--':>7} {'--':>7} {'--':>7} {'--':>6} {'--':>5}"
            )

    print("-" * len(hdr))
    print()
    print(
        f">> Detection rate: {summary['detected']}/{summary['total_images']} "
        f"({summary['detection_rate'] * 100:.1f}%)"
    )
    if summary["failed_filenames"]:
        print(f">> Failed: {summary['failed_filenames']}")
    if summary["mean_corner_error_px"] is not None:
        print(f">> Mean corner error:   {summary['mean_corner_error_px']:.2f} px")
        print(f">> Median corner error: {summary['median_corner_error_px']:.2f} px")
        print(f">> Max  corner error:   {summary['max_corner_error_px']:.2f} px")
        print(f">> Mean IoU:            {summary['mean_iou']:.4f}")
        print(f">> Median IoU:          {summary['median_iou']:.4f}")
    print()


def main() -> None:
    images = load_ground_truth()
    print(f"Loaded {len(images)} ground-truth images")

    per_image = [evaluate_image(info) for info in images]
    summary = summarize(per_image)
    print_report(per_image, summary)

    payload = {
        "command": "python -m evaluation.evaluate_detection_v2",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": (
            "Runs the production detect_grid_v2 fallback chain from "
            "app/core/extraction.py:605 against the 38-image GT set. "
            "Detector is deterministic, so a single pass is sufficient. "
            "Any non-None return from detect_grid_v2 counts as 'detected'; "
            "IoU and per-corner Euclidean distance are computed against "
            "the 4 outer corners extracted from corners_16 at indices "
            "[0, 3, 15, 12] (TL, TR, BR, BL)."
        ),
        "summary": summary,
        "per_image": per_image,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
