#!/usr/bin/env python3
"""Evaluate sudoku detector against ground truth annotations.

This script compares detector output with human-labeled ground truth to compute
accuracy metrics and identify failure patterns.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DetectionConfig
from src.detector import SudokuDetector

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def load_ground_truth(filepath: str) -> List[dict]:
    """Load ground truth annotations from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def evaluate_detection(
    annotation: dict,
    result,
) -> dict:
    """Evaluate a single detection against ground truth.

    Args:
        annotation: Ground truth annotation dict.
        result: Detection result from SudokuDetector.

    Returns:
        Evaluation result dict with classification and details.
    """
    status = annotation["detection_status"]
    is_sudoku = annotation.get("is_sudoku", status != "skip")
    detected = result.success

    # Determine evaluation category
    if status == "skip":
        # Not a sudoku image
        if detected:
            category = "false_positive_not_sudoku"
        else:
            category = "true_negative"
    elif status == "correct":
        # Marked as correct detection
        if detected:
            category = "true_positive"
        else:
            # Previously worked but now doesn't
            category = "regression"
    elif status == "not_detected":
        # Should be detected but wasn't
        if detected:
            # Regression fix - now detecting previously missed
            category = "improvement"
        else:
            category = "false_negative"
    elif status in ("boundary_too_large", "boundary_too_small", "wrong_region"):
        # Detected but with issues
        if detected:
            # Still detecting, but we need to verify if quality improved
            category = "quality_issue"
        else:
            # Now not detecting at all - different failure mode
            category = "quality_to_not_detected"
    else:
        category = "unknown"

    return {
        "image_path": annotation["image_path"],
        "ground_truth_status": status,
        "is_sudoku": is_sudoku,
        "detected": detected,
        "category": category,
        "detector_method": result.detection_method if detected else None,
        "detector_confidence": float(result.confidence) if detected else None,
        "error_message": result.error_message if not detected else None,
    }


def run_evaluation(
    ground_truth_file: str,
    image_base_dir: str,
    config: Optional[DetectionConfig] = None,
    save_failures: bool = False,
    output_dir: str = "evaluation_output",
) -> dict:
    """Run full evaluation against ground truth.

    Args:
        ground_truth_file: Path to ground truth JSON file.
        image_base_dir: Base directory for resolving image paths.
        config: Detection configuration (None for default).
        save_failures: Whether to save failure visualizations.
        output_dir: Directory for output files.

    Returns:
        Evaluation results dictionary.
    """
    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_file)
    print(f"Loaded {len(ground_truth)} ground truth annotations")

    # Initialize detector
    detector = SudokuDetector(config)

    # Setup output
    output_path = Path(output_dir)
    if save_failures:
        output_path.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    results = []
    iterator = tqdm(ground_truth, desc="Evaluating") if HAS_TQDM else ground_truth

    for annotation in iterator:
        image_path = Path(image_base_dir) / annotation["image_path"]

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        # Run detection
        result = detector.detect_from_file(str(image_path))

        # Evaluate
        eval_result = evaluate_detection(annotation, result)
        results.append(eval_result)

    # Compute metrics
    metrics = compute_metrics(results)

    # Add metadata
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["ground_truth_file"] = ground_truth_file
    metrics["total_annotations"] = len(ground_truth)
    metrics["total_evaluated"] = len(results)
    metrics["results"] = results

    return metrics


def compute_metrics(results: List[dict]) -> dict:
    """Compute evaluation metrics from results.

    Args:
        results: List of evaluation result dicts.

    Returns:
        Metrics dictionary.
    """
    total = len(results)
    if total == 0:
        return {"error": "No results to evaluate"}

    # Count categories
    categories = {}
    for r in results:
        cat = r["category"]
        categories[cat] = categories.get(cat, 0) + 1

    # Count by ground truth status
    by_status = {}
    for r in results:
        status = r["ground_truth_status"]
        by_status[status] = by_status.get(status, 0) + 1

    # Compute standard metrics
    # True Positive: Correctly detected sudoku with good quality
    tp = categories.get("true_positive", 0)
    # True Negative: Correctly ignored non-sudoku
    tn = categories.get("true_negative", 0)
    # False Negative: Failed to detect actual sudoku
    fn = categories.get("false_negative", 0)
    # False Positive on non-sudoku images
    fp_not_sudoku = categories.get("false_positive_not_sudoku", 0)
    # Quality issues (detected but boundaries wrong)
    quality_issues = categories.get("quality_issue", 0)
    # Regressions (previously worked, now doesn't)
    regressions = categories.get("regression", 0)
    # Improvements (previously missed, now works)
    improvements = categories.get("improvement", 0)

    # For precision/recall, treat quality issues as partial success
    # Detection accuracy = (TP + quality_issues + improvements) / (sudoku images)
    sudoku_images = total - tn - fp_not_sudoku
    detected_sudokus = tp + quality_issues + improvements

    detection_rate = detected_sudokus / sudoku_images if sudoku_images > 0 else 0

    # Perfect accuracy = TP only (no quality issues)
    perfect_accuracy = tp / sudoku_images if sudoku_images > 0 else 0

    # Standard precision/recall treating quality issues as detections
    precision = detected_sudokus / (detected_sudokus + fp_not_sudoku) if (detected_sudokus + fp_not_sudoku) > 0 else 0
    recall = detected_sudokus / (detected_sudokus + fn + regressions) if (detected_sudokus + fn + regressions) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total": total,
        "categories": categories,
        "by_ground_truth_status": by_status,
        "metrics": {
            "detection_rate": detection_rate,
            "perfect_accuracy": perfect_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "counts": {
            "true_positive": tp,
            "true_negative": tn,
            "false_negative": fn,
            "false_positive_not_sudoku": fp_not_sudoku,
            "quality_issues": quality_issues,
            "regressions": regressions,
            "improvements": improvements,
        },
    }


def print_report(metrics: dict):
    """Print a human-readable evaluation report."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    print(f"\nTotal evaluated: {metrics['total_evaluated']} / {metrics['total_annotations']}")

    print("\n--- Category Breakdown ---")
    categories = metrics.get("categories", {})
    for cat, count in sorted(categories.items()):
        pct = 100 * count / metrics["total"] if metrics["total"] > 0 else 0
        print(f"  {cat:30}: {count:4} ({pct:5.1f}%)")

    print("\n--- Ground Truth Status Breakdown ---")
    by_status = metrics.get("by_ground_truth_status", {})
    for status, count in sorted(by_status.items()):
        pct = 100 * count / metrics["total"] if metrics["total"] > 0 else 0
        print(f"  {status:25}: {count:4} ({pct:5.1f}%)")

    print("\n--- Key Metrics ---")
    m = metrics.get("metrics", {})
    print(f"  Detection Rate:    {m.get('detection_rate', 0)*100:6.2f}%  (detects sudoku, any quality)")
    print(f"  Perfect Accuracy:  {m.get('perfect_accuracy', 0)*100:6.2f}%  (correct boundary)")
    print(f"  Precision:         {m.get('precision', 0)*100:6.2f}%")
    print(f"  Recall:            {m.get('recall', 0)*100:6.2f}%")
    print(f"  F1 Score:          {m.get('f1_score', 0)*100:6.2f}%")

    print("\n--- Counts ---")
    c = metrics.get("counts", {})
    print(f"  True Positives:    {c.get('true_positive', 0):4}  (correct detections)")
    print(f"  Quality Issues:    {c.get('quality_issues', 0):4}  (detected but boundary issues)")
    print(f"  False Negatives:   {c.get('false_negative', 0):4}  (missed detections)")
    print(f"  Regressions:       {c.get('regressions', 0):4}  (previously worked, now fails)")
    print(f"  Improvements:      {c.get('improvements', 0):4}  (previously failed, now works)")
    print(f"  True Negatives:    {c.get('true_negative', 0):4}  (correctly ignored non-sudoku)")
    print(f"  FP (not sudoku):   {c.get('false_positive_not_sudoku', 0):4}  (detected non-sudoku)")

    # List failures for investigation
    results = metrics.get("results", [])
    failures = [r for r in results if r["category"] in ("false_negative", "regression", "quality_issue")]

    if failures:
        print(f"\n--- Failures to Investigate ({len(failures)}) ---")
        for f in failures[:20]:  # Show first 20
            print(f"  [{f['category']:15}] {f['image_path']}")
            if f.get("error_message"):
                print(f"                     Error: {f['error_message']}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")

    print("\n" + "=" * 70)


def list_failures(
    ground_truth_file: str,
    image_base_dir: str,
    category_filter: Optional[str] = None,
) -> List[str]:
    """Get list of failure image paths for a specific category.

    Args:
        ground_truth_file: Path to ground truth JSON.
        image_base_dir: Base directory for images.
        category_filter: Filter to specific category (e.g., "false_negative").

    Returns:
        List of image paths that are failures.
    """
    metrics = run_evaluation(ground_truth_file, image_base_dir)
    results = metrics.get("results", [])

    failure_categories = {"false_negative", "regression", "quality_issue", "quality_to_not_detected"}

    failures = []
    for r in results:
        if r["category"] in failure_categories:
            if category_filter is None or r["category"] == category_filter:
                failures.append(r["image_path"])

    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate sudoku detector against ground truth"
    )
    parser.add_argument(
        "--ground-truth", "-g",
        default="data/ground_truth.json",
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--images", "-i",
        default=".",
        help="Base directory for resolving image paths"
    )
    parser.add_argument(
        "--output", "-o",
        default="evaluation_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detailed results to JSON"
    )
    parser.add_argument(
        "--list-failures",
        action="store_true",
        help="Just list failure image paths"
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filter failures to specific category"
    )

    args = parser.parse_args()

    if args.list_failures:
        failures = list_failures(args.ground_truth, args.images, args.category)
        print(f"Found {len(failures)} failures:")
        for f in failures:
            print(f"  {f}")
    else:
        metrics = run_evaluation(
            args.ground_truth,
            args.images,
            output_dir=args.output,
        )

        print_report(metrics)

        if args.save_results:
            output_file = Path(args.output) / "evaluation_results.json"
            Path(args.output).mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                # Remove results list for summary file (too large)
                summary = {k: v for k, v in metrics.items() if k != "results"}
                json.dump(summary, f, indent=2)
            print(f"\nResults saved to {output_file}")
