#!/usr/bin/env python3
"""Batch test sudoku detector on random image samples."""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DetectionConfig
from src.debug import create_side_by_side, draw_detection_result
from src.detector import SudokuDetector


def get_image_files(
    directory: str,
    extensions: tuple = (".jpg", ".jpeg", ".png"),
) -> list:
    """Recursively find all image files in directory.

    Args:
        directory: Directory to search.
        extensions: Tuple of file extensions to include.

    Returns:
        List of Path objects for found images.
    """
    files = []
    for ext in extensions:
        files.extend(Path(directory).rglob(f"*{ext}"))
        files.extend(Path(directory).rglob(f"*{ext.upper()}"))
    return files


def sample_images(
    files: list,
    n: int,
    seed: int = None,
) -> list:
    """Randomly sample n images from file list.

    Args:
        files: List of file paths.
        n: Number of images to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of sampled file paths.
    """
    if seed is not None:
        random.seed(seed)
    if n >= len(files):
        return files
    return random.sample(files, n)


def run_batch_test(
    input_dir: str,
    output_dir: str,
    n_samples: int = 50,
    seed: int = None,
    save_debug: bool = True,
    save_failures_only: bool = False,
) -> dict:
    """Run batch detection test on random sample of images.

    Args:
        input_dir: Directory containing images (e.g., 'examples').
        output_dir: Directory to save results.
        n_samples: Number of random images to test.
        seed: Random seed for reproducibility (None for random).
        save_debug: Whether to save debug visualizations.
        save_failures_only: If True, only save visualizations for failures.

    Returns:
        Dictionary containing test results and statistics.
    """
    # Find all images
    all_files = get_image_files(input_dir)
    print(f"Found {len(all_files)} images in {input_dir}")

    if len(all_files) == 0:
        print("No images found!")
        return {}

    # Sample
    sampled = sample_images(all_files, n_samples, seed)
    print(f"Testing {len(sampled)} randomly sampled images (seed={seed})")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if save_debug:
        (output_path / "success").mkdir(exist_ok=True)
        (output_path / "failure").mkdir(exist_ok=True)
        (output_path / "warped").mkdir(exist_ok=True)

    # Initialize detector
    detector = SudokuDetector()

    # Track results
    results = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "n_samples": n_samples,
        "seed": seed,
        "total": 0,
        "success": 0,
        "failure": 0,
        "success_rate": 0.0,
        "avg_confidence": 0.0,
        "method_counts": {"contour": 0, "hough": 0},
        "failures": [],
        "successes": [],
    }

    confidences = []

    for i, filepath in enumerate(sampled):
        print(f"[{i+1}/{len(sampled)}] Processing {filepath.name}...", end=" ")

        result = detector.detect_from_file(str(filepath))
        results["total"] += 1

        # Load original for visualization
        original = cv2.imread(str(filepath))

        if result.success:
            results["success"] += 1
            results["method_counts"][result.detection_method] += 1
            confidences.append(result.confidence)

            results["successes"].append({
                "file": str(filepath.name),
                "method": result.detection_method,
                "confidence": float(result.confidence),
            })

            print(f"✓ {result.detection_method} (conf: {result.confidence:.3f})")

            if save_debug and not save_failures_only:
                # Save annotated original
                annotated = draw_detection_result(original, result)
                cv2.imwrite(
                    str(output_path / "success" / f"{filepath.stem}_annotated.jpg"),
                    annotated,
                )

                # Save warped output
                cv2.imwrite(
                    str(output_path / "warped" / f"{filepath.stem}_warped.jpg"),
                    result.warped_image,
                )

                # Save side-by-side
                side_by_side = create_side_by_side(annotated, result.warped_image)
                cv2.imwrite(
                    str(output_path / "success" / f"{filepath.stem}_comparison.jpg"),
                    side_by_side,
                )
        else:
            results["failure"] += 1
            results["failures"].append({
                "file": str(filepath.name),
                "error": result.error_message,
            })

            print(f"✗ {result.error_message}")

            if save_debug:
                # Save annotated failure
                annotated = draw_detection_result(original, result)
                cv2.imwrite(
                    str(output_path / "failure" / f"{filepath.stem}_failed.jpg"),
                    annotated,
                )

    # Calculate final stats
    results["success_rate"] = (
        results["success"] / results["total"] if results["total"] > 0 else 0
    )
    results["avg_confidence"] = (
        sum(confidences) / len(confidences) if confidences else 0
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH TEST RESULTS")
    print("=" * 60)
    print(f"Total images tested: {results['total']}")
    print(f"Successful:          {results['success']}")
    print(f"Failed:              {results['failure']}")
    print(f"Success rate:        {results['success_rate']*100:.1f}%")
    print(f"Avg confidence:      {results['avg_confidence']:.3f}")
    print(
        f"Detection methods:   Contour={results['method_counts']['contour']}, "
        f"Hough={results['method_counts']['hough']}"
    )

    if results["failures"]:
        print(f"\nFailed images ({len(results['failures'])}):")
        for f in results["failures"][:10]:  # Show first 10
            print(f"  - {f['file']}: {f['error']}")
        if len(results["failures"]) > 10:
            print(f"  ... and {len(results['failures']) - 10} more")

    # Save results JSON
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch test sudoku detector on random image samples"
    )
    parser.add_argument(
        "--input", "-i", default="examples", help="Input directory with images"
    )
    parser.add_argument(
        "--output", "-o", default="test_output", help="Output directory for results"
    )
    parser.add_argument(
        "--samples", "-n", type=int, default=50, help="Number of images to sample"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-debug", action="store_true", help="Don't save debug visualizations"
    )
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Only save visualizations for failures",
    )

    args = parser.parse_args()

    run_batch_test(
        input_dir=args.input,
        output_dir=args.output,
        n_samples=args.samples,
        seed=args.seed,
        save_debug=not args.no_debug,
        save_failures_only=args.failures_only,
    )
