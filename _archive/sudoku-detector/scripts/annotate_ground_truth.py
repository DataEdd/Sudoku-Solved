#!/usr/bin/env python3
"""Tool for creating ground truth annotations for sudoku detection evaluation.

This script helps create a labeled dataset by:
1. Displaying each image
2. Running the current detector
3. Asking for human classification of the result
4. Saving annotations to a JSON file
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DetectionConfig
from src.debug import draw_corners
from src.detector import SudokuDetector


# Detection status classifications
DETECTION_STATUSES = {
    "c": "correct",
    "l": "boundary_too_large",
    "s": "boundary_too_small",
    "w": "wrong_region",
    "n": "not_detected",
    "x": "skip",  # Not a sudoku image
}

STATUS_DESCRIPTIONS = {
    "correct": "Detection is accurate",
    "boundary_too_large": "Grid detected but includes extra margin/content",
    "boundary_too_small": "Grid detected but cuts off part of the puzzle",
    "wrong_region": "Detected region is not the sudoku grid",
    "not_detected": "No grid detected but image contains a sudoku",
    "skip": "Image does not contain a sudoku puzzle",
}


def get_image_files(
    directory: str,
    extensions: tuple = (".jpg", ".jpeg", ".png"),
) -> list:
    """Recursively find all image files in directory."""
    files = []
    for ext in extensions:
        files.extend(Path(directory).rglob(f"*{ext}"))
        files.extend(Path(directory).rglob(f"*{ext.upper()}"))
    return sorted(files)


def load_existing_annotations(filepath: str) -> dict:
    """Load existing ground truth annotations."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            # Convert list to dict for easier lookup
            if isinstance(data, list):
                return {item["image_path"]: item for item in data}
            return data
    return {}


def save_annotations(annotations: dict, filepath: str):
    """Save annotations to JSON file."""
    # Convert dict back to list for consistent output
    annotation_list = list(annotations.values())
    # Sort by image path for consistent ordering
    annotation_list.sort(key=lambda x: x["image_path"])

    with open(filepath, "w") as f:
        json.dump(annotation_list, f, indent=2)


def create_display_image(
    original: np.ndarray,
    result,
    max_display_height: int = 800,
) -> np.ndarray:
    """Create a display image with detection overlay and warped result.

    Args:
        original: Original image.
        result: Detection result from SudokuDetector.
        max_display_height: Maximum height for display.

    Returns:
        Combined display image.
    """
    # Annotate original with detection
    display = original.copy()

    if result.success:
        display = draw_corners(display, result.corners, color=(0, 255, 0), thickness=3)
        status_text = f"DETECTED ({result.detection_method}, conf: {result.confidence:.2f})"
        text_color = (0, 255, 0)
    else:
        status_text = f"NOT DETECTED: {result.error_message}"
        text_color = (0, 0, 255)

    # Add status text
    cv2.putText(
        display, status_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2
    )

    # If detected, create side-by-side view with warped image
    if result.success and result.warped_image is not None:
        warped = result.warped_image

        # Resize warped to match original height
        h_orig = display.shape[0]
        scale = h_orig / warped.shape[0]
        warped_resized = cv2.resize(
            warped,
            (int(warped.shape[1] * scale), h_orig)
        )

        # Add border and label
        warped_labeled = warped_resized.copy()
        cv2.putText(
            warped_labeled, "WARPED OUTPUT", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        )

        # Combine side by side
        display = np.hstack([display, warped_labeled])

    # Resize for display if too large
    if display.shape[0] > max_display_height:
        scale = max_display_height / display.shape[0]
        display = cv2.resize(
            display,
            (int(display.shape[1] * scale), max_display_height)
        )

    return display


def print_instructions():
    """Print annotation instructions."""
    print("\n" + "=" * 60)
    print("ANNOTATION CONTROLS")
    print("=" * 60)
    for key, status in DETECTION_STATUSES.items():
        desc = STATUS_DESCRIPTIONS[status]
        print(f"  [{key}] {status:20} - {desc}")
    print(f"  [q] quit                 - Save and exit")
    print(f"  [b] back                 - Go to previous image")
    print("=" * 60 + "\n")


def annotate_images(
    input_dir: str,
    output_file: str,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    start_from: int = 0,
    skip_annotated: bool = True,
):
    """Main annotation loop.

    Args:
        input_dir: Directory containing images.
        output_file: Path to save annotations JSON.
        n_samples: Number of images to sample (None = all).
        seed: Random seed for sampling.
        start_from: Index to start from.
        skip_annotated: Skip images that already have annotations.
    """
    # Find all images
    all_files = get_image_files(input_dir)
    print(f"Found {len(all_files)} images in {input_dir}")

    if len(all_files) == 0:
        print("No images found!")
        return

    # Sample if requested
    if n_samples is not None and n_samples < len(all_files):
        if seed is not None:
            random.seed(seed)
        all_files = random.sample(all_files, n_samples)
        all_files = sorted(all_files)
        print(f"Sampled {n_samples} images (seed={seed})")

    # Load existing annotations
    annotations = load_existing_annotations(output_file)
    print(f"Loaded {len(annotations)} existing annotations")

    # Initialize detector
    detector = SudokuDetector()

    # Filter out already annotated if requested
    if skip_annotated:
        unannotated_files = [
            f for f in all_files
            if str(f.relative_to(Path(input_dir).parent)) not in annotations
        ]
        print(f"Skipping {len(all_files) - len(unannotated_files)} already annotated")
        files_to_annotate = unannotated_files
    else:
        files_to_annotate = all_files

    print_instructions()

    # Annotation loop
    idx = max(0, start_from)
    history = []  # For back navigation

    while idx < len(files_to_annotate):
        filepath = files_to_annotate[idx]
        rel_path = str(filepath.relative_to(Path(input_dir).parent))

        print(f"\n[{idx + 1}/{len(files_to_annotate)}] {filepath.name}")

        # Check if already annotated
        if rel_path in annotations and skip_annotated:
            existing = annotations[rel_path]
            print(f"  Already annotated: {existing['detection_status']}")
            idx += 1
            continue

        # Load and detect
        original = cv2.imread(str(filepath))
        if original is None:
            print(f"  Failed to load image, skipping...")
            idx += 1
            continue

        result = detector.detect(original)

        # Create and show display
        display = create_display_image(original, result)

        window_name = f"Annotate: {filepath.name}"
        cv2.imshow(window_name, display)

        # Wait for input
        while True:
            key = cv2.waitKey(0) & 0xFF
            key_char = chr(key).lower() if key < 128 else ""

            if key_char == "q":
                # Quit
                save_annotations(annotations, output_file)
                print(f"\nSaved {len(annotations)} annotations to {output_file}")
                cv2.destroyAllWindows()
                return

            elif key_char == "b":
                # Go back
                if history:
                    idx = history.pop()
                    break
                else:
                    print("  Already at first image")
                    continue

            elif key_char in DETECTION_STATUSES:
                # Valid annotation
                status = DETECTION_STATUSES[key_char]

                # Create annotation entry
                annotation = {
                    "image_path": rel_path,
                    "is_sudoku": status != "skip",
                    "detection_status": status,
                    "detector_succeeded": result.success,
                    "detector_method": result.detection_method if result.success else None,
                    "detector_confidence": float(result.confidence) if result.success else None,
                    "notes": "",
                }

                annotations[rel_path] = annotation
                print(f"  Annotated: {status}")

                # Save periodically
                if len(annotations) % 10 == 0:
                    save_annotations(annotations, output_file)
                    print(f"  (Auto-saved {len(annotations)} annotations)")

                # Move to next
                history.append(idx)
                idx += 1
                break

            else:
                print(f"  Unknown key: '{key_char}'. Press c/l/s/w/n/x or q to quit.")

        cv2.destroyWindow(window_name)

    # Final save
    save_annotations(annotations, output_file)
    print(f"\nCompleted! Saved {len(annotations)} annotations to {output_file}")
    cv2.destroyAllWindows()


def print_summary(output_file: str):
    """Print summary of existing annotations."""
    annotations = load_existing_annotations(output_file)

    if not annotations:
        print("No annotations found.")
        return

    print(f"\n{'=' * 60}")
    print(f"ANNOTATION SUMMARY: {output_file}")
    print(f"{'=' * 60}")
    print(f"Total annotations: {len(annotations)}")

    # Count by status
    status_counts = {}
    for ann in annotations.values():
        status = ann.get("detection_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    print("\nBy detection status:")
    for status, count in sorted(status_counts.items()):
        pct = 100 * count / len(annotations)
        print(f"  {status:20}: {count:4} ({pct:5.1f}%)")

    # Count detector success vs annotation
    detector_correct = sum(
        1 for ann in annotations.values()
        if ann.get("detector_succeeded") and ann.get("detection_status") == "correct"
    )
    print(f"\nDetector accuracy on labeled data: {detector_correct}/{len(annotations)} "
          f"({100*detector_correct/len(annotations):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create ground truth annotations for sudoku detection"
    )
    parser.add_argument(
        "--input", "-i", default="examples",
        help="Input directory with images"
    )
    parser.add_argument(
        "--output", "-o", default="data/ground_truth.json",
        help="Output JSON file for annotations"
    )
    parser.add_argument(
        "--samples", "-n", type=int, default=None,
        help="Number of images to sample (default: all)"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Index to start from"
    )
    parser.add_argument(
        "--include-annotated", action="store_true",
        help="Include already annotated images"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Just print summary of existing annotations"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.summary:
        print_summary(args.output)
    else:
        annotate_images(
            input_dir=args.input,
            output_file=args.output,
            n_samples=args.samples,
            seed=args.seed,
            start_from=args.start,
            skip_annotated=not args.include_annotated,
        )
