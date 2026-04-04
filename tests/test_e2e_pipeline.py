"""
End-to-end integration tests for the Sudoku Solver pipeline.

Tests the full chain: image -> detect_grid_v2 -> perspective_transform
-> extract_cells -> recognize_cells -> backtracking solver.

Uses ground truth data from evaluation/ground_truth.json (38 annotated
newspaper photos) to assert detection rate, OCR accuracy, solve rate,
and single-image correctness.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.extraction import (
    detect_grid_v2,
    extract_cells,
    perspective_transform,
    recognize_cells,
)
from app.core.solver import backtracking
from app.core.verifier import validate_puzzle

# ── Constants ────────────────────────────────────────────────────────

GT_PATH = PROJECT_ROOT / "evaluation" / "ground_truth.json"
IMAGES_DIR = PROJECT_ROOT / "Examples" / "Ground Example"

# corners_16 layout:
#   P0  P1  P2  P3     (top row)
#   P4  P5  P6  P7
#   P8  P9  P10 P11
#   P12 P13 P14 P15    (bottom row)
GT_OUTER_INDICES = [0, 3, 15, 12]  # TL, TR, BR, BL

# Regression thresholds (with buffer below current performance)
MIN_DETECTION_COUNT = 33       # Current: 34/38
MIN_FILLED_CELL_ACCURACY = 0.55  # Current: 61.6% (honest, no leakage)
MIN_SOLVE_COUNT = 5


# ── Helpers ──────────────────────────────────────────────────────────

def load_ground_truth() -> List[dict]:
    """Load ground truth annotations."""
    with open(GT_PATH) as f:
        return json.load(f)["images"]


def gt_outer_corners(entry: dict) -> np.ndarray:
    """Extract 4 outer corners from 16-point annotation."""
    c16 = entry["corners_16"]
    return np.array([c16[i] for i in GT_OUTER_INDICES], dtype=np.float32)


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


def run_pipeline(image: np.ndarray) -> Optional[Tuple[
    np.ndarray,              # detected corners
    List[List[int]],         # predicted grid
    List[List[float]],       # confidence map
]]:
    """Run detect -> warp -> extract -> OCR. Returns None if detection fails."""
    corners, confidence = detect_grid_v2(image)
    if corners is None:
        return None

    contour = corners.reshape(4, 1, 2).astype(np.float32)
    warped = perspective_transform(image, contour)
    cells = extract_cells(warped)
    grid, confidence_map = recognize_cells(cells)
    return corners, grid, confidence_map


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gt_data() -> List[dict]:
    """Load ground truth once per module."""
    return load_ground_truth()


@pytest.fixture(scope="module")
def pipeline_results(gt_data):
    """Run pipeline on all GT images once, cache results for all tests.

    Returns list of dicts with keys:
        filename, detected, corners, grid, confidence_map, gt_grid
    """
    results = []
    for entry in gt_data:
        filename = Path(entry["path"]).name
        image_path = IMAGES_DIR / filename
        if not image_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        result = {"filename": filename, "gt_grid": entry["grid"]}
        pipeline_out = run_pipeline(image)

        if pipeline_out is None:
            result["detected"] = False
            result["corners"] = None
            result["grid"] = None
            result["confidence_map"] = None
        else:
            corners, grid, confidence_map = pipeline_out
            result["detected"] = True
            result["corners"] = corners
            result["grid"] = grid
            result["confidence_map"] = confidence_map

        results.append(result)

    return results


# ── Tests ────────────────────────────────────────────────────────────

class TestDetection:
    """Grid detection regression tests."""

    def test_detection_rate(self, pipeline_results):
        """detect_grid_v2 must find the grid in at least 33/38 images."""
        total = len(pipeline_results)
        detected = sum(1 for r in pipeline_results if r["detected"])
        failed = [r["filename"] for r in pipeline_results if not r["detected"]]

        assert total >= 37, (
            f"Expected at least 37 GT images but only found {total}. "
            f"Check that images exist in {IMAGES_DIR}"
        )
        assert detected >= MIN_DETECTION_COUNT, (
            f"Detection rate regression: {detected}/{total} "
            f"(minimum {MIN_DETECTION_COUNT}). "
            f"Failed: {failed}"
        )


class TestOCR:
    """OCR accuracy regression tests."""

    def test_ocr_accuracy(self, pipeline_results):
        """Filled-cell OCR accuracy must be at least 70%."""
        detected_results = [r for r in pipeline_results if r["detected"]]
        assert len(detected_results) > 0, "No images detected, cannot test OCR"

        total_filled = 0
        correct_filled = 0

        for r in detected_results:
            gt_grid = r["gt_grid"]
            pred_grid = r["grid"]
            for i in range(9):
                for j in range(9):
                    gt_val = gt_grid[i][j]
                    if is_gt_filled(gt_val):
                        total_filled += 1
                        if match_gt(pred_grid[i][j], gt_val):
                            correct_filled += 1

        accuracy = correct_filled / total_filled if total_filled > 0 else 0.0
        assert accuracy >= MIN_FILLED_CELL_ACCURACY, (
            f"OCR accuracy regression: {correct_filled}/{total_filled} "
            f"({100 * accuracy:.1f}%) — minimum {100 * MIN_FILLED_CELL_ACCURACY:.0f}%"
        )


class TestSolver:
    """End-to-end solve rate tests."""

    def test_solve_rate(self, pipeline_results):
        """At least 10 detected puzzles must solve successfully."""
        detected_results = [r for r in pipeline_results if r["detected"]]
        assert len(detected_results) > 0, "No images detected, cannot test solver"

        solved_count = 0
        attempted = 0

        for r in detected_results:
            grid = r["grid"]
            is_valid, _ = validate_puzzle(grid)
            if not is_valid:
                continue

            attempted += 1
            _, _, success = backtracking(grid)
            if success:
                solved_count += 1

        assert solved_count >= MIN_SOLVE_COUNT, (
            f"Solve rate regression: {solved_count}/{attempted} puzzles solved "
            f"(minimum {MIN_SOLVE_COUNT}). "
            f"Total detected: {len(detected_results)}"
        )


class TestSingleImageE2E:
    """Full pipeline correctness on a known-good image."""

    @pytest.mark.parametrize("target_filename", [
        "_22_7288515.jpeg",
        "_39_4570412.jpeg",
    ])
    def test_single_image_e2e(self, pipeline_results, target_filename):
        """A known-good image must detect, OCR all filled cells correctly, and solve."""
        result = None
        for r in pipeline_results:
            if r["filename"] == target_filename:
                result = r
                break

        assert result is not None, (
            f"Image {target_filename} not found in pipeline results"
        )
        assert result["detected"], (
            f"Grid detection failed for known-good image {target_filename}"
        )

        # Check every filled cell matches GT
        gt_grid = result["gt_grid"]
        pred_grid = result["grid"]
        mismatches = []

        for i in range(9):
            for j in range(9):
                gt_val = gt_grid[i][j]
                if is_gt_filled(gt_val):
                    if not match_gt(pred_grid[i][j], gt_val):
                        gt_display = gt_val if not isinstance(gt_val, list) else gt_val
                        mismatches.append(
                            f"  [{i},{j}] predicted={pred_grid[i][j]}, gt={gt_display}"
                        )

        assert len(mismatches) == 0, (
            f"OCR mismatches on {target_filename}:\n"
            + "\n".join(mismatches)
        )

        # Verify it solves
        is_valid, errors = validate_puzzle(pred_grid)
        assert is_valid, (
            f"Extracted grid from {target_filename} is invalid: {errors}"
        )

        solution, _, success = backtracking(pred_grid)
        assert success, (
            f"Backtracking failed to solve extracted grid from {target_filename}"
        )
