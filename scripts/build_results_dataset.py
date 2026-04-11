"""Build the enriched results dataset for Kaggle publication.

For each of the 38 ground-truth images, runs the full v5.1 production
pipeline (detect_grid + 4-point warp + CNN OCR + backtracking solver),
records the per-image results, classifies the failure mode, and emits:

  data/results_dataset/
    data.jsonl              — one record per image, nested fields
    data.csv                — flat CSV mirror, grids JSON-encoded
    README.md               — dataset card for the Kaggle upload
    images/                 — copy of the 38 source JPEGs

The resulting directory is ready for Kaggle upload via `kaggle datasets
create -p data/results_dataset/`, or as an artefact in this repo.

Usage:
    python -m scripts.build_results_dataset
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.extraction import (  # noqa: E402
    detect_grid,
    extract_cells,
    perspective_transform,
    recognize_cells,
)
from app.core.solver import backtracking  # noqa: E402
from app.core.verifier import validate_puzzle  # noqa: E402
from evaluation.evaluate_detection import compute_iou  # noqa: E402
from evaluation.evaluate_ocr import (  # noqa: E402
    gt_corners_outer,
    is_gt_filled,
    match_gt,
)

GT_PATH = PROJECT_ROOT / "evaluation" / "ground_truth.json"
IMAGES_DIR = PROJECT_ROOT / "Examples" / "Ground Example"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results_dataset"

# Explicit per-image failure categorisation from the 2026-04-11 walkthrough
# of notebooks/failure_analysis.ipynb §2 + §4.5. Images not listed here fall
# back to a data-driven categorisation (none / minor-ocr / major-ocr) in
# `classify_failure()` below.
EXPLICIT_CATEGORY: Dict[str, Dict[str, str]] = {
    "_1_2180648.jpeg": {
        "category": "undetected-invalid-gt",
        "notes": (
            "detect_grid fails — one outer corner is not recoverable by "
            "the 4-point warp. The puzzle is also filled with invalid "
            "human-entered digits (duplicate entries in rows/columns), "
            "so even a perfect read would yield an unsolvable grid."
        ),
    },
    "_11_257486.jpeg": {
        "category": "wrong-region-inner-3x3",
        "notes": (
            "detect_grid locks onto an inner 3x3 box instead of the "
            "outer 9x9 grid (structure-scoring local optimum). Cells "
            "are hand-filled imprecisely with scratches and multi-digit "
            "entries, so OCR has no clean signal even if detection "
            "returned the correct region."
        ),
    },
    "_37_8708315.jpeg": {
        "category": "wrong-region-inner-3x3",
        "notes": (
            "detect_grid locks onto an inner 3x3 box instead of the "
            "outer 9x9 grid (obstruction suspected). Same failure mode "
            "as _11_ but without the handwritten-cell complication."
        ),
    },
    "_4_3941682.jpeg": {
        "category": "wrong-object-crossword",
        "notes": (
            "detect_grid selects a crossword puzzle on the same "
            "newspaper page instead of the Sudoku. Neither the "
            "squareness/area/centeredness scoring nor the line-count "
            "structure score penalises a non-Sudoku grid object with "
            "similar 9x9 structure."
        ),
    },
    "_0_1436352.jpeg": {
        "category": "wrong-region-header-footer",
        "notes": (
            "detected quad includes the puzzle's header/footer bands, "
            "so cells near the outer border get clipped by the margin "
            "crop. Centre cells OCR correctly; border cells are "
            "unreliable due to the incorrect grid extent."
        ),
    },
    "_33_3803709.jpeg": {
        "category": "blur-extreme",
        "notes": (
            "grid detected correctly; cell contents are genuinely too "
            "blurry for any OCR to achieve 100 percent. The ground "
            "truth itself may be unverifiable on some cells. Strong "
            "detection result on a difficult source image."
        ),
    },
    "_21_3151013.jpeg": {
        "category": "undetected",
        "notes": (
            "detect_grid returns no valid quad. Low mean intensity "
            "(~68) and high variance (std ~85) suggest extreme contrast "
            "or lighting conditions that break the contour-based chain."
        ),
    },
    "_35_4619569.jpeg": {
        "category": "undetected",
        "notes": (
            "detect_grid returns no valid quad. High mean intensity "
            "(~167) suggests a very light/faded print that the contour "
            "chain could not binarise."
        ),
    },
    "_38_4338143.jpeg": {
        "category": "undetected",
        "notes": (
            "detect_grid returns no valid quad. Very high edge density "
            "(~80%) suggests visual clutter around the puzzle that "
            "confuses the contour chain."
        ),
    },
}


def classify_failure(
    detectable: bool,
    filled_accuracy: Optional[float],
    filename: str,
) -> Dict[str, str]:
    """Assign a failure category + short note to an image."""
    if filename in EXPLICIT_CATEGORY:
        return EXPLICIT_CATEGORY[filename]
    if not detectable:
        return {
            "category": "undetected",
            "notes": "detect_grid returned no valid quad (uncharacterised).",
        }
    assert filled_accuracy is not None
    if filled_accuracy >= 0.90:
        return {"category": "none", "notes": ""}
    if filled_accuracy >= 0.70:
        return {
            "category": "minor-ocr-errors",
            "notes": (
                f"detection succeeded; OCR got {filled_accuracy:.1%} "
                f"filled-cell accuracy (minor classifier errors)."
            ),
        }
    return {
        "category": "major-ocr-errors",
        "notes": (
            f"detection succeeded; OCR got {filled_accuracy:.1%} "
            f"filled-cell accuracy (systematic classifier errors, "
            f"usually upstream warp-quality issue)."
        ),
    }


def gt_solveable(grid: List[List[Any]]) -> bool:
    """Check whether the ground-truth grid itself is a valid Sudoku.

    Uses the primary-digit view of multi-value cells. Returns False
    for the three GT puzzles with duplicate-digit transcription errors
    that the solver benchmark flagged.
    """
    flat = [
        [(v[0] if isinstance(v, list) else v) for v in row]
        for row in grid
    ]
    valid, _ = validate_puzzle(flat)
    if not valid:
        return False
    _, _, success = backtracking(flat)
    return success


def process_image(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Run the full v5.1 production pipeline on one GT entry."""
    filename = Path(entry["path"]).name
    image_path = IMAGES_DIR / filename
    if not image_path.exists():
        print(f"  SKIP {filename} — file not found")
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  SKIP {filename} — cv2 failed to decode")
        return None

    idx_token = filename.split("_")[1]
    index = int(idx_token)

    gt_grid = entry["grid"]
    gt_corners = entry["corners_16"]
    gt_solvable_flag = gt_solveable(gt_grid)

    # --- Detection ----------------------------------------------------
    detected_corners_arr, _ = detect_grid(image)
    detectable = detected_corners_arr is not None

    if detectable:
        arr2d = np.asarray(detected_corners_arr).reshape(4, 2)
        detected_corners = [
            [float(x), float(y)] for x, y in arr2d
        ]
        gt_outer = gt_corners_outer(entry)
        detection_iou = round(
            float(compute_iou(detected_corners_arr, gt_outer)), 4,
        )
        detection_corner_errors = [
            round(float(np.linalg.norm(arr2d[i] - gt_outer[i])), 2)
            for i in range(4)
        ]
    else:
        detected_corners = None
        detection_iou = None
        detection_corner_errors = None

    # --- OCR pipeline -------------------------------------------------
    best_guess_grid: Optional[List[List[int]]] = None
    best_guess_confidence: Optional[List[List[float]]] = None
    filled_correct = filled_total = 0
    empty_correct = empty_total = 0
    wrong_count = missed_count = hall_count = 0

    if detectable:
        contour = np.array(
            detected_corners, dtype=np.float32
        ).reshape(4, 1, 2)
        warped = perspective_transform(image, contour)
        cells = extract_cells(warped)
        grid_int, conf_map = recognize_cells(cells)

        best_guess_grid = [list(row) for row in grid_int]
        best_guess_confidence = [
            [round(float(c), 4) for c in row] for row in conf_map
        ]

        for i in range(9):
            for j in range(9):
                gt_val = gt_grid[i][j]
                pred = grid_int[i][j]
                if is_gt_filled(gt_val):
                    filled_total += 1
                    if match_gt(pred, gt_val):
                        filled_correct += 1
                    elif pred == 0:
                        missed_count += 1
                    else:
                        wrong_count += 1
                else:
                    empty_total += 1
                    if pred == 0:
                        empty_correct += 1
                    else:
                        hall_count += 1

        filled_accuracy = (
            filled_correct / filled_total if filled_total else None
        )
        empty_accuracy = (
            empty_correct / empty_total if empty_total else None
        )
    else:
        filled_accuracy = None
        empty_accuracy = None

    # --- Solve pipeline output ---------------------------------------
    pipeline_solvable = False
    solved_grid: Optional[List[List[int]]] = None
    solve_time_ms: Optional[float] = None
    if best_guess_grid is not None:
        valid, _ = validate_puzzle(best_guess_grid)
        if valid:
            start = time.perf_counter()
            solved, _, success = backtracking(best_guess_grid)
            elapsed = (time.perf_counter() - start) * 1000.0
            if success:
                pipeline_solvable = True
                solved_grid = [list(row) for row in solved]
                solve_time_ms = round(elapsed, 3)

    # --- Failure category --------------------------------------------
    failure = classify_failure(detectable, filled_accuracy, filename)

    record: Dict[str, Any] = {
        "filename": filename,
        "index": index,
        "source": "mexwell/sudoku-image-dataset (subset)",

        "gt_corners_16": gt_corners,
        "gt_grid": gt_grid,
        "gt_solvable": gt_solvable_flag,

        "detectable": detectable,
        "detected_corners_4": detected_corners,
        "detection_iou": detection_iou,
        "detection_corner_errors_px": detection_corner_errors,

        "best_guess_grid": best_guess_grid,
        "best_guess_confidence": best_guess_confidence,
        "filled_cell_accuracy": (
            round(filled_accuracy, 4) if filled_accuracy is not None else None
        ),
        "empty_cell_accuracy": (
            round(empty_accuracy, 4) if empty_accuracy is not None else None
        ),
        "filled_cell_correct": filled_correct if detectable else None,
        "filled_cell_total": filled_total if detectable else None,
        "empty_cell_correct": empty_correct if detectable else None,
        "empty_cell_total": empty_total if detectable else None,
        "wrong_count": wrong_count if detectable else None,
        "missed_count": missed_count if detectable else None,
        "hallucinated_count": hall_count if detectable else None,

        "solvable": pipeline_solvable,
        "solved_grid": solved_grid,
        "solve_time_ms": solve_time_ms,

        "failure_category": failure["category"],
        "notes": failure["notes"],
    }
    return record


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    """Write a flat CSV mirror. Nested columns are JSON-encoded strings."""
    nested_fields = {
        "gt_corners_16",
        "gt_grid",
        "detected_corners_4",
        "detection_corner_errors_px",
        "best_guess_grid",
        "best_guess_confidence",
        "solved_grid",
    }
    if not records:
        return
    columns = list(records[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for r in records:
            row = []
            for col in columns:
                v = r[col]
                if v is None:
                    row.append("")
                elif col in nested_fields:
                    row.append(json.dumps(v))
                elif isinstance(v, bool):
                    row.append("true" if v else "false")
                else:
                    row.append(v)
            writer.writerow(row)


def copy_images(records: List[Dict[str, Any]], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for r in records:
        src = IMAGES_DIR / r["filename"]
        shutil.copy2(src, dest / r["filename"])


def write_dataset_card(
    records: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Emit README.md (aka dataset card) for the Kaggle upload."""
    n_total = len(records)
    n_detected = sum(1 for r in records if r["detectable"])
    n_solvable = sum(1 for r in records if r["solvable"])
    n_gt_solvable = sum(1 for r in records if r["gt_solvable"])
    filled_rates = [
        r["filled_cell_accuracy"]
        for r in records
        if r["filled_cell_accuracy"] is not None
    ]
    mean_filled = (
        sum(filled_rates) / len(filled_rates) if filled_rates else 0.0
    )
    solve_times = [
        r["solve_time_ms"]
        for r in records
        if r["solve_time_ms"] is not None
    ]
    median_solve = (
        sorted(solve_times)[len(solve_times) // 2]
        if solve_times
        else 0.0
    )

    categories: Dict[str, int] = {}
    for r in records:
        categories[r["failure_category"]] = (
            categories.get(r["failure_category"], 0) + 1
        )
    cat_lines = "\n".join(
        f"| `{cat}` | {count} |"
        for cat, count in sorted(
            categories.items(), key=lambda kv: -kv[1]
        )
    )

    card = rf"""# Sudoku OCR Results Dataset

38 newspaper-photograph Sudoku puzzles with **16-point corner
annotations**, **9×9 ground-truth digit grids**, and **full end-to-end
pipeline outputs** from a custom 102K-parameter CNN + backtracking
solver (v5.1 checkpoint of the
[Sudoku-Solved](https://github.com/DataEdd/Sudoku-Solved) project).

Each image carries:

1. The raw ground-truth annotation (corners + digit grid)
2. Whether the production `detect_grid` fallback chain **detects** the grid
3. The CNN's **best-guess** 9×9 grid and per-cell confidence map
4. Whether the best-guess grid is **solvable** under backtracking
5. A per-image failure-mode classification tying each wrong result to
   a concrete root cause in the pipeline

This dataset is the result of running the shipped v5.1 pipeline over
a curated 38-image subset of **[mexwell/sudoku-image-dataset]
(https://www.kaggle.com/datasets/mexwell/sudoku-image-dataset/data)**
that was selected to cover the full range of failure modes: clean
prints, faded ink, crumpled paper, crossword-adjacent grids, extreme
blur, hand-filled cells, and more. The point of publishing it is to
give other people working on Sudoku OCR/pipeline work a single
concrete benchmark they can compare their own pipeline against,
without having to re-annotate 38 images themselves.

## Summary statistics

| Metric | Value |
|---|---:|
| Images | {n_total} |
| Detected by `detect_grid` | **{n_detected}/{n_total}** ({100 * n_detected / n_total:.1f}%) |
| Pipeline output solvable | **{n_solvable}/{n_total}** ({100 * n_solvable / n_total:.1f}%) |
| Ground truth solvable | {n_gt_solvable}/{n_total} ({100 * n_gt_solvable / n_total:.1f}%) |
| Mean filled-cell accuracy (detected only) | **{100 * mean_filled:.1f}%** |
| Median solver latency | {median_solve:.2f} ms |

Three ground-truth grids ({n_total - n_gt_solvable} by count) contain
duplicate-digit transcription errors that make them unsolvable as a
CSP — they are kept in the dataset with `gt_solvable = false` as a
data-quality signal, not a pipeline bug.

## Failure-mode breakdown

| Category | Count |
|---|---:|
{cat_lines}

Category definitions:

- **none** — detection succeeded, OCR ≥90% filled-cell accuracy,
  pipeline output solves.
- **minor-ocr-errors** — detection succeeded, 70% ≤ OCR < 90%.
- **major-ocr-errors** — detection succeeded, OCR < 70% (usually
  driven by upstream warp quality, not classifier capacity).
- **undetected** — `detect_grid` returned no valid quadrilateral.
- **wrong-region-inner-3x3** — detector locked onto an inner 3×3 box
  instead of the outer 9×9 grid.
- **wrong-region-header-footer** — detected quad extends past the
  grid into page header/footer.
- **wrong-object-crossword** — detector selected a crossword puzzle
  on the same newspaper page instead of the Sudoku.
- **blur-extreme** — grid detected correctly but cell contents are
  too blurry for reliable OCR.
- **undetected-invalid-gt** — `_1_2180648` specifically: detection
  fails AND the grid is filled with invalid human-entered digits
  (duplicate entries), so even a perfect read would be unsolvable.

## Files

```
sudoku_results_dataset/
├── README.md                 ← this file (dataset card)
├── data.jsonl                ← {n_total} records, one per image, nested schema
├── data.csv                  ← flat mirror; grid/corner columns are JSON strings
└── images/
    ├── _0_1436352.jpeg
    ├── _10_970725.jpeg
    ├── ...
    └── _46_3178649.jpeg      ({n_total} files total)
```

## Schema (one record per image)

| Field | Type | Description |
|---|---|---|
| `filename` | str | JPEG filename, matches `images/` entry |
| `index` | int | Numeric index from the `_<index>_<hash>.jpeg` naming scheme |
| `source` | str | Upstream attribution string |
| `gt_corners_16` | list[16][2] | 16-point ground-truth corner annotation (pixel coordinates). Outer corners at indices `[0, 3, 15, 12]`; inner 3×3-box corners at `[5, 6, 10, 9]` |
| `gt_grid` | list[9][9] | 9×9 ground-truth digit grid. `0` means empty; entries may be a list to denote multi-value cells (ambiguous reads accepted if any match) |
| `gt_solvable` | bool | Whether the ground-truth grid itself is a valid, solvable Sudoku |
| `detectable` | bool | Whether `detect_grid` returned a valid 4-point quadrilateral |
| `detected_corners_4` | list[4][2] \| null | 4-point quadrilateral returned by `detect_grid`, or `null` if undetected |
| `detection_iou` | float \| null | IoU of the detected quad vs. the ground-truth outer quad |
| `detection_corner_errors_px` | list[4] \| null | Per-corner pixel error (Euclidean) vs. GT outer corners |
| `best_guess_grid` | list[9][9] \| null | The CNN's 9×9 predicted grid via the full production pipeline |
| `best_guess_confidence` | list[9][9] \| null | Per-cell max softmax probability (over classes 1–9) |
| `filled_cell_accuracy` | float \| null | Fraction of GT-filled cells the pipeline OCR'd correctly |
| `empty_cell_accuracy` | float \| null | Fraction of GT-empty cells the pipeline correctly left empty |
| `filled_cell_correct` / `_total` | int \| null | Raw counts for filled cells |
| `empty_cell_correct` / `_total` | int \| null | Raw counts for empty cells |
| `wrong_count` | int \| null | Number of filled cells predicted as the wrong nonzero digit |
| `missed_count` | int \| null | Number of filled cells predicted as empty |
| `hallucinated_count` | int \| null | Number of empty cells predicted as a nonzero digit |
| `solvable` | bool | Whether the pipeline's `best_guess_grid` solves under backtracking |
| `solved_grid` | list[9][9] \| null | Full 9×9 solution if `solvable`, else null |
| `solve_time_ms` | float \| null | Backtracking solver latency |
| `failure_category` | str | One of the category labels listed above |
| `notes` | str | Short human-readable description of the failure mode |

## Usage example (Python + pandas)

```python
import pandas as pd
import json

df = pd.read_csv("data.csv")

# Nested columns are JSON-encoded strings; decode on demand:
df["gt_grid"] = df["gt_grid"].apply(json.loads)
df["best_guess_grid"] = df["best_guess_grid"].apply(
    lambda s: json.loads(s) if isinstance(s, str) and s else None
)

# "How often does our pipeline output solve vs. the GT being solvable?"
print(df[["filename", "gt_solvable", "detectable", "solvable"]])

# "What's the average filled-cell accuracy on detected images?"
detected = df[df["detectable"]]
print(f"{{detected['filled_cell_accuracy'].mean():.1%}}")

# "Which images fall into each failure category?"
print(df.groupby("failure_category")["filename"].apply(list))
```

Or stream the nested JSONL directly:

```python
import json

with open("data.jsonl") as f:
    records = [json.loads(line) for line in f]

for r in records:
    print(r["filename"], r["failure_category"], r["filled_cell_accuracy"])
```

## Attribution and license

- **Images:** curated subset of
  [mexwell/sudoku-image-dataset](https://www.kaggle.com/datasets/mexwell/sudoku-image-dataset/data)
  — credit mexwell as the upstream photographic source.
- **Annotations + pipeline outputs:** produced by
  [DataEdd/Sudoku-Solved](https://github.com/DataEdd/Sudoku-Solved),
  a portfolio project implementing a Sudoku CV + CNN + solver pipeline.
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  on both the annotations (produced here) and the underlying image
  files (upstream mexwell set). Use freely for research, education,
  commercial work, or derivative benchmarks — the only requirement is
  that you credit this dataset and the upstream mexwell source when
  you republish or extend it.

## Citation

If you use this dataset in a paper, blog post, or downstream benchmark:

```bibtex
@misc{{sudoku_ocr_results_dataset,
  author = {{Joshi, Aditya (DataEdd)}},
  title  = {{Sudoku OCR Results Dataset — 38-image benchmark with
            full-pipeline annotations}},
  year   = {{2026}},
  url    = {{https://github.com/DataEdd/Sudoku-Solved}}
}}
```

## Reproduction

Every field in this dataset is reproducible from the source code. From a
fresh clone of the main project:

```bash
git clone https://github.com/DataEdd/Sudoku-Solved
cd Sudoku-Solved
python -m venv venv && source venv/bin/activate
pip install -r requirements-deploy.txt
python -m scripts.build_results_dataset
```

This regenerates `data/results_dataset/` byte-for-byte (modulo solver
latency, which varies by CPU).
"""
    (output_dir / "README.md").write_text(card)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading ground truth from {GT_PATH}")
    with GT_PATH.open() as f:
        gt_data = json.load(f)["images"]

    print(f"Processing {len(gt_data)} images...")
    records: List[Dict[str, Any]] = []
    for entry in gt_data:
        rec = process_image(entry)
        if rec is not None:
            records.append(rec)
            flag = "✓" if rec["detectable"] else "✗"
            solv = "✓" if rec["solvable"] else "✗"
            fa = (
                f"{rec['filled_cell_accuracy']:.1%}"
                if rec["filled_cell_accuracy"] is not None
                else "    -"
            )
            print(
                f"  {flag} det | {solv} solv | {fa} filled | "
                f"{rec['filename']:<26s} {rec['failure_category']}"
            )

    print()
    print(f"Writing {len(records)} records to {OUTPUT_DIR}")
    write_jsonl(records, OUTPUT_DIR / "data.jsonl")
    write_csv(records, OUTPUT_DIR / "data.csv")
    copy_images(records, OUTPUT_DIR / "images")
    write_dataset_card(records, OUTPUT_DIR)

    print("Done.")
    print(f"  {OUTPUT_DIR / 'data.jsonl'}")
    print(f"  {OUTPUT_DIR / 'data.csv'}")
    print(f"  {OUTPUT_DIR / 'README.md'}")
    print(f"  {OUTPUT_DIR / 'images/'} ({len(records)} files)")


if __name__ == "__main__":
    main()
