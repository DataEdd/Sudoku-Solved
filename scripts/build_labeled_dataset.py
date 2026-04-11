"""Build the 2620-image labeled Sudoku dataset for Kaggle publication.

Runs the shipped v5.1 production pipeline (`detect_grid` → 4-point
warp → CNN OCR → backtracking solver) on every image in the mexwell
`unsolved` set (the extracted copy at ``Examples/aug/``), and records
the pipeline's best-attempt labels with a deliberately simple schema:

  filename
  source                             — upstream attribution string
  has_ground_truth_benchmark         — True for the 38 GT images
  detectable                         — did detect_grid return a quad?
  detected_corners_4                 — 4-point quad (nullable)
  best_guess_grid                    — 9×9 predicted cells (nullable)
  best_guess_confidence              — 9×9 max softmax (nullable)
  best_guess_filled_count            — number of nonzero cells (nullable)
  solvable                           — does the best guess solve?
  solved_grid                        — full 9×9 solution (nullable)
  solve_time_ms                      — backtracking latency (nullable)

Unlike ``data/results_dataset/`` (which is a 38-image GT benchmark
with detection IoU, filled/empty accuracy, a hand-written failure
taxonomy, etc.), this dataset has no ground-truth-dependent fields.
For the 2582 images without annotations we cannot compute accuracy,
so we don't pretend to. The ``has_ground_truth_benchmark`` flag lets
downstream users cross-reference with ``data/results_dataset/`` when
they want the full annotations for the 38-image subset.

Output:

  data/labeled_dataset/
    data.jsonl                 — one record per image, nested fields
    data.csv                   — flat CSV mirror
    README.md                  — dataset card for Kaggle upload
    images/                    — copy of the 2620 JPEGs (gitignored)

Usage:
    python -m scripts.build_labeled_dataset
    python -m scripts.build_labeled_dataset --limit 20   # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import signal
import sys
import time
from contextlib import contextmanager
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

SOURCE_IMAGES_DIR = PROJECT_ROOT / "Examples" / "aug"
GT_PATH = PROJECT_ROOT / "evaluation" / "ground_truth.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "labeled_dataset"

# Per-image solver timeout. The shipped app.core.solver.backtracking()
# has no max-iteration bound, and pathological OCR outputs can thrash
# the solver for many seconds before hitting a contradiction. Cap at
# 2s per image so one bad record can't stall the whole run.
SOLVER_TIMEOUT_SEC = 2


class SolverTimeout(Exception):
    """Raised when the solver timeout alarm fires."""


@contextmanager
def solver_timeout(seconds: int):
    """Context manager that raises SolverTimeout after `seconds`."""

    def _handler(signum, frame):
        raise SolverTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def load_gt_filenames() -> set[str]:
    """Return the set of filenames that have ground-truth annotations."""
    if not GT_PATH.exists():
        return set()
    with GT_PATH.open() as f:
        gt_entries = json.load(f)["images"]
    return {Path(entry["path"]).name for entry in gt_entries}


def parse_index(filename: str) -> Optional[int]:
    """Extract the <index> token from a `_<index>_<hash>.jpeg` name."""
    parts = filename.split("_")
    if len(parts) < 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def process_image(
    image_path: Path,
    gt_filenames: set[str],
) -> Dict[str, Any]:
    """Run the v5.1 production pipeline on one image and package results."""
    filename = image_path.name
    record: Dict[str, Any] = {
        "filename": filename,
        "index": parse_index(filename),
        "source": "mexwell/sudoku-image-dataset",
        "has_ground_truth_benchmark": filename in gt_filenames,

        "detectable": False,
        "detected_corners_4": None,

        "best_guess_grid": None,
        "best_guess_confidence": None,
        "best_guess_filled_count": None,

        "solvable": False,
        "solved_grid": None,
        "solve_time_ms": None,
    }

    image = cv2.imread(str(image_path))
    if image is None:
        return record

    # --- Detection (same 4-point path as production /api/extract) ---
    detected_corners_arr, _ = detect_grid(image)
    if detected_corners_arr is None:
        return record

    record["detectable"] = True
    arr2d = np.asarray(detected_corners_arr).reshape(4, 2)
    record["detected_corners_4"] = [
        [float(x), float(y)] for x, y in arr2d
    ]

    # --- OCR pipeline ---
    contour = arr2d.reshape(4, 1, 2).astype(np.float32)
    warped = perspective_transform(image, contour)
    cells = extract_cells(warped)
    grid_int, conf_map = recognize_cells(cells)

    best_guess_grid = [list(row) for row in grid_int]
    best_guess_confidence = [
        [round(float(c), 4) for c in row] for row in conf_map
    ]
    filled_count = sum(
        1 for row in best_guess_grid for v in row if v != 0
    )

    record["best_guess_grid"] = best_guess_grid
    record["best_guess_confidence"] = best_guess_confidence
    record["best_guess_filled_count"] = filled_count

    # --- Solver (with per-image timeout) ---
    valid, _ = validate_puzzle(best_guess_grid)
    if valid:
        try:
            start = time.perf_counter()
            with solver_timeout(SOLVER_TIMEOUT_SEC):
                solved, _, success = backtracking(best_guess_grid)
            elapsed = (time.perf_counter() - start) * 1000.0
            if success:
                record["solvable"] = True
                record["solved_grid"] = [list(row) for row in solved]
                record["solve_time_ms"] = round(elapsed, 3)
        except SolverTimeout:
            # Treat timeouts as unsolvable. The pipeline output is
            # not a valid Sudoku in practical terms — either it's
            # wrong OCR that happens to be a hard dead-end, or it
            # would take longer than the UX budget to solve.
            record["solve_time_ms"] = SOLVER_TIMEOUT_SEC * 1000.0

    return record


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    """Flat CSV mirror; nested columns are JSON-encoded strings."""
    nested_fields = {
        "detected_corners_4",
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


def copy_images(
    records: List[Dict[str, Any]],
    source_dir: Path,
    dest: Path,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for r in records:
        src = source_dir / r["filename"]
        if src.exists():
            shutil.copy2(src, dest / r["filename"])


def write_dataset_card(
    records: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Emit the Kaggle dataset card (README.md)."""
    n_total = len(records)
    n_detectable = sum(1 for r in records if r["detectable"])
    n_solvable = sum(1 for r in records if r["solvable"])
    n_gt = sum(1 for r in records if r["has_ground_truth_benchmark"])

    filled_counts = [
        r["best_guess_filled_count"]
        for r in records
        if r["best_guess_filled_count"] is not None
    ]
    mean_filled = (
        sum(filled_counts) / len(filled_counts) if filled_counts else 0.0
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

    card = rf"""# Sudoku Pipeline Labels — mexwell/sudoku-image-dataset

**{n_total} Sudoku images from
[mexwell/sudoku-image-dataset](https://www.kaggle.com/datasets/mexwell/sudoku-image-dataset/data)
labeled with a best-attempt pass from the shipped v5.1
[Sudoku-Solved](https://github.com/DataEdd/Sudoku-Solved) pipeline.**

Each image gets:

1. **`detectable`** — did `detect_grid` return a 4-point quadrilateral?
2. **`best_guess_grid`** — the CNN's 9×9 predicted grid (or `null`)
3. **`best_guess_confidence`** — 9×9 per-cell max-softmax confidence
4. **`solvable`** — does the best-guess grid solve under MRV
   backtracking without contradiction?
5. **`solved_grid`** — the full 9×9 solution if solvable

No ground-truth columns (corners, digit grids, per-cell accuracy)
are included for these labels — they are pipeline predictions, not
hand-verified annotations. For the **{n_gt}-image ground-truth
subset** that does have full annotations (16-point corner
annotations, 9×9 digit grids, per-cell accuracy metrics, failure-
mode classification), see the companion
[Sudoku OCR GT Benchmark](https://github.com/DataEdd/Sudoku-Solved/tree/main/data/results_dataset)
inside the main repository. The `has_ground_truth_benchmark` flag
on each record here points at the subset.

This dataset is intended as a starting point for:

- Training set for alternative Sudoku OCR approaches (soft labels
  with pipeline confidence give you a larger-than-raw training
  signal than the {n_gt}-image hand-annotated benchmark alone).
- Competitor pipelines: compare your own Sudoku detector / OCR /
  solver against this one by recomputing labels on the same images
  and diffing.
- Failure-mode inspection: the `detectable=False` subset and
  `solvable=False` subset are both interesting failure buckets.

## Summary statistics

| Metric | Value |
|---|---:|
| Total images | {n_total} |
| Detectable by `detect_grid` | **{n_detectable}/{n_total}** ({100 * n_detectable / n_total:.1f}%) |
| Best-guess grid is solvable | **{n_solvable}/{n_total}** ({100 * n_solvable / n_total:.1f}%) |
| Ground-truth-benchmark subset | {n_gt}/{n_total} ({100 * n_gt / n_total:.1f}%) |
| Mean filled cells per detected image | {mean_filled:.1f} |
| Median solver latency (ms) | {median_solve:.2f} |

Note: the **solvable** count does not mean the labels are correct —
only that the pipeline's best-guess grid happens to be a valid and
completable Sudoku. Many correct-looking grids can still be wrong
at the per-cell level if OCR made offsetting errors that happen
not to contradict. The ground-truth benchmark subset is the only
way to check pipeline correctness directly.

## Files

```
sudoku_pipeline_labels/
├── README.md                 ← this dataset card
├── data.jsonl                ← {n_total} records, one per image, nested schema
├── data.csv                  ← flat CSV mirror; nested columns JSON-encoded
└── images/
    ├── _0_1018787.jpeg
    ├── _0_1436352.jpeg
    ├── ...
    └── _<last>_<hash>.jpeg   ({n_total} files total)
```

## Schema (one record per image)

| Field | Type | Description |
|---|---|---|
| `filename` | str | JPEG filename, matches `images/` entry |
| `index` | int \| null | Numeric index from the `_<index>_<hash>.jpeg` naming scheme |
| `source` | str | Upstream attribution string |
| `has_ground_truth_benchmark` | bool | True for the 38 images also in the companion GT-benchmark dataset |
| `detectable` | bool | Whether `detect_grid` returned a valid 4-point quadrilateral |
| `detected_corners_4` | list[4][2] \| null | 4-point detected quadrilateral (pixel coordinates) |
| `best_guess_grid` | list[9][9] \| null | CNN's 9×9 predicted grid — `0` means empty |
| `best_guess_confidence` | list[9][9] \| null | Per-cell max-softmax probability (classes 1–9) |
| `best_guess_filled_count` | int \| null | Number of nonzero cells in `best_guess_grid` |
| `solvable` | bool | Whether the best-guess grid solves under backtracking |
| `solved_grid` | list[9][9] \| null | Full 9×9 solution if `solvable` |
| `solve_time_ms` | float \| null | Backtracking solver latency |

## Usage example

```python
import json
import pandas as pd

df = pd.read_csv("data.csv")

# Nested columns are JSON-encoded strings; decode on demand:
df["best_guess_grid"] = df["best_guess_grid"].apply(
    lambda s: json.loads(s) if isinstance(s, str) and s else None
)

# "Which images did the pipeline fail to detect?"
undetected = df[~df["detectable"]]
print(f"{{len(undetected)}} undetected images")

# "Which images produced a solvable best guess?"
solvable = df[df["solvable"]]
print(f"{{len(solvable)}} solvable pipeline outputs")

# "Cross-reference with the GT benchmark"
gt_subset = df[df["has_ground_truth_benchmark"]]
print(f"{{len(gt_subset)}} images also in the GT benchmark")
```

Or stream the nested JSONL directly:

```python
import json

with open("data.jsonl") as f:
    records = [json.loads(line) for line in f]

for r in records[:5]:
    print(r["filename"], r["detectable"], r["solvable"])
```

## Attribution and license

- **Images:** [mexwell/sudoku-image-dataset]
  (https://www.kaggle.com/datasets/mexwell/sudoku-image-dataset/data)
  — credit mexwell as the upstream photographic source.
- **Labels:** produced by [DataEdd/Sudoku-Solved]
  (https://github.com/DataEdd/Sudoku-Solved) v5.1 shipped checkpoint
  (102K-parameter custom CNN + MRV backtracking solver).
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Use the dataset freely for research, education, commercial work, or
  derivative benchmarks — the only requirement is that you credit this
  dataset and the upstream mexwell source when you republish or extend
  it. Both the label fields (produced here) and the underlying image
  files (upstream mexwell set) are released under the same CC BY 4.0
  terms.

## Pipeline details

Labels were produced by running the shipped v5.1 Sudoku-Solved
pipeline over each image:

1. **Detection**: `detect_grid` runs a 4-step contour-based
   fallback chain (`RETR_TREE` + structure-aware scoring → morph
   close → CLAHE → second morph close). Returns the highest-scoring
   quadrilateral or `null`.
2. **Warp**: standard 4-point perspective transform on the
   detected quad (the same production path `/api/extract` uses).
3. **Cell extraction**: 9×9 grid split with a 10% interior margin
   per cell.
4. **OCR**: custom 102K-parameter CNN (3 Conv-BN-ReLU blocks,
   channels 32/64/128, dropout 0.3) trained on MNIST (digits 1–9)
   + 67-font printed digit set + Chars74K held-out fonts + synthetic
   empty-cell distribution. Confidence threshold 0.50, empty
   threshold 0.03.
5. **Solver**: MRV-ordered backtracking, single-solution.

All per-cell predictions include a confidence score so downstream
users can filter out low-quality reads. A `solvable=True` flag
means the resulting grid is a valid Sudoku puzzle (no row/column/
box contradictions) AND the backtracking solver finds a complete
solution; it does **not** guarantee the predicted cells match the
original image.

## Reproducing this dataset

From a fresh clone of the main project:

```bash
git clone https://github.com/DataEdd/Sudoku-Solved
cd Sudoku-Solved
python -m venv venv && source venv/bin/activate
pip install -r requirements-deploy.txt

# The 2620 source images live at Examples/aug/, which is gitignored.
# Obtain them from the upstream mexwell Kaggle dataset:
#   https://www.kaggle.com/datasets/mexwell/sudoku-image-dataset/data
# and unzip into Examples/aug/ before running the build.

python -m scripts.build_labeled_dataset
```

This regenerates `data/labeled_dataset/` (~25 min on MPS).
Records will be byte-equivalent to the published version modulo
solver latency, which varies by CPU.
"""
    (output_dir / "README.md").write_text(card)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the 2620-image labeled Sudoku dataset"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N images (smoke testing)",
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="Skip copying images into output dir (just write metadata)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_filenames = load_gt_filenames()
    print(f"Loaded {len(gt_filenames)} GT-benchmark filenames", flush=True)

    all_images = sorted(SOURCE_IMAGES_DIR.glob("*.jpeg"))
    if args.limit:
        all_images = all_images[: args.limit]
    print(
        f"Processing {len(all_images)} images from {SOURCE_IMAGES_DIR}",
        flush=True,
    )
    print(
        f"Solver timeout per image: {SOLVER_TIMEOUT_SEC}s (bad OCR outputs "
        "can stall the solver otherwise)",
        flush=True,
    )

    # Incremental JSONL checkpoint so partial progress survives a crash.
    jsonl_path = OUTPUT_DIR / "data.jsonl"
    jsonl_fh = jsonl_path.open("w")

    records: List[Dict[str, Any]] = []
    t_start = time.time()
    n_timeouts = 0
    for i, image_path in enumerate(all_images, 1):
        rec = process_image(image_path, gt_filenames)
        records.append(rec)
        jsonl_fh.write(json.dumps(rec) + "\n")
        if rec["solve_time_ms"] == SOLVER_TIMEOUT_SEC * 1000.0:
            n_timeouts += 1

        if i % 50 == 0 or i == len(all_images):
            jsonl_fh.flush()
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(all_images) - i) / rate if rate > 0 else 0
            det = sum(1 for r in records if r["detectable"])
            solv = sum(1 for r in records if r["solvable"])
            print(
                f"  [{i:>5d}/{len(all_images)}] "
                f"det {det:>5d}  solv {solv:>5d}  "
                f"timeouts {n_timeouts:>4d}  "
                f"{rate:>5.1f} img/s  eta {eta / 60:>5.1f}m",
                flush=True,
            )

    jsonl_fh.close()
    print(flush=True)
    print(f"Wrote {len(records)} records to {jsonl_path}", flush=True)
    write_csv(records, OUTPUT_DIR / "data.csv")
    write_dataset_card(records, OUTPUT_DIR)
    print(f"Solver timeouts (treated as unsolvable): {n_timeouts}", flush=True)

    if not args.no_images:
        print(f"Copying images to {OUTPUT_DIR / 'images'} ...")
        copy_images(records, SOURCE_IMAGES_DIR, OUTPUT_DIR / "images")
        n_copied = len(list((OUTPUT_DIR / "images").glob("*.jpeg")))
        print(f"  {n_copied} images copied")

    total_elapsed = time.time() - t_start
    print()
    print(f"Done in {total_elapsed / 60:.1f} min")
    print(f"  {OUTPUT_DIR / 'data.jsonl'}")
    print(f"  {OUTPUT_DIR / 'data.csv'}")
    print(f"  {OUTPUT_DIR / 'README.md'}")
    if not args.no_images:
        print(f"  {OUTPUT_DIR / 'images/'}")


if __name__ == "__main__":
    main()
