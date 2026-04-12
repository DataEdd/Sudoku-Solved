"""Build the 2620-image labeled Sudoku dataset for Kaggle publication.

Runs the shipped v5.1 production pipeline (`detect_grid` → 4-point
warp → CNN OCR → backtracking solver) on every image in the augmented
Sudoku photograph set (the extracted copy at ``Examples/aug/``), and
records the pipeline's best-attempt labels with a deliberately simple
schema:

**Provenance chain of the source images** (3 levels):

1. Canonical — ``github.com/wichtounet/sudoku_dataset`` (Baptiste
   Wicht / iCoSys research group, University of Fribourg). Original
   photograph set with per-image ground-truth annotations.
2. Proximate — ``kaggle.com/datasets/macfooty/sudoku-box-detection``
   (augmented redistribution on Kaggle; the README on that dataset
   promised annotations but never uploaded them, leaving it as
   images-only).
3. This dataset — pipeline labels produced by running the v5.1
   Sudoku-Solved pipeline over macfooty's augmented images.

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

# The 38-image GT benchmark is maintained separately at
# data/results_dataset/ by scripts/build_results_dataset.py. We mirror
# its metadata (README.md, data.jsonl, data.csv — no images, since the
# 38 JPEGs already appear in the parent images/ folder) into a
# ground_truth_benchmark/ subfolder of the labeled dataset so Kaggle
# users can browse both datasets on one page with clear separation.
GT_BENCHMARK_SOURCE = PROJECT_ROOT / "data" / "results_dataset"
GT_SUBFOLDER_NAME = "ground_truth_benchmark"

# Wicht V2 external-validation set: 40 images + per-image pipeline
# results. Built by `scripts/eval_wicht_test.py` and mirrored into a
# wicht_v2_external_validation/ subfolder of the labeled dataset for
# the Kaggle publication. Source images come from the gitignored
# `research/wichtounet_dataset/` clone of github.com/wichtounet/sudoku_dataset.
WICHT_DATASET_ROOT = PROJECT_ROOT / "research" / "wichtounet_dataset"
WICHT_IMAGES_DIR = WICHT_DATASET_ROOT / "images"
WICHT_TEST_MANIFEST = WICHT_DATASET_ROOT / "datasets" / "v2_test.desc"
WICHT_SUBFOLDER_NAME = "wicht_v2_external_validation"

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
        "source": "macfooty/sudoku-box-detection (augmented from wichtounet/sudoku_dataset)",
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


def mirror_gt_benchmark_subfolder(dest: Path) -> None:
    """Copy the GT-benchmark metadata files into a subfolder of the
    labeled dataset with an added prefix explaining the relationship.

    We do NOT copy the 38 GT image files — they already appear in the
    parent ``images/`` directory of the labeled dataset, and Kaggle
    users can cross-reference by filename. Keeps the subfolder purely
    about the enriched annotations.
    """
    if not GT_BENCHMARK_SOURCE.exists():
        print(
            f"  SKIP: {GT_BENCHMARK_SOURCE} not found — "
            f"run `python -m scripts.build_results_dataset` first"
            f" to populate the GT benchmark",
            flush=True,
        )
        return

    subfolder = dest / GT_SUBFOLDER_NAME
    subfolder.mkdir(parents=True, exist_ok=True)

    # Copy data files as-is
    for fname in ("data.jsonl", "data.csv"):
        src = GT_BENCHMARK_SOURCE / fname
        if src.exists():
            shutil.copy2(src, subfolder / fname)

    # Copy README with a prefix that explains the subfolder context
    src_readme = GT_BENCHMARK_SOURCE / "README.md"
    if src_readme.exists():
        original = src_readme.read_text()
        prefix = (
            "> **Subfolder note.** This is the **ground-truth benchmark** "
            "companion to the main [Sudoku Pipeline Labels](..) dataset. "
            "The 38 images listed here are a hand-annotated subset of the "
            "parent dataset's 2620 images — each record in this folder "
            "carries the rich schema below (16-point corner annotation, "
            "multi-value 9×9 ground truth, per-cell accuracy, hand-authored "
            "failure taxonomy) on top of the bulk `best_guess_*` labels "
            "that appear in the parent `data.jsonl`. The image files "
            "themselves live in the parent `images/` directory of this "
            "dataset — look them up by filename.\n"
            ">\n"
            "> Use this subfolder when you want **validated** pipeline "
            "outputs for benchmarking, and use the parent `data.jsonl` "
            "when you want the full 2620-image best-attempt labels.\n"
            "\n"
        )
        (subfolder / "README.md").write_text(prefix + original)


def mirror_wicht_subfolder(dest: Path) -> None:
    """Build the wicht_v2_external_validation/ subfolder with the 40
    Wicht V2 test images, per-image v5.1 pipeline results, and a
    README citing the canonical wichtounet source.

    Skips gracefully if ``research/wichtounet_dataset/`` is not cloned.
    Reuses the existing evaluation driver from ``scripts.eval_wicht_test``
    so the logic stays in one place.
    """
    if not WICHT_TEST_MANIFEST.exists():
        print(
            f"  SKIP: {WICHT_TEST_MANIFEST} not found — "
            f"run `git clone https://github.com/wichtounet/sudoku_dataset "
            f"{WICHT_DATASET_ROOT}` first to populate the Wicht subfolder",
            flush=True,
        )
        return

    from scripts.eval_wicht_test import run_evaluation  # lazy import

    subfolder = dest / WICHT_SUBFOLDER_NAME
    subfolder.mkdir(parents=True, exist_ok=True)
    images_dir = subfolder / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"  Running v5.1 pipeline on Wicht V2 test set "
        f"(this takes ~1 minute)...",
        flush=True,
    )
    run = run_evaluation(use_gt_corners=False)
    per_image = run["per_image"]
    summary = run["summary"]

    # Copy the 40 test images
    n_copied = 0
    for r in per_image:
        src = WICHT_IMAGES_DIR / r["filename"]
        if src.exists():
            shutil.copy2(src, images_dir / r["filename"])
            n_copied += 1
    print(f"  Copied {n_copied} Wicht test images", flush=True)

    # Flatten records to a simple-schema form suitable for Kaggle
    flat_records: List[Dict[str, Any]] = []
    for r in per_image:
        stats = r.get("stats", {})
        flat_records.append({
            "filename": r["filename"],
            "phone": r["phone"],
            "resolution": r["resolution"],
            "has_outline": r["has_outline"],
            "detected": r["detected"],
            "solvable": r["solvable"],
            "solve_time_ms": r["solve_time_ms"],
            "gt_grid": r["gt_grid"],
            "pred_grid": r["pred_grid"],
            "filled_total": stats.get("filled_total", 0),
            "filled_correct": stats.get("filled_correct", 0),
            "empty_total": stats.get("empty_total", 0),
            "empty_correct": stats.get("empty_correct", 0),
            "wrong_count": stats.get("wrong_count", 0),
            "missed_count": stats.get("missed_count", 0),
            "hallucinated_count": stats.get("hallucinated_count", 0),
            "perfect_image": stats.get("perfect_image", False),
        })

    # Write results.jsonl + results.csv with nested grids JSON-encoded
    jsonl_path = subfolder / "results.jsonl"
    with jsonl_path.open("w") as f:
        for rec in flat_records:
            f.write(json.dumps(rec) + "\n")

    csv_path = subfolder / "results.csv"
    nested = {"gt_grid", "pred_grid"}
    if flat_records:
        columns = list(flat_records[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(columns)
            for rec in flat_records:
                row = []
                for col in columns:
                    v = rec[col]
                    if v is None:
                        row.append("")
                    elif col in nested:
                        row.append(json.dumps(v))
                    elif isinstance(v, bool):
                        row.append("true" if v else "false")
                    else:
                        row.append(v)
                writer.writerow(row)

    # Write the subfolder README
    n = summary["images_scored"]
    detected = summary["detected"]
    solvable = summary["solvable"]
    perfect = summary["perfect_images"]
    filled_det = summary.get("filled_det_rate", 0.0)
    empty_det = summary.get("empty_det_rate", 0.0)

    sub_readme = rf"""# Wicht V2 external validation — v5.1 pipeline results

> **Subfolder note.** This is an **external-validation companion** to
> the parent [Real-World Sudoku OCR dataset](..). It contains the
> 40-image V2 test set from Baptiste Wicht's Sudoku dataset, each
> image paired with the v5.1 pipeline's full output (detection
> success, best-guess 9×9 grid, solver outcome, per-cell accuracy
> against Wicht's own ground truth). The point is to give Kaggle
> users a single-location reproduction of the external-validation
> numbers that sit in the parent dataset's card.

## What's in here

```
wicht_v2_external_validation/
├── README.md       this file
├── results.jsonl   40 records, one per image
├── results.csv     flat CSV mirror with gt_grid / pred_grid JSON-encoded
└── images/
    ├── image1005.jpg
    ├── image1009.jpg
    └── ... (40 files)
```

## Summary of v5.1 results on Wicht V2

| Metric | Value |
|---|---:|
| Total images scored | {n} |
| Detected by `detect_grid` | **{detected}/{n}** ({100 * detected / n:.1f}%) |
| Pipeline output solvable | {solvable}/{n} ({100 * solvable / n:.1f}%) |
| **Perfect images (all 81 cells correct)** | **{perfect}/{n}** ({100 * perfect / n:.1f}%) |
| Filled-cell accuracy (detected) | {100 * filled_det:.1f}% |
| Empty-cell accuracy (detected) | {100 * empty_det:.1f}% |

## Schema (one record per image)

| Field | Type | Description |
|---|---|---|
| `filename` | str | Original Wicht filename (e.g. `image1005.jpg`) |
| `phone` | str | Phone brand + model that took the picture (from the .dat metadata) |
| `resolution` | str | Pixel resolution (e.g. `1600x1200`) |
| `has_outline` | bool | True if a 4-point outline exists in Wicht's `outlines_sorted.csv` |
| `detected` | bool | Whether v5.1's `detect_grid` returned a valid 4-point quadrilateral |
| `solvable` | bool | Whether the predicted grid solves under MRV backtracking |
| `solve_time_ms` | float \| null | Backtracking solver latency |
| `gt_grid` | list[9][9] | Wicht's ground-truth 9×9 digit grid (0 = empty) |
| `pred_grid` | list[9][9] \| null | v5.1's predicted 9×9 grid |
| `filled_total` / `filled_correct` | int | Filled-cell accuracy numerator / denominator |
| `empty_total` / `empty_correct` | int | Empty-cell accuracy numerator / denominator |
| `wrong_count` | int | Number of GT-filled cells predicted as a WRONG nonzero digit |
| `missed_count` | int | Number of GT-filled cells predicted as empty (missed) |
| `hallucinated_count` | int | Number of GT-empty cells predicted as a nonzero digit |
| `perfect_image` | bool | True if ALL 81 cells are correctly predicted |

## Attribution

The **40 image files** in `images/` are redistributed from the
V2 test set of
**[wichtounet/sudoku_dataset](https://github.com/wichtounet/sudoku_dataset)**
under CC BY 4.0. Credit **Baptiste Wicht** (iCoSys research group,
University of Fribourg / HES-SO) as the canonical source. The paper
to cite is:

> Wicht, B., Hennebert, J. (2014). *Camera-based Sudoku recognition
> with Deep Belief Network.* 6th International Conference on Soft
> Computing and Pattern Recognition (SoCPaR), pp. 83-88.

The **per-image prediction results** in `results.jsonl` /
`results.csv` are produced by running the shipped v5.1
[Sudoku-Solved](https://github.com/DataEdd/Sudoku-Solved) pipeline
over the images with no Wicht-specific training — v5.1 was trained
on MNIST + synthetic fonts + Chars74K + synthetic empty cells, not
on any real newspaper photos. This is deliberately a zero-shot
cross-distribution evaluation.

**License:** CC BY 4.0 (inherited from the upstream Wicht dataset).

## The comparison

Wicht's 2014 paper reported **87.5% perfect-image rate on V1**
(160 images, 40 test). His Ph.D. thesis reports **82.5% on V2**.
v5.1 gets **{100 * perfect / n:.1f}%** on V2 with zero in-distribution
training, dominated by failures on 2007-era phones that v5.1's
training distribution doesn't cover. Full per-phone breakdown and
methodology comparison lives in the parent project's
[`README.md`](https://github.com/DataEdd/Sudoku-Solved) under the
"External validation against Wicht (2014)" section.
"""
    (subfolder / "README.md").write_text(sub_readme)
    print(
        f"  Wicht subfolder: "
        f"{n} records, {perfect} perfect, {detected} detected",
        flush=True,
    )


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

    card = rf"""# Real-World Sudoku OCR: Detection, Labels, Solved

**TL;DR** — {n_total} real-world Sudoku photographs (augmented from
[wichtounet/sudoku_dataset](https://github.com/wichtounet/sudoku_dataset)
via [macfooty/sudoku-box-detection](https://www.kaggle.com/datasets/macfooty/sudoku-box-detection))
run through a custom CV + CNN + backtracking pipeline, with the
detection quad, per-cell OCR labels, and solver output attached to
each image. **{n_detectable}/{n_total}** images detect cleanly,
**{n_solvable}/{n_total}** have a pipeline output that solves. A
**{n_gt}-image hand-annotated ground-truth subset** sits in
`ground_truth_benchmark/` for validation work (annotations were
absent from the macfooty redistribution, so we hand-authored our
own for this repo). License: CC BY 4.0.

```python
import pandas as pd
df = pd.read_csv("data.csv")
print(f"{{df['detectable'].sum()}}/{{len(df)}} detected, "
      f"{{df['solvable'].sum()}}/{{len(df)}} solvable")
```

## 📊 Summary statistics

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
not to contradict. The `ground_truth_benchmark/` subfolder is the
only way to check pipeline correctness directly.

## 📁 Files

```
.
├── README.md                        ← this dataset card
├── data.jsonl                       ← {n_total} bulk records, one per image
├── data.csv                         ← flat CSV mirror; nested columns JSON-encoded
├── images/                          ← all {n_total} source JPEGs
│   ├── _0_1018787.jpeg
│   ├── _0_1436352.jpeg
│   └── ...
└── ground_truth_benchmark/          ← {n_gt}-image hand-annotated benchmark
    ├── README.md                    ← GT-benchmark dataset card
    ├── data.jsonl                   ← {n_gt} records with rich schema
    └── data.csv                     ← flat CSV mirror of the {n_gt}
```

## 🔑 Schema (one record per image)

| Field | Type | Description |
|---|---|---|
| `filename` | str | JPEG filename, matches `images/` entry |
| `index` | int \| null | Numeric index from the `_<index>_<hash>.jpeg` naming scheme |
| `source` | str | Upstream attribution string |
| `has_ground_truth_benchmark` | bool | True for the {n_gt} images in `ground_truth_benchmark/` |
| `detectable` | bool | Whether `detect_grid` returned a valid 4-point quadrilateral |
| `detected_corners_4` | list[4][2] \| null | 4-point detected quadrilateral (pixel coordinates) |
| `best_guess_grid` | list[9][9] \| null | CNN's 9×9 predicted grid — `0` means empty |
| `best_guess_confidence` | list[9][9] \| null | Per-cell max-softmax probability (classes 1–9) |
| `best_guess_filled_count` | int \| null | Number of nonzero cells in `best_guess_grid` |
| `solvable` | bool | Whether the best-guess grid solves under backtracking |
| `solved_grid` | list[9][9] \| null | Full 9×9 solution if `solvable` |
| `solve_time_ms` | float \| null | Backtracking solver latency |

## 💻 Usage example

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

# "Cross-reference with the ground-truth benchmark"
gt_subset = df[df["has_ground_truth_benchmark"]]
print(f"{{len(gt_subset)}} images also in ground_truth_benchmark/")
```

Or stream the nested JSONL directly:

```python
import json

with open("data.jsonl") as f:
    records = [json.loads(line) for line in f]

for r in records[:5]:
    print(r["filename"], r["detectable"], r["solvable"])
```

## 🎯 Ground-truth benchmark subfolder

The **`ground_truth_benchmark/`** subfolder contains a
**{n_gt}-image hand-annotated validation subset** with a richer
schema than the bulk dataset:

- **16-point corner annotations** (not just the 4-point outer quad)
- **Multi-value 9×9 ground-truth grids** (ambiguous cells accept any
  of the listed digits)
- **Per-cell accuracy metrics** — filled/empty rates, wrong / missed
  / hallucinated cell counts
- **Detection IoU and pixel-level corner error** against the GT
- **Hand-authored failure taxonomy** + per-image notes for the
  worst-performing images

The image files for these {n_gt} records are **not** duplicated
inside the subfolder — they already appear in the parent `images/`
directory and can be looked up by filename via the
`has_ground_truth_benchmark` flag on the main `data.jsonl`.

**Use the bulk dataset** (`data.jsonl` / `data.csv` + `images/`)
when you want the full {n_total}-image best-attempt labels for
training, benchmarking, or failure-mode inspection.

**Use the GT subfolder** (`ground_truth_benchmark/data.jsonl`)
when you want validated pipeline outputs to compare your own
CV/OCR pipeline against a known-good reference.

## 📜 Attribution and license

- **Canonical image source:** [wichtounet/sudoku_dataset]
  (https://github.com/wichtounet/sudoku_dataset) — Baptiste Wicht,
  iCoSys research group at University of Fribourg. The original
  photograph set (with per-image annotations) lives there. Please
  cite the canonical source if you use these images in research.
- **Proximate source (where these specific files came from):**
  [macfooty/sudoku-box-detection]
  (https://www.kaggle.com/datasets/macfooty/sudoku-box-detection) —
  an augmented extension of the wichtounet set. macfooty's README
  promised annotations ("I will be uploading the annotations soon
  too") but never uploaded them, so this dataset inherits only the
  images from the redistribution, not the ground-truth labels.
- **Labels in this dataset:** produced by [DataEdd/Sudoku-Solved]
  (https://github.com/DataEdd/Sudoku-Solved) v5.1 shipped checkpoint
  (102K-parameter custom CNN + MRV backtracking solver). The
  38-image hand-annotated subset in `ground_truth_benchmark/` was
  authored from scratch because the macfooty redistribution did
  not carry the wichtounet annotations.
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Use the dataset freely for research, education, commercial work, or
  derivative benchmarks — the only requirement is that you credit this
  dataset AND the upstream wichtounet/macfooty sources when you
  republish or extend it. Both the label fields (produced here) and
  the underlying image files (upstream) are released under the same
  CC BY 4.0 terms.

<details>
<summary><strong>🔧 Pipeline details — click to expand</strong></summary>

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
5. **Solver**: MRV-ordered backtracking, single-solution, with a
   per-image 2-second timeout to avoid thrashing on pathological
   OCR outputs.

All per-cell predictions include a confidence score so downstream
users can filter out low-quality reads. A `solvable=True` flag
means the resulting grid is a valid Sudoku puzzle (no row/column/
box contradictions) AND the backtracking solver finds a complete
solution; it does **not** guarantee the predicted cells match the
original image.

</details>

## 🔁 Reproducing this dataset

From a fresh clone of the main project:

```bash
git clone https://github.com/DataEdd/Sudoku-Solved
cd Sudoku-Solved
python -m venv venv && source venv/bin/activate
pip install -r requirements-deploy.txt

# The {n_total} source images live at Examples/aug/, which is
# gitignored. Obtain them from the proximate macfooty Kaggle dataset:
#   https://www.kaggle.com/datasets/macfooty/sudoku-box-detection
# (or the canonical wichtounet GitHub repo:
#   https://github.com/wichtounet/sudoku_dataset)
# and unzip into Examples/aug/ before running the build.

python -m scripts.build_labeled_dataset
```

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

    print(
        f"Mirroring GT benchmark metadata into "
        f"{OUTPUT_DIR / GT_SUBFOLDER_NAME} ...",
        flush=True,
    )
    mirror_gt_benchmark_subfolder(OUTPUT_DIR)

    print(
        f"Building Wicht V2 external-validation subfolder into "
        f"{OUTPUT_DIR / WICHT_SUBFOLDER_NAME} ...",
        flush=True,
    )
    mirror_wicht_subfolder(OUTPUT_DIR)

    if not args.no_images:
        print(f"Copying images to {OUTPUT_DIR / 'images'} ...", flush=True)
        copy_images(records, SOURCE_IMAGES_DIR, OUTPUT_DIR / "images")
        n_copied = len(list((OUTPUT_DIR / "images").glob("*.jpeg")))
        print(f"  {n_copied} images copied", flush=True)

    total_elapsed = time.time() - t_start
    print()
    print(f"Done in {total_elapsed / 60:.1f} min")
    print(f"  {OUTPUT_DIR / 'data.jsonl'}")
    print(f"  {OUTPUT_DIR / 'data.csv'}")
    print(f"  {OUTPUT_DIR / 'README.md'}")
    print(f"  {OUTPUT_DIR / GT_SUBFOLDER_NAME}/ (GT benchmark mirror)")
    if not args.no_images:
        print(f"  {OUTPUT_DIR / 'images/'}")


if __name__ == "__main__":
    main()
