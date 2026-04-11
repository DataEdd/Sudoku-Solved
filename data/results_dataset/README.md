# Sudoku OCR Results Dataset

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
| Images | 38 |
| Detected by `detect_grid` | **34/38** (89.5%) |
| Pipeline output solvable | **5/38** (13.2%) |
| Ground truth solvable | 35/38 (92.1%) |
| Mean filled-cell accuracy (detected only) | **67.7%** |
| Median solver latency | 3.79 ms |

Three ground-truth grids (3 by count) contain
duplicate-digit transcription errors that make them unsolvable as a
CSP — they are kept in the dataset with `gt_solvable = false` as a
data-quality signal, not a pipeline bug.

## Failure-mode breakdown

| Category | Count |
|---|---:|
| `none` | 12 |
| `major-ocr-errors` | 9 |
| `minor-ocr-errors` | 8 |
| `undetected` | 3 |
| `wrong-region-inner-3x3` | 2 |
| `wrong-region-header-footer` | 1 |
| `undetected-invalid-gt` | 1 |
| `blur-extreme` | 1 |
| `wrong-object-crossword` | 1 |

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
├── data.jsonl                ← 38 records, one per image, nested schema
├── data.csv                  ← flat mirror; grid/corner columns are JSON strings
└── images/
    ├── _0_1436352.jpeg
    ├── _10_970725.jpeg
    ├── ...
    └── _46_3178649.jpeg      (38 files total)
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
print(f"{detected['filled_cell_accuracy'].mean():.1%}")

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
- **License:** CC BY 4.0 — use freely for research, education, or
  derivative benchmarks; please credit this dataset and the upstream
  mexwell source if you republish or extend it. If mexwell's source
  dataset uses a stricter license, that license applies to the image
  files and this dataset inherits it.

## Citation

If you use this dataset in a paper, blog post, or downstream benchmark:

```bibtex
@misc{sudoku_ocr_results_dataset,
  author = {Joshi, Aditya (DataEdd)},
  title  = {Sudoku OCR Results Dataset — 38-image benchmark with
            full-pipeline annotations},
  year   = {2026},
  url    = {https://github.com/DataEdd/Sudoku-Solved}
}
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
