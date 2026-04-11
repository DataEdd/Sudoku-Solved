# Sudoku Pipeline Labels — mexwell/sudoku-image-dataset

**2620 Sudoku images from
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
hand-verified annotations. For the **38-image ground-truth
subset** that does have full annotations (16-point corner
annotations, 9×9 digit grids, per-cell accuracy metrics, failure-
mode classification), see the companion
[Sudoku OCR GT Benchmark](https://github.com/DataEdd/Sudoku-Solved/tree/main/data/results_dataset)
inside the main repository. The `has_ground_truth_benchmark` flag
on each record here points at the subset.

This dataset is intended as a starting point for:

- Training set for alternative Sudoku OCR approaches (soft labels
  with pipeline confidence give you a larger-than-raw training
  signal than the 38-image hand-annotated benchmark alone).
- Competitor pipelines: compare your own Sudoku detector / OCR /
  solver against this one by recomputing labels on the same images
  and diffing.
- Failure-mode inspection: the `detectable=False` subset and
  `solvable=False` subset are both interesting failure buckets.

## Summary statistics

| Metric | Value |
|---|---:|
| Total images | 2620 |
| Detectable by `detect_grid` | **2505/2620** (95.6%) |
| Best-guess grid is solvable | **525/2620** (20.0%) |
| Ground-truth-benchmark subset | 38/2620 (1.5%) |
| Mean filled cells per detected image | 28.5 |
| Median solver latency (ms) | 1.71 |

Note: the **solvable** count does not mean the labels are correct —
only that the pipeline's best-guess grid happens to be a valid and
completable Sudoku. Many correct-looking grids can still be wrong
at the per-cell level if OCR made offsetting errors that happen
not to contradict. The ground-truth benchmark subset is the only
way to check pipeline correctness directly.

## Files

```
sudoku_pipeline_labels/
├── README.md                        ← this dataset card
├── data.jsonl                       ← 2620 bulk records, one per image
├── data.csv                         ← flat CSV mirror; nested columns JSON-encoded
├── images/                          ← all 2620 source JPEGs
│   ├── _0_1018787.jpeg
│   ├── _0_1436352.jpeg
│   └── ...
└── ground_truth_benchmark/          ← 38-image hand-annotated benchmark
    ├── README.md                    ← GT-benchmark dataset card
    ├── data.jsonl                   ← 38 records with rich schema
    └── data.csv                     ← flat CSV mirror of the 38
```

The **`ground_truth_benchmark/`** subfolder contains the 38-image
hand-annotated validation subset with a richer schema: 16-point corner
annotations, multi-value 9×9 ground-truth grids, per-cell accuracy
metrics, detection-IoU, pixel-level corner error, and a hand-authored
per-image failure taxonomy. The image files for these 38 records are
*not* duplicated inside the subfolder — they already appear in the
parent `images/` directory and can be looked up by filename. Use the
subfolder when you want validated pipeline outputs for benchmarking
your own CV/OCR work against a known-good reference.

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
print(f"{len(undetected)} undetected images")

# "Which images produced a solvable best guess?"
solvable = df[df["solvable"]]
print(f"{len(solvable)} solvable pipeline outputs")

# "Cross-reference with the GT benchmark"
gt_subset = df[df["has_ground_truth_benchmark"]]
print(f"{len(gt_subset)} images also in the GT benchmark")
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
