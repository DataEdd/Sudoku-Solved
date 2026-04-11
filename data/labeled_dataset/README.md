# Real-World Sudoku OCR: Detection, Labels, Solved

**TL;DR** — 2620 real-world Sudoku photographs (augmented from
[wichtounet/sudoku_dataset](https://github.com/wichtounet/sudoku_dataset)
via [macfooty/sudoku-box-detection](https://www.kaggle.com/datasets/macfooty/sudoku-box-detection))
run through a custom CV + CNN + backtracking pipeline, with the
detection quad, per-cell OCR labels, and solver output attached to
each image. **2505/2620** images detect cleanly,
**525/2620** have a pipeline output that solves. A
**38-image hand-annotated ground-truth subset** sits in
`ground_truth_benchmark/` for validation work (annotations were
absent from the macfooty redistribution, so we hand-authored our
own for this repo). License: CC BY 4.0.

```python
import pandas as pd
df = pd.read_csv("data.csv")
print(f"{df['detectable'].sum()}/{len(df)} detected, "
      f"{df['solvable'].sum()}/{len(df)} solvable")
```

## 📊 Summary statistics

| Metric | Value |
|---|---:|
| Total images | 2620 |
| Detectable by `detect_grid` | **2505/2620** (95.6%) |
| Best-guess grid is solvable | **525/2620** (20.0%) |
| Ground-truth-benchmark subset | 38/2620 (1.5%) |
| Mean filled cells per detected image | 28.5 |
| Median solver latency (ms) | 1.69 |

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

## 🔑 Schema (one record per image)

| Field | Type | Description |
|---|---|---|
| `filename` | str | JPEG filename, matches `images/` entry |
| `index` | int \| null | Numeric index from the `_<index>_<hash>.jpeg` naming scheme |
| `source` | str | Upstream attribution string |
| `has_ground_truth_benchmark` | bool | True for the 38 images in `ground_truth_benchmark/` |
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
print(f"{len(undetected)} undetected images")

# "Which images produced a solvable best guess?"
solvable = df[df["solvable"]]
print(f"{len(solvable)} solvable pipeline outputs")

# "Cross-reference with the ground-truth benchmark"
gt_subset = df[df["has_ground_truth_benchmark"]]
print(f"{len(gt_subset)} images also in ground_truth_benchmark/")
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
**38-image hand-annotated validation subset** with a richer
schema than the bulk dataset:

- **16-point corner annotations** (not just the 4-point outer quad)
- **Multi-value 9×9 ground-truth grids** (ambiguous cells accept any
  of the listed digits)
- **Per-cell accuracy metrics** — filled/empty rates, wrong / missed
  / hallucinated cell counts
- **Detection IoU and pixel-level corner error** against the GT
- **Hand-authored failure taxonomy** + per-image notes for the
  worst-performing images

The image files for these 38 records are **not** duplicated
inside the subfolder — they already appear in the parent `images/`
directory and can be looked up by filename via the
`has_ground_truth_benchmark` flag on the main `data.jsonl`.

**Use the bulk dataset** (`data.jsonl` / `data.csv` + `images/`)
when you want the full 2620-image best-attempt labels for
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

# The 2620 source images live at Examples/aug/, which is
# gitignored. Obtain them from the proximate macfooty Kaggle dataset:
#   https://www.kaggle.com/datasets/macfooty/sudoku-box-detection
# (or the canonical wichtounet GitHub repo:
#   https://github.com/wichtounet/sudoku_dataset)
# and unzip into Examples/aug/ before running the build.

python -m scripts.build_labeled_dataset
```

Records will be byte-equivalent to the published version modulo
solver latency, which varies by CPU.
