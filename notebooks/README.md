# Notebooks — Pipeline Deep Dives

Four notebooks, one per pipeline stage, matched one-to-one with the subsections of the main [`README.md`](../README.md)'s *How it works* section. Each notebook walks through *why* the stage was built this way — earlier iterations, lessons learned, and parameter-tuning rationale — which the main README deliberately keeps out of its compact stage descriptions.

| Notebook | Stage | Covers |
|---|---|---|
| [`01_detection.ipynb`](01_detection.ipynb) | Detection — candidate generation | Why `detect_grid` is a 4-step fallback chain. The single-pass failure on `_33_` (crossword-nested Sudoku), Step 1's use of `RETR_TREE`, and the preprocessing recipes for Steps 2–4. |
| [`02_scoring.ipynb`](02_scoring.ipynb) | Scoring — candidate ranking | The 5-component structure-aware scorer used by Step 1 of detection. Deep dive into `grid_structure` (10-peak line check, spacing regularity, coverage) and `cell_count` (connected-component count vs 81). Weight-tuning rationale. |
| [`03_ocr.ipynb`](03_ocr.ipynb) | Digit recognition (OCR) | The 102K-parameter custom CNN, the training data sources (MNIST + 67 allowlist-validated Latin-digit fonts + Chars74K held-out fonts + synthetic empty cells), the data-leakage incident that almost shipped, the train/inference preprocessing mismatch, the 99.4% synthetic vs 66.6% real-photo gap, and the ONNX deployment pragmatics. |
| [`04_solving.ipynb`](04_solving.ipynb) | Solving | MRV-ordered backtracking. The Kamal et al. (2015) comparison of backtracking vs simulated annealing vs genetic algorithms, the IEEE-style pseudocode, the benchmark-caught ground-truth data errors, and the Bhattarai et al. (2025) constraint-propagation roadmap. |

## Prerequisites

- **Detection and scoring notebooks** need the 38-image ground-truth set at `../Examples/Ground Example/` (gitignored — obtain separately).
- **OCR notebook** needs PyTorch for architecture introspection and dataset sampling (`pip install -r ../requirements.txt`).
- **Solving notebook** only needs the repo itself (no ML dependencies).

All four notebooks also need the repo's Python modules on the path, which they set up in their first code cell via `sys.path.insert(0, str(Path.cwd().resolve().parent))`.

## Related committed notebooks

- [`ocr_analysis.ipynb`](ocr_analysis.ipynb) — the per-cell OCR error taxonomy that drove the "stop tuning OCR, fix the warp" decision. Read alongside `03_ocr.ipynb` for the full filled-cell breakdown (41 wrong / 407 missed / 101 hallucinated on the 38-image GT set).
- [`wicht_ocr_evaluation.ipynb`](wicht_ocr_evaluation.ipynb) — cross-dataset evaluation of v5.1 (CNN and Tesseract) on Baptiste Wicht's V2 test set (40 real newspaper photos). Headline: 95% detection, 58.5% filled-cell accuracy, 5/40 perfect images — vs Wicht's 2014 paper reporting ~82.5% perfect-image rate in-distribution. Quantifies the training-distribution gap that motivates the OCR v2 sub-project.
- [`dataset_analysis.ipynb`](dataset_analysis.ipynb) — small living notebook for training-data exploration.

## Historical experimentation notebooks (not committed)

The project's exploratory notebooks — preprocessing sweeps, detection-strategy comparisons, SIFT experiments, early prototypes — are not committed to the public repo. They exist in the git history and can be restored locally:

```bash
# List every notebook path that has ever been tracked
git log --all --diff-filter=AM --name-only --format= -- "*.ipynb" | sort -u

# Extract a specific notebook from its most recent committed version
git show <commit>:<path> > restored.ipynb
```

The most notable historical notebooks:

| Notebook | Commit | What it contains |
|---|---|---|
| `evaluation/ensemble_detection.ipynb` | `9077aa3` | **The ancestor of `02_scoring.ipynb`.** `MaxScore` heuristic derivation, `grid_structure`, `cell_count`, multiplicative scoring. Commit message: *"MaxScore ensemble reaches 34/38."* |
| `evaluation/experiment_detection.ipynb` | `9077aa3` | Preprocessing / morph / weights sweep that fed into Steps 2–4 of the fallback chain. |
| `evaluation/compare_preprocessing.ipynb` | `e1278b0` | Early side-by-side preprocessing visualisation (superseded by the ensemble notebook). |
| `notebooks/cell_detection/` | `dd3724c` | January 2026 cell-detection learning subproject. |
| `notebooks/sudoku_detection/` | `dd3724c` | January 2026 sudoku-detection learning subproject. |

The full exploration arc runs 2026-01-10 through 2026-04-04 in the git log (`git log --all --oneline`).
