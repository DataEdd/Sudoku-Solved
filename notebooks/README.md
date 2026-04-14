# Notebooks — Pipeline Deep Dives

Four notebooks, one per pipeline stage, matched one-to-one with the subsections of the main [`README.md`](../README.md)'s *How it works* section. Each notebook walks through *why* the stage was built this way — earlier iterations, lessons learned, and parameter-tuning rationale — which the main README deliberately keeps out of its compact stage descriptions.

| Notebook | Stage | Covers |
|---|---|---|
| [`01_detection.ipynb`](01_detection.ipynb) | Detection — candidate generation | Why `detect_grid` is a 4-step fallback chain. The single-pass failure on `_33_` (grid nested inside an article panel), Step 1's use of `RETR_TREE`, and the preprocessing recipes for Steps 2–4. |
| [`02_scoring.ipynb`](02_scoring.ipynb) | Scoring — candidate ranking | The 5-component structure-aware scorer used by Step 1 of detection. Deep dive into `grid_structure` (10-peak line check, spacing regularity, coverage) and `cell_count` (connected-component count vs 81). Weight-tuning rationale. |
| [`03_ocr.ipynb`](03_ocr.ipynb) | Digit recognition (OCR) | The 102K-parameter custom CNN, the training data sources (MNIST + 67 allowlist-validated Latin-digit fonts + Chars74K held-out fonts + synthetic empty cells), the train/inference preprocessing mismatch, the synthetic-vs-real-photo gap, and the ONNX deployment pragmatics. |
| [`04_solving.ipynb`](04_solving.ipynb) | Solving | MRV-ordered backtracking. The Kamal et al. (2015) comparison of backtracking vs simulated annealing vs genetic algorithms, IEEE-style pseudocode, the benchmark-caught ground-truth data errors, and the Bhattarai et al. (2025) constraint-propagation roadmap. |

## Supporting analysis

- [`ocr_analysis.ipynb`](ocr_analysis.ipynb) — per-cell OCR error taxonomy that drove the "stop tuning OCR, fix the warp" decision. Breaks filled-cell errors into *wrong digit* / *missed* / *hallucinated* buckets and tracks each bucket per-image.
- [`failure_analysis.ipynb`](failure_analysis.ipynb) — per-image decomposition of the bottom performers on the 38-image benchmark, classified by root cause (faded print, curved paper, handwritten-over-printed, crossword interference).

## Prerequisites

- The 38-image benchmark set lives at [`../Examples/Ground Example/`](../Examples/Ground%20Example/) — committed to the repo, no extra setup needed.
- All four pipeline notebooks plus `failure_analysis.ipynb` run from the inference-only dependencies (`pip install -r ../requirements-deploy.txt`).
- `03_ocr.ipynb` and `ocr_analysis.ipynb` pull in PyTorch for architecture introspection and dataset sampling — install the full development requirements (`pip install -r ../requirements.txt`).

All notebooks put the repo root on `sys.path` in their first code cell, so `from app.core import extraction` works without further setup.
