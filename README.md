# Sudoku Solver

![tests](https://github.com/DataEdd/Sudoku-Solved/actions/workflows/test.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.11+-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Extract Sudoku grids from photos and solve them. Ships as a single FastAPI container: a deterministic 4-step grid-detection fallback chain, a 102K-parameter custom CNN for per-cell digit OCR, and MRV-ordered backtracking with a 0.4 ms median solve time.

## Pipeline

| 1. Input | 2. Preprocess | 3. Detect |
|:---:|:---:|:---:|
| ![Input](docs/pipeline/1_input.jpg) | ![Preprocess](docs/pipeline/2_threshold.jpg) | ![Detect](docs/pipeline/3_detected.jpg) |

| 4. Warp | 5. OCR | 6. Solve |
|:---:|:---:|:---:|
| ![Warp](docs/pipeline/4_warped.jpg) | ![OCR](docs/pipeline/5_ocr_overlay.jpg) | ![Solve](docs/pipeline/6_solved.jpg) |

## Benchmarks

Measured end-to-end against 38 hand-annotated newspaper photos with 16-point corner ground truth. Every number in this table is the output of a single `python -m evaluation.*` command — no hand-entered headline figures.

| Stage | Metric | Script |
|---|---|---|
| Grid detection | **34/38 (89.5%)** · median corner error 1.6 px · median IoU 0.99 | `evaluation/evaluate_detection.py` |
| Digit OCR (custom CNN) | **66.6%** filled-cell · **98.4%** empty-cell · **81.3%** overall | `evaluation/evaluate_ocr.py` |
| Solver (backtracking) | **0.42 ms** median · 35/38 solvable | `evaluation/benchmark_solver.py` |

The OCR figures are measured using the shipped 102K-parameter CNN checkpoint trained on MNIST + system-font-rendered printed digits (67 allowlist-validated Latin-digit fonts) + Chars74K held-out printed digits + synthetic empty cells — not a plain MNIST baseline. Full per-cell breakdown in `notebooks/ocr_analysis.ipynb`.

## How it works

The pipeline has **four stages**, organised in the order data flows through it, and each stage is matched one-to-one with a deep-dive notebook in [`notebooks/`](notebooks/). The subsections below describe *what* each stage does; the notebooks contain *why* it was built that way — earlier iterations, lessons learned, and parameter-tuning rationale.

| # | Stage | Source | Deep dive |
|---|---|---|---|
| 1 | **Detection** — candidate generation | `app/core/extraction.py` | [`01_detection.ipynb`](notebooks/01_detection.ipynb) |
| 2 | **Scoring** — candidate ranking | `app/core/extraction.py` | [`02_scoring.ipynb`](notebooks/02_scoring.ipynb) |
| 3 | **Digit recognition (OCR)** | `app/ml/` | [`03_ocr.ipynb`](notebooks/03_ocr.ipynb) |
| 4 | **Solving** | `app/core/solver.py` | [`04_solving.ipynb`](notebooks/04_solving.ipynb) |

### 1. Detection — `app/core/extraction.py`

**Role in the pipeline:** input photo → candidate quadrilaterals.

`detect_grid` is a 4-step deterministic fallback chain. Each step runs a contour-based detector against a differently-preprocessed version of the input. The first step that returns a valid quadrilateral wins; later steps only run if earlier ones failed. The chain has no tunable parameters on the production path.

```text
                    Input image (BGR)
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │ Step 1:  RETR_TREE                      │
        │          + 5-component structure score  │  ── hit ──┐
        │          (grid_structure + cell_count)  │           │
        └─────────────────────────────────────────┘           │
                   │ miss                                     │
                   ▼                                          │
        ┌─────────────────────────────────────────┐           │
        │ Step 2:  morph dilate=3, erode=5        │  ── hit ──┤
        │          RETR_EXTERNAL + standard score │           │
        └─────────────────────────────────────────┘           │
                   │ miss                                     │    first
                   ▼                                          │    success
        ┌─────────────────────────────────────────┐           │    wins
        │ Step 3:  aggressive CLAHE (clip=6, C=7) │  ── hit ──┤
        │          RETR_EXTERNAL + standard score │           │
        └─────────────────────────────────────────┘           │
                   │ miss                                     │
                   ▼                                          │
        ┌─────────────────────────────────────────┐           │
        │ Step 4:  morph dilate=3, erode=3        │  ── hit ──┤
        │          RETR_EXTERNAL + standard score │           │
        └─────────────────────────────────────────┘           │
                   │ miss                                     │
                   ▼                                          │
              (None, 0.0)                                     │
                                                              │
                                                              ▼
                                       corners [TL, TR, BR, BL] + confidence
```

Step 1 handles 29 of the 34 successful detections. It uses `cv2.RETR_TREE` to find nested contours — important when the Sudoku grid lives inside a larger enclosing shape such as a crossword block — and ranks the resulting candidate quads with the structure-aware scorer described in [§ 2 below](#2-scoring--appcoreextractionpy). Steps 2–4 are simpler `RETR_EXTERNAL` fallbacks; each runs with a different preprocessing recipe targeting a specific real-world failure mode:

| Step | Preprocessing delta vs Step 1 | Rescues images | Targets |
|---|---|---|---|
| **2** | `dilate(3×3)` then `erode(5×5)` (net thinning) | `_39_` | Fragmented contour |
| **3** | **CLAHE `clipLimit=6.0`**, threshold `C=7` | `_17_`, `_24_` | Faint print |
| **4** | `dilate(3×3)` then `erode(3×3)` (symmetric closing) | `_23_`, `_26_` | Broken lines |

After a successful detection the corners are ordered `[TL, TR, BR, BL]` via a sum/diff heuristic and refined with `cv2.cornerSubPix`. Confidence is `min(1.0, area_ratio / 0.5)` when the quad covers at least 5% of the image, else 0. On the 38-image ground-truth set the chain detects 34/38 (89.5%) with median per-corner error 1.6 px and median IoU 0.99; the four missed images (`_1_`, `_21_`, `_35_`, `_38_`) have grid edges too low-contrast for any single-pass contour detector in this family.

**Deep dive:** [`notebooks/01_detection.ipynb`](notebooks/01_detection.ipynb) — the crossword-nesting failure case, per-step visualisations, and per-image rescue breakdown.

### 2. Scoring — `app/core/extraction.py`

**Role in the pipeline:** candidate quadrilaterals → the single best candidate.

Step 1 of the detection chain can produce many convex-quad candidates on a single image — the outer Sudoku grid, any crossword block next to it, header boxes, thumbnails, ad blocks, and noise. The scoring function is what picks the Sudoku grid out of that pile. It's a 5-component weighted sum:

```
score  =  0.20 · area_norm        ← classical: relative size
       +  0.20 · squareness       ← classical: min(w,h) / max(w,h)
       +  0.10 · centeredness     ← classical: distance from image centre
       +  0.30 · grid_structure   ← structure-aware: interior line-pattern check
       +  0.20 · cell_count       ← structure-aware: interior component-count check
```

The two structure-aware components are what make Step 1 reliable on hard images. Both warp the candidate to a 200×200 square and ask *does the interior look like a 9×9 grid?* — `grid_structure` extracts horizontal and vertical lines via morphological opening, projects them to 1-D profiles, and targets **~10 peaks per axis** with regular spacing and full coverage (a 9×9 grid has 10 line positions per axis: 8 interior dividers + 2 outer borders); `cell_count` runs `cv2.connectedComponentsWithStats` on the inverted warped image and targets **~81 cell-sized components** with consistent areas. Together they let the scorer reject candidates that are *shaped* like a grid (square, centered, large area) but don't actually contain a grid interior.

**Deep dive:** [`notebooks/02_scoring.ipynb`](notebooks/02_scoring.ipynb) — each component with code demos on image `_33_`, the weight-tuning rationale, and the final showdown where the Sudoku grid beats a larger, more centered crossword block.

### 3. Digit recognition (OCR) — `app/ml/`

**Role in the pipeline:** warped grid → 9×9 integer grid.

A 102,026-parameter custom CNN maps 28×28 grayscale cell crops to 10 classes (`0` = empty, `1`–`9` = digits). The architecture is three Conv-BN-ReLU blocks followed by global pooling and a small classifier head:

```text
Input (28 × 28 × 1 grayscale cell)
  │
  ▼
Conv2d(1  → 32, kernel 3×3, pad 1) → BN → ReLU → MaxPool(2×2)    ▶ (14 × 14 × 32)
  │
  ▼
Conv2d(32 → 64, kernel 3×3, pad 1) → BN → ReLU → MaxPool(2×2)    ▶ ( 7 ×  7 × 64)
  │
  ▼
Conv2d(64 →128, kernel 3×3, pad 1) → BN → ReLU → AdaptiveAvgPool(1)  ▶ ( 1 ×  1 × 128)
  │
  ▼
Flatten → Linear(128 → 64) → ReLU → Dropout(p = 0.3) → Linear(64 → 10)
  │
  ▼
Logits (10)   →   softmax   →   predicted class + confidence
```

Trained on MNIST **labels 1-9** (handwritten digits; label 0 is dropped because Sudoku's class 0 means "empty cell", not "digit zero") + ~4,500 system-font-rendered printed digits drawn from 67 allowlist-validated Latin-digit fonts (of 371 discovered in `/System/Library/Fonts/*` — the rest are rejected by a two-stage filter: font-family allowlist then a per-font distinct-Latin-digit rendering check that catches `LastResort.otf`, `Symbol.ttf`, dingbats, and CJK/math fallbacks that would otherwise ship silent label noise; the printed-digit render uses a 56×56 → `INTER_AREA` resize pipeline so bold/condensed glyphs don't clip at the output border) with rotation/noise/blur augmentation + ~1,800 Chars74K held-out-font printed digits + 5,000 `EmptyCellDataset` variants grounded in the measured GT empty-cell distribution (mean ~56, std ~8, paper pedestal p5 ~44, with synthetic grid-line remnants and faint ink residue). Training uses `CrossEntropyLoss(weight=[2, 1, 1, ...])` to compensate for class 0's reduced pool share after the MNIST-0 drop. Best test-split accuracy **99.70%** at epoch 30; real-photo accuracy on the 38-image ground truth is **66.6% filled-cell / 98.4% empty-cell / 81.3% overall** via the production detect_grid path, rising to a measured upper bound of **~84.7% / ~98.5% / ~90.8%** with ground-truth corners and the 8-point piecewise warp (the ceiling once `extract_cells_piecewise` lands in `/api/extract`). Hallucinations (empty→digit predictions) dropped from 40 to **21** between the pre-cleanup checkpoint and the shipped one.

Deployed via ONNX Runtime — `requirements-deploy.txt` is PyTorch-free to keep the Docker image small. The `.onnx` file is a thin protobuf header that references a required `sudoku_cnn.onnx.data` sidecar holding the weight tensors:

```
app/ml/checkpoints/sudoku_cnn.onnx         4,152 B  (~ 4 KB protobuf header)
app/ml/checkpoints/sudoku_cnn.onnx.data  405,120 B  (~396 KB weight tensors, REQUIRED)
                                        ─────────
                                         409,272 B  (~400 KB total deployment footprint)
```

Loading the `.onnx` without its `.data` sidecar fails at session initialization. Both files are committed to git and both ship in the Docker image. Pytesseract is wired in as a fallback only when the CNN checkpoint is missing.

**Deep dive:** [`notebooks/03_ocr.ipynb`](notebooks/03_ocr.ipynb) covers the architecture choice, training-data composition, a train/inference preprocessing mismatch that cost 5 accuracy points, a data-leakage incident that nearly shipped an 87.8% fake result, the full filled-cell error taxonomy, and the "fix the inputs, not the model" conclusion. The per-cell deep analysis lives in [`notebooks/ocr_analysis.ipynb`](notebooks/ocr_analysis.ipynb).

### 4. Solving — `app/core/solver.py`

**Role in the pipeline:** 9×9 integer grid (with empty cells) → solved 9×9 grid.

The solver is MRV-ordered backtracking with per-cell domain restriction. At each recursive step it picks the empty cell with the fewest remaining candidates, tries each candidate in turn, and recurses. On a dead end — a cell whose candidate set becomes empty — it returns False so the parent frame can try the next value. The skeleton is the classical CSP procedure described in Kamal et al. [\[1\]](#references); the Minimum Remaining Value heuristic on top shrinks the average branching factor so that each recursion expands the most-constrained cell first.

```text
Algorithm 1  MRV-Ordered Backtracking for Sudoku

 1:  procedure Solve(G)
 2:      Input:   9 × 9 grid G with 0 denoting empty cells
 3:      Output:  True if a solution is written into G in place; False otherwise
 4:
 5:      best        ← None
 6:      best_cands  ← None
 7:
 8:      for each cell (r, c) in G do
 9:          if G[r][c] = 0 then
10:              cands ← {1, …, 9} − used_in_row(r) − used_in_col(c) − used_in_box(r, c)
11:              if cands = ∅ then
12:                  return False                                 ▷ dead end: backtrack
13:              end if
14:              if best = None or |cands| < |best_cands| then
15:                  best       ← (r, c)
16:                  best_cands ← cands
17:                  if |cands| = 1 then
18:                      break                                    ▷ forced move: stop scanning
19:                  end if
20:              end if
21:          end if
22:      end for
23:
24:      if best = None then
25:          return True                                          ▷ grid is full: puzzle solved
26:      end if
27:
28:      (r, c) ← best
29:      for each val in best_cands do
30:          G[r][c] ← val
31:          if Solve(G) then
32:              return True
33:          end if
34:          G[r][c] ← 0                                          ▷ undo assignment
35:      end for
36:      return False
37:  end procedure
```

Median latency on the 38-puzzle ground-truth set is **0.42 ms**; `python -m evaluation.benchmark_solver` reproduces the full min/median/mean/p95/max distribution from a clean checkout.

**Deep dive:** [`notebooks/04_solving.ipynb`](notebooks/04_solving.ipynb) covers the Kamal et al. Algorithm 1 baseline and why backtracking outperforms SA and genetic algorithms on 9×9 Sudoku, the MRV branching-factor argument, the three GT puzzles that fail solvability due to duplicate-digit annotation errors (caught by the benchmark harness), why the simulated-annealing solver was removed in the 2026-04-10 cleanup, and the Bhattarai et al. [\[2\]](#references) constraint-propagation roadmap for a 1.27×–2.91× follow-up speedup.

## What I learned

Every bullet below is something I got wrong first and fixed second. The project's whole history is in git — earlier iterations, abandoned approaches, and the measurements that forced each pivot.

**Start end-to-end on day one, or you'll ship the wrong number.** For weeks this project had thorough *detection* evaluation — 38 hand-annotated ground-truth images, parameter sweeps over preprocessing knobs, per-corner Euclidean error, convex-hull IoU — and **zero OCR validation**. I was citing the CNN's 99.5% MNIST test accuracy as if it were the pipeline number. The data to evaluate the OCR layer was already sitting in the ground-truth file the whole time: every entry had a 9×9 digit grid right next to its corner annotations. Nobody had bothered to write `evaluate_ocr.py` that used them. When I finally did, the first honest number was **61% filled-cell accuracy on real photos** — a 38-percentage-point gap I had zero visibility into because I'd never measured the whole system together. Component-level numbers give false confidence about the pipeline: the detector could hit 89% and the CNN could hit 99.5% on their respective test sets while the system as a whole was failing, and nothing in my dashboard would have noticed.

**"100% on clean test images, 0% on real photos."** That's a real benchmark result from an earlier iteration of the detector (it's in the 2026-04-03 commit message of the first big README rewrite). The first contour-based approach got perfect scores on clean synthetic grids but collapsed on the first batch of real newspaper photos I threw at it — crumpled paper, non-uniform lighting, headers and footers around the grid, and in one memorable case a crossword puzzle sharing the page. If I'd kept evaluating on the clean set alone I'd still be telling people the detector was perfect. The 38-image newspaper ground-truth set exists because "works on my clean test data" is a failure mode, not an accomplishment. A lot of the early legacy code that used to live in `_archive/legacy_detection/` — Hough transforms, Sobel edges, generalized Hough, border detection, line-segment detection — looked fine on synthetic inputs and fell over on the first real image.

**Self-reported confidence is a lie — I learned this twice.** An early Sobel-based detector reported 1.00 confidence on every image while getting 61% of quads right against annotated corners. I later caught the same pattern in the OCR layer: the CNN reports a mean softmax confidence of 0.88 across filled cells while being wrong on roughly 38% of them on real photos. Softmax confidence is the model's opinion of itself; it's an output of the same function whose errors you're trying to measure, so of course it doesn't tell you whether the function is correct. The only honest evaluation is comparison against externally-annotated ground truth. After the second time I stopped trusting anything a model says about its own certainty without a GT benchmark to back it up.

**The current detector is the fourth detector I wrote.** Before `detect_grid` there's a graveyard in the git history: standard Hough transform, generalized Hough transform, Sobel edge detection, line segment detection, several variants of border detection, and earlier single-pass contour pipelines I called Chains A/B/C before settling on D. I also drafted an elaborate three-stage 16-point detector (outer corners → interior line detection → intersection computation → header/footer refinement) in `.claude/responses/DETECTION_PLAN.md` that never shipped, because the simpler 4-step contour fallback chain outperformed it on the same ground-truth set. The final chain isn't an elegant first guess — it's the result of annotating 38 newspaper photos by hand, benchmarking candidate chains against them, and watching which step rescues which image. Step 1 uses `cv2.RETR_TREE` specifically because image `_33_` has the Sudoku grid nested inside a crossword block, so `RETR_EXTERNAL` was picking the crossword as the "outer" contour (IoU on that image went from 0.515 to 0.978 with the TREE + structure-scoring change alone). Step 3's aggressive CLAHE at clip=6 exists because image `_17_` has faint print that normal contrast enhancement couldn't recover. Steps 2 and 4 exist because `_23_`, `_26_`, and `_39_` had fragmented or broken contours that needed morphological closure. Every step in the chain is a "this specific image failed before" story made permanent.

**Leaked test sets flatter the model.** My first attempt at fixing the 61% OCR number was to dump the 3,078 ground-truth cells straight into the training set. The next `evaluate_ocr.py` run reported **87.8% overall and 1 hallucination** — a near-perfect jump. I almost shipped that. It was textbook data leakage: the network had seen every cell it was being graded on, and the eval script was effectively a train-accuracy report. The honest fix was to replace those cells with **4,500 rendered-font digits across 370 system typefaces** (Helvetica, Times, Courier, Arial, etc.) plus 5,000 synthetic empty-cell variants, and accept the resulting **76.3% overall / 86 hallucinations** as the real baseline — up from 57.3% / 594 hallucinations before the fix. The leaked 87.8% number would have looked better on a CV and told me nothing about generalization. Along the way I also caught a **train/inference preprocessing mismatch**: the CNN trained on grayscale MNIST tensors in `[0, 1]` continuous, but inference was applying Otsu binarization, forcing pixels to `{0, 1}`. The network was seeing a distribution it was never trained on. Matching inference to training preprocessing bumped accuracy from 57.3% to 62.2% — a free 5 points from a bug that would have been obvious if I'd printed one training sample and one inference sample side by side at any point before the cleanup.

**Look at the errors before retraining.** After the 61% number came in, my instinct was "the CNN isn't good enough, train a bigger one." Instead I wrote `notebooks/ocr_analysis.ipynb` first and classified every error into buckets: **41 wrong digits** (2.4% of filled cells — the CNN is confident and correct when it commits), **407 missed cells** (low-confidence refusals, not wrong predictions), **101 hallucinations** (empty cells confidently labeled as digits). The 407 missed cells had a median confidence of 0.58 — genuinely ambiguous inputs, not a threshold problem. A threshold sweep confirmed it: lowering the confidence threshold rescues correct digits at a **2:1 correct-to-wrong ratio**, which is net negative for puzzle solving (every "rescued" cell brings another half-cell of garbage into the solver input). Per-image the distribution is bimodal: **15 of 38 images hit ≥90% accuracy**, **9 of 38 stay below 50%**. The 9 bad ones are extreme cases — toilet-paper sudoku, a cat sitting on the puzzle, extreme motion blur, heavily faded print. The bottleneck is cell-crop quality coming out of the perspective warp, not the classifier itself. Training-data improvements (Chars74K, SVHN, francois-rozet/sudoku) would yield marginal +2–5% gains on top of the current score. The real leverage is wiring the 8-point piecewise warp into `/api/extract` so that cells from curved newsprint arrive with cleaner boundaries. That's why the Roadmap says "wire `extract_cells_piecewise`" and not "train a bigger CNN."

**Every headline number needs a runnable script.** This README has claimed, at various points, "solver runs in <1 ms median," "CNN runs 14× faster than Tesseract," "24 KB ONNX deployment," and "89% detection accuracy" without footnotes. Writing the three benchmark harnesses in `evaluation/` turned the solver claim into a real **0.42 ms measurement**, exposed the Tesseract comparison as having no underlying benchmark anywhere in the code (I removed the line rather than fabricate a number to back it up), and caught that the "24 KB" footprint is a protobuf header pointing at a required **396 KB external weights sidecar** — the real deployment footprint is ~420 KB, and loading the `.onnx` file without its `.data` sidecar fails at session initialization. The solver benchmark also surfaced a bug in my *own* ground-truth file: 3 of the 38 annotated puzzles are unsolvable because the clue grids I hand-entered contain duplicate digits in a single row, which is a CSP contradiction. That's a data-quality issue, not a solver regression, and the benchmark harness catching it is a feature rather than a bug. Every metric in the Benchmarks table at the top of this README is now tied to a `python -m evaluation.*` command that any cloner can reproduce — not to my memory, not to stale CV_NOTES, not to earlier README drafts.

## Quick start

```bash
git clone https://github.com/DataEdd/Sudoku-Solved.git
cd Sudoku-Solved
python -m venv venv && source venv/bin/activate
pip install -r requirements-deploy.txt    # inference only (ONNX Runtime, no PyTorch)
uvicorn main:app --reload
```

Visit **http://localhost:8000**. Use the camera capture to photograph a Sudoku puzzle, or click one of the six bundled sample images served from `/api/samples`.

To train the CNN from scratch, install the full development requirements instead and run the training script:

```bash
pip install -r requirements.txt
python -m app.ml.train                    # 30 epochs, ~15 min on CPU
python -m app.ml.train --epochs 10        # quick run
```

Docker deployment uses `requirements-deploy.txt` and the committed ONNX checkpoint, so no PyTorch is needed at runtime. A minimal `render.yaml` is included for Render's free-plan Docker runtime.

> **Ground-truth images.** The 38 newspaper photos the benchmark harnesses and evaluation notebooks depend on are *not* committed (the `Examples/` directory is gitignored). The web app, solver benchmark, and CNN training all work from a fresh clone without them — the detection and OCR evaluation scripts + `notebooks/01_detection.ipynb`, `notebooks/02_scoring.ipynb`, and `notebooks/ocr_analysis.ipynb` do not. See [`docs/data/README.md`](docs/data/README.md) for what's in the dataset, how to install it, and what falls back when it's missing.

**Regenerating the committed ONNX model.** If you retrain the CNN and want a drop-in replacement for the committed `app/ml/checkpoints/sudoku_cnn.onnx` (+ sidecar), run:

```bash
python -m app.ml.export_onnx --verify
```

This converts the latest `sudoku_cnn.pth` to the same external-data ONNX layout the production `CNNRecognizer` loads, and (with `--verify`) runs a PyTorch ↔ ONNX Runtime numerical parity check.

## Development

```bash
# End-to-end regression suite
pytest tests/test_e2e_pipeline.py -v

# Per-stage benchmarks against the 38-image ground-truth set
python -m evaluation.evaluate_detection      # detection rate + per-corner error + IoU
python -m evaluation.evaluate_ocr            # filled/empty cell accuracy + per-cell breakdown
python -m evaluation.benchmark_solver        # backtracking latency distribution
```

Each benchmark writes its results to `evaluation/*_results.json` so the committed numbers in the README are reproducible.

The `/debug` route serves an interactive pipeline visualizer — upload an image and see the raw input, grayscale conversion, Gaussian blur, adaptive threshold, the top-15 contour candidates, the selected quad with draggable corner handles, the warped grid, and per-cell CNN predictions with confidence scores. It's a parameterized single-pass view of the detection pipeline with tunable preprocessing knobs (blur kernel, block size, threshold constant, epsilon, cell margin, empty-cell threshold, CNN confidence threshold), intended for poking at individual failure cases. The production `/api/extract` endpoint uses the deterministic `detect_grid` fallback chain instead, which has no tunable parameters.

## Done

- [x] **4-step deterministic detection chain** — `detect_grid` with `RETR_TREE` + 5-component structure-aware scoring (`grid_structure`, `cell_count`); 34/38 on the newspaper ground-truth benchmark with median per-corner error 1.6 px and median IoU 0.99
- [x] **102K-parameter custom CNN** for per-cell digit recognition, trained on MNIST (labels 1-9 only) + 67 allowlist-validated Latin-digit system fonts + Chars74K held-out fonts + GT-grounded synthetic empty cells, with class-weighted CE loss to compensate for class 0's share of the training pool; **99.70% synthetic test accuracy** (font-disjoint Chars74K split), **66.6% filled / 98.4% empty / 81.3% overall** on real photos via detect_grid, **~84.7% / ~98.5% / ~90.8%** with ground-truth corners + piecewise warp (measured upper bound)
- [x] **Architecture ablation study** — 9-config reduced pass of `evaluation/ablation.py` at fixed `dropout=0.3`, covering the full `depth ∈ {2,3,4} × channels ∈ {small,medium,large}` diagonal under a protocol matching v5.1 production (class-weighted CE, confidence threshold 0.50, v4.2 dataset, 20 epochs per config, GT-corners real-photo eval to isolate classifier quality from detection quality). Results in `evaluation/ablation_results.json`. Headline: `d4_c-medium_drop0.3` (405,898 params, depth 4 at the middle width) tops the ranking at **80.52% real filled / 97.57% real empty** on the 38-image GT, beating the shipped production baseline `d3_c-medium_drop0.3` (102,026 params) by **+5.2 filled points** under identical data and training protocol. Depth=2 is clearly insufficient (all three depth=2 configs sit ≥19 points below the leader), and the 1.58M `d4_c-large` config is slightly overcapacity — 0.87 points below the 406K winner. Follow-up work — a fair-budget retrain of the top candidate on the production training pipeline — is the first item in the Roadmap below.
- [x] **Training → ONNX reproducibility** — `python -m app.ml.export_onnx` regenerates the committed `sudoku_cnn.onnx` + `sudoku_cnn.onnx.data` sidecar from a fresh `.pth` checkpoint, with PyTorch ↔ ONNX Runtime numerical parity verification
- [x] **MRV-ordered backtracking solver** — 0.42 ms median latency on the 38-puzzle ground-truth benchmark; three ground-truth puzzles flagged as unsolvable due to duplicate-digit annotation errors (data-quality signal, not solver regression)
- [x] **Reproducible benchmark harnesses** — `evaluation/evaluate_detection.py`, `evaluation/evaluate_ocr.py`, `evaluation/benchmark_solver.py`, each writing a committed `*_results.json`; every metric in the Benchmarks table above is reproducible from a clean checkout via a `python -m evaluation.*` command
- [x] **Interactive `/debug` pipeline visualizer** — per-stage preview, tunable preprocessing parameters, draggable corner handles on a canvas overlay
- [x] **Docker + Render deployment manifest** — `requirements-deploy.txt` is PyTorch-free; the ~400 KB ONNX model + FastAPI server ship in a single container
- [x] **GitHub Actions CI** — `.github/workflows/test.yml` runs `pytest tests/test_e2e_pipeline.py` on every push
- [x] **End-to-end regression tests** — five `tests/test_e2e_pipeline.py` cases covering detection, OCR, solve rate, and per-image correctness

## Roadmap

- [ ] **v6 architecture retrain based on the ablation winner** — the 9-config reduced ablation pass (`evaluation/ablation_results.json`) ranks `d4_c-medium_drop0.3` (405,898 params, depth 4, channels [32, 64, 128, 256], dropout 0.3) at the top of the real-photo grid with 80.52% filled-cell accuracy on GT corners, +5.2 filled points above the shipped `d3_c-medium` production baseline. Follow-up: retrain the candidate on the full production training pipeline at 30 epochs via `app/ml/train.py` (not the ablation script, which caps at 20 epochs for budget), export the checkpoint to ONNX via `app/ml/export_onnx.py`, re-run `evaluate_ocr.py` on the `detect_grid` production path to confirm the GT-corners lift translates to the end-to-end pipeline, run `tests/test_e2e_pipeline.py` as a regression gate, and if all three pass cleanly, promote as the shipped checkpoint. Inference-latency impact is the main cost to watch — 406K parameters vs the current 102K is a 4× capacity bump that translates to roughly 4× the per-cell forward-pass cost on ONNX Runtime CPU, pushing the 81-cell batch from ~40 ms toward ~160 ms, which is still well within the 500 ms responsiveness budget but worth remeasuring before promotion. Open question the retrain should also answer: whether the ablation's +5.2 point lift is an artefact of the 20-epoch budget reduction (larger models may simply benefit more from longer training) or a real architectural improvement that holds under the production 30-epoch schedule.
- [ ] Extend the ablation to the full 27-config grid by running the remaining `dropout ∈ {0.2, 0.5}` rows (18 additional configs, ~90 min on MPS via `python -m evaluation.ablation` without the `--dropout-only` flag). The reduced pass pinned `dropout=0.3`; the full grid would empirically test whether 0.3 is actually optimal for the winning d4_c-medium family, or whether the dropout rate should shift once depth and width are known.
- [ ] Wire `extract_cells_piecewise` (`app/core/extraction.py`) into `/api/extract` so curved newspaper pages use the 8-point piecewise warp. Currently it's only used in `evaluate_ocr.py --piecewise` and in the annotation preview. Measured upper bound on the 38-image GT is +6.2 filled-cell points over the production 4-point path, and the per-image failure analysis (`notebooks/failure_analysis.ipynb`) shows this is the dominant lever for the worst-performing images (`_1_2180648`, `_11_257486`, `_37_8708315`, partially `_0_1436352`).
- [ ] **Grid-extent validation in `_find_best_quad_structured`** — add a post-warp check that downranks candidate quads whose outer strips show header/footer ink density or whose interior line count deviates from the expected ~10-per-axis Sudoku structure. Addresses the `_0_1436352` failure mode where the current scoring picks a quad that extends past the grid into the page header.
- [ ] **Sudoku-vs-crossword disambiguation** — Sudokus have sparse ink (mostly blank cells), crosswords have dense ink (black blocks). Add a per-candidate black-fill ratio signal to `_find_best_quad_structured` with a ceiling around 40%, tuned carefully so near-completed Sudokus aren't falsely rejected. Addresses the `_4_3941682` failure mode where the detector locks onto a crossword puzzle on the same newspaper page.
- [ ] Publish the 38-image ground-truth dataset as a standalone HuggingFace dataset for reproducibility — the `Examples/Ground Example/` directory is currently gitignored because the images are larger than the repo should carry.
- [ ] Extend GitHub Actions CI to run `evaluate_detection.py` as a regression guard (blocked on making the ground-truth images accessible to CI — resolves once the HuggingFace dataset publication above lands).
- [ ] Add an end-to-end latency benchmark covering the full `image → solved grid` pipeline, not just per-stage timings.
- [ ] Explore constraint-propagation solvers (arc-consistency, naked pairs, hidden singles) as a faster-than-backtracking alternative for expert-level puzzles — the 2025 comparative study of Bhattarai et al. [\[2\]](#references) reports 1.27×–2.91× speedups for heuristic-based CSP solvers over pure recursive backtracking.
- [ ] Re-annotate the three ground-truth puzzles with duplicate-digit transcription errors that the solver benchmark surfaced, bringing the solvable count from 35/38 to 38/38.

## References

<a id="references"></a>

1. **S. Kamal, S. S. Chawla, and N. Goel**, "Detection of Sudoku Puzzle using Image Processing and Solving by Backtracking, Simulated Annealing and Genetic Algorithms: A Comparative Analysis," in *2015 Third International Conference on Image Information Processing (ICIIP)*, IEEE, Dec. 2015, pp. 179–184.
2. **A. Bhattarai, D. Uprety, P. Pathak, S. N. Shrestha, S. Narkarmi, and S. Sigdel**, "A Study Of Sudoku Solving Algorithms: Backtracking and Heuristic," *arXiv preprint* [arXiv:2507.09708](https://arxiv.org/abs/2507.09708), July 2025.

## License

MIT
