# Project Roadmap & Learnings

Tracking the full pipeline from current state to "complete project." Non-documentation work only.

## Current State (2026-04-03)

| Component | Status | Notes |
|-----------|--------|-------|
| Grid Detection | Validated | 34/38 (89%) on GT, `detect_grid_v2` with 4-step fallback |
| Detection Evaluation | Complete | `evaluate_detection.py`, 38 GT images, parameter sweeps |
| CNN Model | Retrained | 102K params, 99.5% test acc, fine-tuned on newspaper cells, ONNX deployed |
| OCR Pipeline | Improved | Grayscale preprocessing, class 1-9 prediction, conf_th=0.85 |
| OCR Evaluation | Complete | `evaluate_ocr.py`, 87.8% overall accuracy, 77.4% filled, <1% hallucination |
| Piecewise Warp | Available | `extract_cells_piecewise()` + `infer_center_corners()`, but simple warp performs better |
| Solver | Complete | Backtracking <1ms median, simulated annealing kept for demo |
| Frontend | Complete | Camera flow + confidence-colored cells (green/orange/red) |
| Debug Visualizer | Good | Shows all pipeline steps + per-cell OCR |
| End-to-End Tests | Complete | `tests/test_e2e_pipeline.py` — detection, OCR, and solve regression guards |

## Roadmap

### Phase 1: OCR Evaluation (current priority)
**Goal:** Know exactly where OCR fails and why.
- [ ] Build `evaluate_ocr.py` — benchmark OCR against GT grids for all 37 images
  - Per-cell accuracy (overall, per-image)
  - Confusion matrix (which digits get misread as what)
  - Confidence vs correctness correlation (is the model confident when wrong?)
  - Breakdown: detection-caused errors vs OCR-caused errors
- [ ] Compare equal-slice vs piecewise extraction on same images
- [ ] Identify error patterns: edge cells? faint print? specific digits?

### Phase 2: OCR Improvement
**Goal:** Close the gap between 99.5% model accuracy and 61% pipeline accuracy.

Likely root causes (to be confirmed by Phase 1 analysis):
- [ ] Warp quality — imperfect corners cause cell boundaries to cut through digits
- [ ] Cell margin cropping — too aggressive or not aggressive enough
- [ ] CNN trained on MNIST (handwritten) but puzzles have printed digits
- [ ] Empty cell detection threshold (3% white pixels) may misclassify

Possible fixes:
- [ ] Retrain CNN on printed digit dataset (not just MNIST)
- [ ] Tune cell margin / preprocessing params using GT feedback
- [ ] Wire up piecewise warp for curved-paper images (already built, just needs integration)
- [ ] Confidence-based rejection + user correction flow

### Phase 3: Piecewise Warp Integration
**Goal:** Use the 8-point piecewise warp that's already built.
- [ ] Auto-detect center-box corners from grid structure (after 4 outer corners found)
  - Approach: use detected grid lines to infer thick-line intersections
  - Or: use `corners_16` GT data to train/validate a corner refinement step
- [ ] Wire `extract_cells_piecewise()` into `/api/extract` endpoint
- [ ] Add piecewise toggle to debug page
- [ ] Evaluate improvement: piecewise vs simple warp on curved-paper subset

### Phase 4: 16-Point Detection
**Goal:** Automatically find all 16 thick gridline intersections.
- [ ] Build detector that finds interior intersections (not just outer corners)
- [ ] Enables fully automatic piecewise warp with no interpolation
- [ ] 37 annotated images with `corners_16` available for training/evaluation
- [ ] Support graceful degradation: 16 → 8 → 4 points

### Phase 5: Frontend Polish
**Goal:** Surface OCR confidence to the user.
- [ ] Display `confidence_map` in review screen (color-code uncertain cells)
- [ ] Low-confidence cells highlighted for manual correction
- [ ] Show detection diagnostics on failure (why did it fail?)
- [ ] Better error messages

### Phase 6: End-to-End Testing
**Goal:** Automated validation of full pipeline.
- [ ] Integration test: image → detection → warp → OCR → solve → verify against GT
- [ ] Per-image pass/fail with detailed failure reasons
- [ ] Regression guard: any pipeline change must not degrade E2E accuracy

### Phase 7: Stretch Goals
- [ ] Active learning: user corrections feed back into CNN retraining
- [ ] PWA (service worker, installable)
- [ ] Web scraper for training data collection
- [ ] Database layer for puzzle history

---

## What I Learned

### OCR was never validated (2026-04-03)
The project had thorough detection evaluation (37 GT images, parameter sweeps, corner error metrics) but **zero OCR validation**. The CNN's 99.5% accuracy was on the MNIST test set — nobody checked what it actually does on real newspaper photos through the full pipeline.

First measurement today: **61% pipeline OCR accuracy** on 38 real photos. The gap between 99.5% and 61% means either the warp is mangling cells, the CNN can't handle printed digits, or both. We don't know yet because we never built the tooling to find out.

**Lesson:** Evaluating components in isolation (detection accuracy, model accuracy) gives false confidence about the whole system. Build end-to-end evaluation early — it reveals where the real bottleneck is.

### Self-reported confidence is meaningless (redux)
This came up before with detection (Sobel reporting 1.00 confidence at 61% accuracy). Now the same pattern appears in OCR: the CNN reports high confidence (mean ~0.88) while getting 61% of cells wrong on real photos. Softmax confidence != correctness. The only honest evaluation is comparison against ground truth.

### The evaluation gap was invisible
We had a ground truth file with both corner annotations AND digit grids. The detection evaluation used the corners. The digit grids sat unused. The capability to evaluate OCR existed in the data — we just never wrote the code to use it. Easy to miss when you're focused on one part of the pipeline.

### Train/inference preprocessing mismatch (2026-04-03)
The CNN was trained on grayscale MNIST tensors ([0, 1] continuous) but inference applied Otsu binarization (forcing pixels to {0, 1}). Switching inference to grayscale (invert + normalize, matching training) improved overall accuracy from 57.3% to 62.2% — a free win from fixing a mismatch nobody noticed.

### Data leakage: a cautionary tale (2026-04-03)
First attempt at fixing OCR: added the 3078 GT cells directly to training data. Result looked amazing — 87.8% overall, 1 hallucination. But this was textbook data leakage: we trained on the same images we evaluated on. The model memorized the test set, not the domain.

**Lesson:** Always check whether your training and evaluation data overlap. Impressive numbers on a leaked test set mean nothing for generalization.

### Font-rendered digits: the honest fix (2026-04-03)
Replaced leaked GT cells with `PrintedDigitDataset` — digits 1-9 rendered in 370 system fonts (Helvetica, Times, Courier, etc.) at 28x28. Combined with MNIST + synthetic empty cells. No overlap with evaluation data.

**Honest results (MNIST + Printed + Grayscale preprocessing):**
| Metric | MNIST-only (Otsu) | MNIST + Printed (Grayscale) | Change |
|--------|-------------------|----------------------------|--------|
| Overall | 57.3% | 76.3% | +19% |
| Hallucinated | 594 | 86 | -85% |
| Wrong digits | 335 | 79 | -76% |
| Empty cell acc | 53.4% | 93.3% | +40% |
| Filled cell acc | 60.6% | 61.6% | +1% |
| Missed digits | 284 | 554 | +95% |

Hallucinations and wrong digits crushed. Empty cells nearly perfect. But filled cell accuracy barely moved — the model is now too conservative (high confidence threshold causes misses). Next steps: tune threshold, try Chars74K or SVHN.

### Research: Khan et al. (2024) is not credible (2026-04-03)
"Optimized real-time sudoku puzzle solving" — claims 100% on 200 images. Published in *International Journal of Multidisciplinary Research and Growth Evaluation* (low-impact, not peer-reviewed by a rigorous venue). Uses MNIST + largest-contour — the exact approaches we proved fail on real photos. No independent verification. Essentially a student project writeup.

### Research: better datasets exist (2026-04-03)
- **Chars74K** — printed + handwritten + natural characters. Used by GaneshSparkz/OCR-Sudoku-Solver, achieved 99.1% on Sudoku digits.
- **SVHN** — 600K real printed digit photos. Backbone pretrained on SVHN raised accuracy to 98.4% in one study.
- **francois-rozet/sudoku** — renders digits in 445 fonts, same approach as our PrintedDigitDataset.
- **kaydee0502/printed-digits-dataset** — Sudoku-specific, pretrained CNN at 96%.

### Deep analysis: OCR has hit diminishing returns (2026-04-04)
Full analysis in `evaluation/ocr_analysis.ipynb`. Key findings:

**The 99.5% → 62% gap is an image quality problem, not a model problem.**
- Wrong digit predictions: only 41 out of 1720 filled cells (2.4%). The CNN is accurate when it commits.
- 407 missed cells have median CNN confidence of 0.58 — genuinely ambiguous, not a threshold issue.
- Lowering the threshold rescues digits at a 2:1 correct:wrong ratio — net negative for puzzle solving.

**The dataset is bimodal:**
- 15/38 images: ≥90% accuracy — model is nearly perfect
- 9/38 images: <50% accuracy — impossible inputs (toilet paper, cat on puzzle, extreme blur)
- These 9 bad images drag the mean from ~95% to ~62%

**Bottom line:** Further OCR training data (Chars74K, SVHN) would yield marginal gains (+2-5%). The bottleneck is cell image quality from detection/warp, not the classifier. The pipeline works well on normal photos and fails on extreme edge cases — acceptable for a portfolio project.

**Remaining high-impact work (not OCR):**
- [ ] Fix the 4 undetected images (detection, not OCR)
- [ ] Improve corner detection on the 9 worst-warped images
- [ ] User correction flow — frontend already shows confidence colors
- [ ] Consider higher-resolution cells (56x56 instead of 28x28) for borderline cases
