# Sudoku Solver

A web application that extracts Sudoku puzzles from photos and solves them. Uses computer vision for grid detection, a custom CNN for digit recognition, and backtracking for solving.

**Stack:** Python (FastAPI, OpenCV, PyTorch) + Vanilla HTML/CSS/JavaScript

```
Photo  →  Grid Detection  →  Perspective Warp  →  CNN OCR  →  Solver  →  Solution
          (4-step fallback)   (piecewise option)   (102K params)  (backtracking)
```

## The Problem

Taking a photo of a Sudoku puzzle and solving it sounds simple. In practice, every step has failure modes:

- **Detection:** Finding a 9x9 grid in a cluttered photo with variable lighting, shadows, and curved paper
- **Extraction:** Straightening a grid that's warped by paper curvature or camera angle
- **Recognition:** Telling a `1` from a `7`, an empty cell from a faint mark, across 81 cells per puzzle
- **Solving:** Handling OCR errors that produce invalid puzzles

## Grid Detection

### What Didn't Work

Early benchmarks on 23 clean/augmented images showed contour detection at 100%. When tested on 20 real newspaper photos, it detected **zero**.

| Method | Clean Images | Real Photos | Why It Failed |
|--------|-------------|-------------|---------------|
| Contour (largest quad) | 100% (23/23) | 0% (0/20) | Background clutter, curved paper |
| Hough Polar | 91% | 0% | Impractically slow (1.4s), fragile |
| Simple Baseline | 70% | 65% | Inconsistent corners |
| Sobel + Flood Fill | 61% | 65% | Finds border, not grid |

**Lesson:** What works on synthetic data can completely fail on real-world images. Always test on real data.

### What Works: `detect_grid_v2`

A 4-step deterministic fallback chain that achieves **34/38 (89.5%)** detection on real newspaper photos:

1. **CLAHE + RETR_TREE + structure-aware scoring** — Warps each quad candidate and checks if its interior has 10 evenly-spaced lines in both directions and ~81 cell-sized regions. This rejects non-grid quads that happen to be large and centered.

2. **Morph dilate/erode fallback** — Closes broken contour gaps from faded print.

3. **Aggressive CLAHE (clip=6)** — Recovers grids in extreme lighting conditions.

4. **Light morph fallback** — Catches remaining edge cases with fragmented lines.

Each step is tried in order; the first detection is accepted. Sub-pixel corner refinement via `cv2.cornerSubPix` on all results.

### Piecewise Perspective Correction

Standard 4-corner homography assumes the grid is planar. Curved newspaper paper breaks this — the interior grid lines warp even when the outer corners are correct.

The solution uses 8 annotated points (4 outer + 4 center-box corners) to define 9 independent homographies, one per 3x3 box region. Each box gets its own perspective correction, so interior distortion from paper curvature is straightened locally.

A 16-point annotation tool captures all thick gridline intersections for evaluation. The ground truth dataset has 37 annotated newspaper photos with full 16-point corner data.

## CNN Digit Recognition

Custom PyTorch CNN replaces Tesseract OCR:

| | CNN | Tesseract |
|---|---|---|
| **Accuracy** | 99.5% | ~90% |
| **Speed (81 cells)** | 223ms | 3,189ms (14x slower) |
| **Confidence scores** | Per-cell softmax | None |
| **Dependencies** | 24KB ONNX model | System binary |

**Architecture** (102K parameters):

```
Input (1, 28, 28) grayscale
  → Conv2d(1→32, 3x3) → BatchNorm → ReLU → MaxPool
  → Conv2d(32→64, 3x3) → BatchNorm → ReLU → MaxPool
  → Conv2d(64→128, 3x3) → BatchNorm → ReLU → AdaptiveAvgPool
  → Linear(128→64) → ReLU → Dropout(0.3)
  → Linear(64→10)
```

Trained on MNIST (60K digits) + 5K synthetic empty cells. 30 epochs on Apple MPS, 5.4 minutes. Deploys as a 24KB ONNX model — ONNX Runtime preferred at inference, PyTorch as fallback.

## Solver

| Puzzle | Backtracking | Simulated Annealing |
|--------|-------------|-------------------|
| Easy | 0.5ms, 51 nodes | 679ms, 57K iters |
| Medium | 38ms, 3.3K nodes | Failed (500K iters) |
| Hard | 18ms, 1.7K nodes | Failed (500K iters) |

**Backtracking** (default): Constraint propagation with MRV (Minimum Remaining Values) heuristic. Deterministic, guaranteed to find the unique solution, solves any valid puzzle in under 40ms.

**Simulated annealing**: Physics-inspired stochastic optimization. Initializes each 3x3 box with missing digits, swaps non-fixed cells to minimize row/column conflicts. Interesting but impractical — fails on medium/hard puzzles. Kept for comparison.

## Pipeline Debug Visualizer

An interactive debug page (`/debug`) exposes every intermediate step of the pipeline with tunable parameters:

- **Preprocessing:** Grayscale → blur → adaptive threshold (adjustable kernel size, block size, threshold C)
- **Contour detection:** All candidates visualized with area labels, winning quad highlighted
- **Corner adjustment:** Drag corners manually and re-run the pipeline from the warp step
- **Cell OCR:** Raw cell images and CNN-processed inputs side by side, with per-cell confidence and white pixel ratio
- **Result:** Extracted grid and solved grid displayed together

Sliders auto-rerun the pipeline with 400ms debounce — tweak a parameter and see the effect immediately.

## Ground Truth & Evaluation

**Annotation tool** (`python -m evaluation.annotate`): Interactive 16-point corner picker + terminal-based grid entry. Supports multi-value cells (e.g., `0/1/7` for ambiguous digits), `--redo` for fixing specific annotations, `--remove` for deleting entries.

**Evaluation** (`python -m evaluation.evaluate_detection`): Runs parameterized detection against all 37 GT images. Reports per-image corner error (px), IoU, and detection rate. Supports full parameter sweeps across CLAHE clip, blur kernel, block size, threshold C, epsilon, and morphological operations.

## Architecture

```
main.py                              # FastAPI server
app/
  api/v1/endpoints/sudoku.py         # POST /api/extract, POST /api/solve, POST /api/debug
  core/
    extraction.py                    # detect_grid_v2 + perspective transform + cell extraction
    ocr.py                           # DigitRecognizer protocol, Tesseract/CNN backends
    solver.py                        # Backtracking + simulated annealing
    verifier.py                      # Puzzle validation + solution verification
  ml/
    model.py                         # SudokuCNN architecture (102K params)
    dataset.py                       # MNIST + synthetic empty cell dataset
    train.py                         # Training script with early stopping
    recognizer.py                    # CNNRecognizer: ONNX/PyTorch batch inference
    checkpoints/                     # ONNX (24KB) + PyTorch model files
  models/schemas.py                  # Pydantic request/response models
evaluation/
  annotate.py                        # 16-point corner annotation tool
  evaluate_detection.py              # Detection accuracy evaluation + parameter sweep
  ground_truth.json                  # 37 annotated images with 16-point corners + grids
templates/
  index.html                         # Production frontend (camera-first mobile UI)
  debug.html                         # Interactive pipeline debug visualizer
static/                              # JS, CSS, icons, PWA manifest
_archive/                            # Archived: legacy detection methods, benchmarks
Examples/                            # Test images
```

## Quick Start

```bash
git clone https://github.com/DataEdd/Sudoku-Solved.git
cd Sudoku-Solved
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Visit http://localhost:8000

### API

```bash
# Extract grid from image
curl -X POST http://localhost:8000/api/extract -F "file=@photo.jpg"

# Solve a puzzle
curl -X POST http://localhost:8000/api/solve \
  -H "Content-Type: application/json" \
  -d '{"grid": [[5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],[8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],[7,0,0,0,2,0,0,0,6],[0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,7,9]]}'

# Health check
curl http://localhost:8000/api/health
```

### Train the CNN

```bash
python -m app.ml.train              # 30 epochs, ~5 min on MPS
python -m app.ml.train --epochs 10  # Quick run
```

### Evaluate Detection

```bash
python -m evaluation.evaluate_detection              # Single run with defaults
python -m evaluation.evaluate_detection --sweep       # Full parameter sweep
```

## What I Learned

1. **Synthetic benchmarks lie.** Contour detection scored 100% on clean/augmented images and 0% on real newspaper photos. The gap wasn't small — it was total failure. Any detection method that hasn't been tested on real-world images with real-world clutter is unproven.

2. **Self-reported confidence is meaningless.** Sobel+Flood reported 1.00 confidence while only detecting 61% of images. The only honest evaluation is running against ground truth. Build the ground truth tooling first.

3. **Simple beats clever — until it doesn't.** The "largest quadrilateral" heuristic is elegant and fast, but it's fundamentally fragile: any background rectangle larger than the puzzle breaks it. Structure-aware scoring (checking that the quad's interior actually looks like a grid) is the key insight that made detection reliable.

4. **The hard problem is paper curvature.** Flat image detection is essentially solved. The real challenge is newspaper photos where paper curves — 4-corner perspective transform produces uneven cell spacing because the interior grid is warped. Piecewise correction using interior grid points fixes this but requires knowing where those points are.

5. **Small CNNs crush legacy OCR.** A 102K-parameter model trained in 5 minutes on a laptop is 14x faster and significantly more accurate than Tesseract for this specific task. The ONNX export is 24KB. There's no reason to use general-purpose OCR when you can train a specialist.

## Tech Stack

- **Backend:** Python 3.10+, FastAPI, OpenCV, PyTorch
- **OCR:** Custom CNN (102K params, ONNX Runtime for deployment)
- **Solver:** Backtracking with constraint propagation + MRV heuristic
- **Frontend:** Vanilla HTML/CSS/JavaScript, camera-first mobile UI
- **Evaluation:** 37-image ground truth with 16-point annotations, automated parameter sweeps

## License

MIT
