# Sudoku Solver

![tests](https://github.com/DataEdd/Sudoku-Solved/actions/workflows/test.yml/badge.svg)

Extract and solve Sudoku puzzles from photos. Point your camera at a puzzle, get the solution.

**Stack:** Python (FastAPI, OpenCV, PyTorch) + Vanilla HTML/CSS/JavaScript

## Results

| Stage | Metric |
|-------|--------|
| **Grid Detection** | 34/38 (89%) on real newspaper photos |
| **Digit Recognition** | 62% filled-cell accuracy, 93% empty-cell accuracy |
| **Solver** | 0.41 ms median on 38 GT puzzles (see `evaluation/benchmark_solver.py`) |

Tested against 38 ground-truth-annotated newspaper photos.

## How It Works

| 1. Input | 2. Preprocess | 3. Detect |
|:---:|:---:|:---:|
| ![Input](docs/pipeline/1_input.jpg) | ![Preprocess](docs/pipeline/2_threshold.jpg) | ![Detect](docs/pipeline/3_detected.jpg) |

| 4. Warp | 5. OCR | 6. Solve |
|:---:|:---:|:---:|
| ![Warp](docs/pipeline/4_warped.jpg) | ![OCR](docs/pipeline/5_ocr_overlay.jpg) | ![Solve](docs/pipeline/6_solved.jpg) |

**Grid Detection** — CLAHE contrast enhancement + adaptive thresholding, then contour detection with structure-aware scoring. Each candidate quad is warped and checked for grid-like interior (evenly-spaced lines, ~81 cell-sized regions). A 4-step fallback chain handles varying lighting, faded print, and extreme contrast.

**Digit Recognition** — 102K-parameter CNN trained on MNIST + font-rendered printed digits + synthetic empty cells. Deploys as an ONNX model via ONNX Runtime (24 KB protobuf header + ~396 KB external weights in `sudoku_cnn.onnx.data` = ~420 KB total; both files ship in the Docker image). Batch inference returns per-cell confidence scores.

**Solver** — MRV-ordered backtracking with per-cell domain restriction. Deterministic; **0.41 ms median** on the 38-puzzle GT set (see `evaluation/benchmark_solver.py`).

## Features

- **Camera-first web UI** — capture directly from phone or webcam
- **Pipeline debug visualizer** (`/debug`) — every intermediate step with tunable parameters
- **Ground truth evaluation** — 38 newspaper photos with 16-point corner ground truth; production detector benchmarked via `evaluation/evaluate_detection_v2.py`, OCR via `evaluation/evaluate_ocr.py`, solver via `evaluation/benchmark_solver.py`.

## Quick Start

```bash
git clone https://github.com/DataEdd/Sudoku-Solved.git
cd Sudoku-Solved
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Visit http://localhost:8000

## Roadmap

- Wire `extract_cells_piecewise` (`app/core/extraction.py:185`) into `/api/extract` for curved newspaper pages. Currently only used in `evaluate_ocr.py --piecewise` and `annotate.py` preview.
- Wire `/api/debug` (`app/api/v1/endpoints/sudoku.py:117`) to use `detect_grid_v2` instead of the legacy `find_grid_contour` so the debug UI reflects the production detector.
- Add a CNN-vs-Tesseract latency benchmark if a speedup claim is to be restored. Previously-advertised "14x faster" was unbacked and has been removed.
- Publish the 38-image ground-truth dataset as a standalone HuggingFace dataset for reproducibility.
- Extend GitHub Actions CI to run `evaluate_detection_v2.py` as a regression guard (blocked on making `Examples/Ground Example/` images available to CI — currently gitignored).

## License

MIT
