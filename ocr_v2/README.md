# OCR v2 — sub-project

A clean reimplementation of the Sudoku OCR component, trained from scratch on Baptiste Wicht's V2 training set (160 real newspaper photographs with per-image .dat ground-truth annotations and Lars @panexe's 4-point corner outlines).

This is a **separate sub-project** from the parent Sudoku-Solved repo. It exists to test a single hypothesis:

> **Does in-distribution training matter more than architecture choice?**
>
> The shipped v5.1 CNN was trained on synthetic data (MNIST + 67 printed fonts + Chars74K + newsprint-augmented synthetic empties) and gets 61.5% filled-cell accuracy on Wicht's V2 test set. Training from scratch on Wicht's actual training set should close most or all of the gap to Wicht's reported 82.5% perfect-image rate.

## Folder layout

```
ocr_v2/
├── README.md                    you are here
├── PROMPT.md                    full brief for the new Claude Code agent
├── LAB_NOTES.md                 required experiment journal — agent fills in
├── research/                    OCR-relevant PDFs (copied from parent ../research/)
│   ├── Wicht_2014_DBN.pdf       canonical baseline paper
│   ├── Kamal_2015_backtracking.pdf
│   ├── Bhattarai_2025_solvers.pdf
│   └── A_24-27.1.pdf
├── data/
│   ├── README.md                schema + the no-test-data rule
│   ├── train_cells/             (gitignored) ~12,900 PNGs after prep
│   ├── train_metadata.jsonl     per-image phone/resolution
│   └── extra_outlines.csv       hand-annotated outlines for missing images
├── scripts/
│   ├── annotate_missing_outlines.py    one-time, before prep
│   ├── prep_training_data.py           one-time, populates train_cells/
│   └── final_eval.py                   GATED — only the user runs this
├── src/                         agent's OCR implementation goes here
├── checkpoints/                 (gitignored) agent's model outputs
├── results/                     (gitignored) agent's eval output
└── .gitignore
```

## How this is used

1. **Parent repo (this directory) runs the prep step** — joins train manifest with outline CSV, hand-annotates any missing-outline images using the existing parent-tree picker, warps each train image, slices into 81 cells, saves labeled PNGs to `data/train_cells/`. This is the only step that touches parent-tree filesystem paths.
2. **The folder is then COPIED to an isolated sibling directory** outside the main repo (e.g. `~/Documents/Sudoku-OCR-v2-isolated/`). A fresh Claude Code session opens there. The agent has zero filesystem access to the parent tree — they cannot peek at v5.1 code, the 38-image GT benchmark, the macfooty labeled dataset, or the Wicht V2 test set.
3. **The agent works from `PROMPT.md`** — literature review, game plan, baseline, iterate, final model.
4. **When the agent finishes**, their `src/`, `checkpoints/`, `LAB_NOTES.md`, `RESULTS.md`, and `GAME_PLAN.md` are synced back into the main repo's `ocr_v2/` directory via `cp -r`.
5. **The user runs `scripts/final_eval.py`** in the main repo, which loads the v2 model and evaluates against `research/wichtounet_dataset/datasets/v2_test.desc` for the official benchmark numbers.

## Why isolated

The agent must NEVER read the V2 test set during development. A standard `git worktree` would still expose the parent tree files (test images, .dat files, v5.1 code, GT benchmark) to the agent's filesystem view. A separate sibling directory has no parent-relationship, so containment is enforced by the filesystem itself, not by an honor-system rule in the prompt.

## Cell-extraction trim observation

The user flagged that the v5.1 cell preprocessing (`app/ml/recognizer.py:115-117`) trims a 10% margin per side before resizing to 28×28, but this may not be aggressive enough — grid-line residue likely remains at cell borders. The prep script saves cells with the **full 1/9 slice (no trim)**, so the v2 agent can experiment with different trim percentages (10% / 12% / 15% / 18% / 20%) as a hyperparameter during preprocessing. This is required as a Phase-2 ablation in the agent's prompt.
