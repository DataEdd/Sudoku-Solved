# OCR v2 — data directory

This is the **only** data directory available to the OCR v2 agent. It contains pre-extracted training cells from Baptiste Wicht's V2 training set (160 real newspaper Sudoku photographs). It does **NOT** contain any test data — by design.

## Files (after `prep_training_data.py` runs)

```
data/
├── README.md                  this file
├── train_metadata.jsonl       per-image phone/resolution/source
├── extra_outlines.csv         hand-annotated 4-point outlines for missing-outline train images
└── train_cells/               ~12,900 PNG files (gitignored — regenerate with prep_training_data.py)
    ├── image1_r0_c0_gt0.png   (empty cell at row 0 col 0 of image1.jpg)
    ├── image1_r0_c1_gt7.png   (filled cell with GT digit 7)
    └── ...
```

## Filename schema for `train_cells/*.png`

```
{image_basename}_r{row}_c{col}_gt{digit}.png
```

- `image_basename`: source image filename without extension (e.g. `image1`, `image1005`)
- `row`: 0-8, top to bottom in the warped grid
- `col`: 0-8, left to right in the warped grid
- `digit`: 0 (empty) or 1-9 (filled)

The labels come from the per-image `.dat` ground truth that ships with the Wicht dataset. The 9x9 cell layout comes from a 4-point perspective warp using the outlines from `outlines_sorted.csv` (with the missing-outline images filled in via hand annotation in `extra_outlines.csv`).

**Cells are saved at the full 1/9 slice (no margin trim).** The OCR agent should test different trim percentages as a hyperparameter — the parent project's v5.1 used 10% trim and the user suspects that's too lenient (grid-line residue remains at cell borders).

## `train_metadata.jsonl` schema

One JSON object per line, one line per source image:

```jsonl
{"filename": "image1.jpg", "phone": "iphone 5s", "resolution": "960x1280", "n_filled": 27, "n_empty": 54}
```

Fields:
- `filename`: source image filename (e.g. `image1.jpg`)
- `phone`: phone brand/model from line 1 of the .dat
- `resolution`: pixel resolution from line 2 of the .dat
- `n_filled`: count of GT non-zero cells in this image's 9x9 grid
- `n_empty`: count of GT zero cells (n_filled + n_empty = 81)

## What is NOT in this directory

- The 40-image V2 test set
- Any image from `image{1005, 1009, 1019, 1024, 1041, 1072, 1073, 1080, 1088, 114, 117, 120, 126, 140, 148, 153, 166, 175, 176, 179, 18, 193, 205, 208, 210, 211, 25, 31, 32, 34, 35, 39, 46, 47, 50, 51, 73, 83, 85, 91}.jpg`
- The macfooty 2620-image augmented set
- The parent project's 38-image hand-annotated benchmark
- Any v5.1 model artifacts

If you find yourself wanting one of those files, **stop**. The whole point of this isolated subproject is to test in-distribution training without contamination. Cross-validation on the train cells is your only generalization estimate during development.
