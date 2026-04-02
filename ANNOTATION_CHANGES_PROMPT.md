# Annotation Tool Changes + Ground Truth Review Notebook

## Context

We have a Sudoku puzzle annotation tool at `evaluation/annotate.py` and ground truth at `evaluation/ground_truth.json` (37 annotated images out of 40 total in `Examples/Ground Example/`). The tool supports multi-digit cells via `X/Y` slash notation (e.g., `1/7` → `[1, 7]`), meaning OCR returning ANY of those values counts as a pass.

**Read these files first:**
- `evaluation/annotate.py` (full file — 422 lines)
- `evaluation/ground_truth.json` (full file — 37 images)
- `CLAUDE.md` for project context

---

## Task 1: Allow `0` in Multi-Value Cell Input

Modify `parse_cell()` in `evaluation/annotate.py` (lines 243-265).

### Current behavior
```python
# Line 255-256: multi-digit values must be 1-9
for v in parts:
    if not 1 <= v <= 9:
        raise ValueError(f"Multi-digit values must be 1-9: {token}")
```

### Required change
Allow `0` in the multi-value slash notation:
- `0/1/7` → `[0, 1, 7]` (valid — means empty, 1, or 7 are all acceptable)
- `0/3` → `[0, 3]` (valid — means empty or 3)
- Change the validation from `1 <= v <= 9` to `0 <= v <= 9`
- Keep the requirement for 2+ unique values
- Keep the sorted unique deduplication

That's it for `parse_cell()`. Do NOT remove the multi-value feature. Do NOT change `format_cell()`. Do NOT change any other annotation logic.

### Update the input instructions

In `enter_grid()` (around line 287), update the help text to mention that `0` can be included in slash notation:
```
Use X/Y for ambiguous cells (e.g., 0/1/7 means empty, 1, or 7 are all acceptable)
```

---

## Task 2: Add `--redo` CLI Argument

Add a `--redo` argument to the annotation tool so specific images can be re-annotated.

### Usage
```bash
python -m evaluation.annotate --redo _0_1436352.jpeg _15_7101336.jpeg
```

### Behavior
- Accepts one or more image filenames (just the filename, not full path)
- When provided, the tool ONLY shows those specific images — regardless of whether they're already annotated
- Skips the "already annotated" filter entirely for redo targets
- When saving: if the image already has an entry in ground_truth.json, **overwrite** (replace) that entry instead of appending a duplicate
- Without `--redo`, the tool works exactly as it does today (skips already-annotated images)

### Implementation hints
- Add `argparse` to `main()` with `--redo` accepting `nargs="*"`
- When `--redo` is provided, build the image list from the redo filenames instead of filtering by `already_done`
- On save, check if an entry with the same `path` already exists in `data["images"]` — if so, replace it in-place; otherwise append

---

## Task 3: Ground Truth Review Notebook

Create `evaluation/review_ground_truth.ipynb` — a Jupyter notebook for visually auditing ALL example images against their ground truth.

### Purpose
- Visually verify ground truth annotations are correct
- Spot mistakes that need re-entry
- Identify examples that should be removed or replaced
- See which images were SKIPPED and decide whether to annotate or replace them
- Re-review after backend/annotation tool changes

### Critical: Show ALL images, not just annotated ones

The notebook must iterate over ALL images in `Examples/Ground Example/` (40 images), not just the 37 in ground_truth.json. For each image:
- If it HAS an annotation in ground_truth.json → show the warped grid + ground truth
- If it was SKIPPED (not in ground_truth.json) → show the raw original image with a clear "SKIPPED — NOT ANNOTATED" label, so I can decide whether to reannotate or replace this example

### Layout

**Cell 1: Setup**
```python
import json, cv2, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import os

# Load ground truth
with open("evaluation/ground_truth.json") as f:
    gt = json.load(f)

# Build lookup by path
gt_lookup = {entry["path"]: entry for entry in gt["images"]}

# Get ALL images from the directory (sorted)
IMAGE_DIR = "Examples/Ground Example"
all_images = sorted([
    f"{IMAGE_DIR}/{f}" for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpeg", ".jpg", ".png", ".webp"))
])

print(f"Total images: {len(all_images)}")
print(f"Annotated: {len(gt['images'])}")
print(f"Skipped: {len(all_images) - len(gt['images'])}")
```

**Cell 2: Warp and display helpers**
- Copy `warp_grid_piecewise()` from `annotate.py` (lines 133-180) — notebooks should be self-contained
- Copy `draw_grid_overlay()` from `annotate.py` (lines 183-194)
- Write a `format_grid(grid)` function that prints the 9×9 grid in a readable table:
  - `.` for 0 (empty)
  - digit for single values (1-9)
  - `0/1/7` for multi-value cells
  - Row numbers on the left, column numbers on top
- Write a `validate_grid(grid)` function that checks for:
  - Duplicate non-zero single-value digits in any row, column, or 3×3 box (indicates annotation error)
  - Returns list of violations found

**Cell 3: Configuration**
```python
# Set to None to show ALL images, or set to an index (e.g., 5) to view just one
IMAGE_INDEX = None
```

**Cell 4: Review loop**

For each image in `all_images` (or single image if `IMAGE_INDEX` is set):

**If annotated (path found in gt_lookup):**
1. Load the original image from its `path`
2. Warp it using `corners_16` with `warp_grid_piecewise()`
3. Draw grid overlay on the warped image
4. Create a matplotlib figure with the warped grid image
5. Print the formatted ground truth grid below the image
6. Print the image path and index number
7. Run validation and print any violations prominently
8. Add clear visual separation between images

**If skipped (path NOT in gt_lookup):**
1. Load the original image
2. Display the raw image (no warp — we don't have corners)
3. Print a prominent "⚠ SKIPPED — NOT ANNOTATED" label
4. Print the image path and index number
5. This lets me decide: reannotate this image, replace it, or leave it out

Display size should be readable — at least 5x5 inches per image.

**Cell 5: Summary**
- Total images in directory vs annotated vs skipped
- List the skipped image filenames explicitly
- Count of filled cells vs empty cells across all annotated images
- Count of multi-value cells and which images contain them
- List all images with Sudoku rule violations (duplicate digits in rows/cols/boxes)
- List images where filled cell count is unusual (typical Sudoku has 17-35 given clues — flag outliers)

### Warping details

The `warp_grid_piecewise()` function uses 16 corner points to define 9 quads (3×3 box regions) and perspective-transforms each independently. Signature: `warp_grid_piecewise(image, points_16, size=450)` where `points_16` is a list of 16 `[x, y]` pairs. The `corners_16` in the JSON are stored as `[[x, y], ...]` — convert to numpy array of tuples as needed.

---

## What NOT to do

- Do NOT remove multi-value cell support — it's needed
- Do NOT change the 16-point corner workflow (CornerPicker, pick_corners, warp logic)
- Do NOT change the JSON structure (`{"images": [...]}` with `path`, `corners_16`, `grid`)
- Do NOT migrate or modify existing ground_truth.json data — only change the tool to accept `0` in slash notation going forward
