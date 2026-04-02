# Annotation Tool v2 — Implementation Prompt

## Task

Rewrite the annotation tool at `_archive/legacy_detection/evaluation/annotate.py` to support 16-point corner annotation and multi-digit ground truth cells. Save the new tool to `_archive/legacy_detection/evaluation/annotate.py` (overwrite the old one). The ground truth file should be saved to `_archive/legacy_detection/evaluation/ground_truth_annotated.json` (fresh start, no migration needed). Ensure to empty all previous ground truth entries.

Read the existing tool first to understand the structure, then implement the changes below.

## Image Set

Iterate over all images in `Examples/Ground Example/` (39 images). Do NOT use random sampling — annotate every image in sorted order.

## Change 1: 16-Point Corner Annotation

Replace the 8-point (4 outer + 4 center-box) annotation with 16 points — all intersections of the thick grid lines (the lines separating the nine 3x3 boxes, plus the outer border).

These 16 points form a 4x4 grid:

```
Row 0 (top edge):    P0  P1  P2  P3
Row 3:               P4  P5  P6  P7
Row 6:               P8  P9  P10 P11
Row 9 (bottom edge): P12 P13 P14 P15
```

Click order should be row-by-row, left to right: P0, P1, P2, P3, then P4, P5, P6, P7, etc.

### UI requirements:
- Show which point to click next with a label (e.g., "Click P5 (row 3, col 3) — 5/16")
- Draw placed points with distinct colors per row (e.g., row 0 = green, row 3 = cyan, row 6 = yellow, row 9 = red)
- Draw lines connecting adjacent points as they are placed (horizontal and vertical grid lines)
- Keep undo ('u'), reset ('r'), skip ('s'), quit ('q') controls
- Single phase — no need for separate outer/center phases, just 16 sequential clicks
- After all 16 are placed, press Enter to confirm

### Warp preview:
After confirming 16 points, show a piecewise-warped preview. With 16 points you have 9 quads where every corner is a real annotated point (no interpolation). The 9 source quads are:

```
Box 0: P0,  P1,  P5,  P4     Box 1: P1,  P2,  P6,  P5     Box 2: P2,  P3,  P7,  P6
Box 3: P4,  P5,  P9,  P8     Box 4: P5,  P6,  P10, P9     Box 5: P6,  P7,  P11, P10
Box 6: P8,  P9,  P13, P12    Box 7: P9,  P10, P14, P13    Box 8: P10, P11, P15, P14
```

Each maps to the corresponding 1/3 x 1/3 region in the output square.

## Change 2: Multi-Digit Cell Entry

When typing the grid in the terminal, support `X/Y` notation for cells where multiple digits are acceptable (e.g., a cell that could be read as 1 or 7).

### Input format:
- `0` → empty cell → stored as `0`
- `3` → single digit → stored as `3`
- `1/7` → either digit is acceptable → stored as `[1, 7]`
- `3/8/9` → any of these → stored as `[3, 8, 9]`

### Parsing rules:
- Split each row by whitespace
- For each token, check if it contains `/` — if yes, split on `/` and parse each part as int
- Validate: all values must be 0-9, lists must have 2+ unique values in range 1-9
- Show the warped grid image while typing so the user can read the digits

### Example row input:
```
Row 1: 5 0 9 1/7 0 0 0 2 0
```

Stored as: `[5, 0, 9, [1, 7], 0, 0, 0, 2, 0]`

## Change 3: JSON Output Format

```json
{
  "images": [
    {
      "path": "Examples/Ground Example/_0_1436352.jpeg",
      "corners_16": [
        [x, y], [x, y], [x, y], [x, y],
        [x, y], [x, y], [x, y], [x, y],
        [x, y], [x, y], [x, y], [x, y],
        [x, y], [x, y], [x, y], [x, y]
      ],
      "grid": [
        [5, 0, 9, [1, 7], 0, 0, 0, 2, 0],
        ...
      ]
    }
  ]
}
```

- `corners_16`: flat list of 16 [x, y] pairs in row-major order (P0 through P15)
- `grid`: 9x9 where each cell is `0`, `int`, or `list[int]`
- Backward-compatible corner extraction: outer corners = indices [0, 3, 15, 12] (TL, TR, BR, BL), center-box corners = indices [5, 6, 10, 9] (CTL, CTR, CBR, CBL)

## What NOT to Change

- Keep the same general UX flow: pick corners in OpenCV window → type grid in terminal
- Keep save-after-each-image behavior (resume-safe)
- Keep the skip and quit functionality
- Keep the `draw_grid_overlay` function for showing grid lines on warped preview

## Future Note (do NOT implement now)

The sudoku puzzles include both handwritten and printed text. The CNN will need retraining on a mixed dataset to handle both styles. This is tracked separately.
