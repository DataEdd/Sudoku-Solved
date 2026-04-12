"""Extract labeled training cells from Wicht's V2 training set into
`ocr_v2/data/train_cells/`.

Pipeline per train image:

1. Read the 4-point outline from `outlines_sorted.csv` (or
   `extra_outlines.csv` for hand-annotated images)
2. Read the 9x9 ground-truth digit grid from `images/imageN.dat`
3. Read phone + resolution metadata from the same .dat
4. Load the JPEG, warp it to a 450x450 grid via the standard 4-point
   `perspective_transform` from the parent project
5. Slice the warped grid into 81 cells via raw 1/9 splits — NO margin
   trim. The trim is the v2 agent's hyperparameter to tune.
6. Save each cell as
   `ocr_v2/data/train_cells/{stem}_r{row}_c{col}_gt{digit}.png`
7. Append per-image metadata (phone, resolution, n_filled, n_empty)
   to `ocr_v2/data/train_metadata.jsonl`

Run after `annotate_missing_outlines.py` (idempotent — re-running
overwrites the output dir cleanly).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.extraction import perspective_transform  # noqa: E402

WICHT_ROOT = PROJECT_ROOT / "research" / "wichtounet_dataset"
TRAIN_MANIFEST = WICHT_ROOT / "datasets" / "v2_train.desc"
OUTLINES_CSV = WICHT_ROOT / "outlines_sorted.csv"
IMAGES_DIR = WICHT_ROOT / "images"

OCR_V2_ROOT = PROJECT_ROOT / "ocr_v2"
EXTRA_OUTLINES_CSV = OCR_V2_ROOT / "data" / "extra_outlines.csv"
OUTPUT_CELLS_DIR = OCR_V2_ROOT / "data" / "train_cells"
OUTPUT_METADATA = OCR_V2_ROOT / "data" / "train_metadata.jsonl"

WARP_SIZE = 450  # 50x50 px per cell, plenty of resolution for trim experiments


def load_train_names() -> List[str]:
    return [
        Path(line.strip()).name
        for line in TRAIN_MANIFEST.read_text().splitlines()
        if line.strip()
    ]


def load_outlines_combined() -> Dict[str, np.ndarray]:
    """Merge outlines_sorted.csv (filtered to train) with extra_outlines.csv.

    Returns dict mapping image filename → (4, 2) float32 array of
    [TL, TR, BR, BL] corners.
    """
    train_names = set(load_train_names())
    outlines: Dict[str, np.ndarray] = {}

    def parse_csv(path: Path) -> None:
        with path.open() as f:
            reader = csv.reader(f)
            try:
                next(reader)  # header
            except StopIteration:
                return
            for row in reader:
                if not row:
                    continue
                name = Path(row[0]).name
                if name not in train_names:
                    continue
                try:
                    coords = [float(x) for x in row[1:9]]
                except (ValueError, IndexError):
                    continue
                outlines[name] = np.array(coords, dtype=np.float32).reshape(4, 2)

    parse_csv(OUTLINES_CSV)
    if EXTRA_OUTLINES_CSV.exists():
        parse_csv(EXTRA_OUTLINES_CSV)

    return outlines


def parse_dat(path: Path) -> Tuple[str, str, List[List[int]]]:
    """Parse a Wicht .dat file → (phone, resolution, 9x9 grid)."""
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    phone = lines[0] if lines else ""
    resolution = ""
    if len(lines) > 1:
        parts = lines[1].replace(":", " ").split()
        if parts:
            resolution = parts[0]
    grid: List[List[int]] = []
    for row_line in lines[2:]:
        tokens = row_line.split()
        try:
            row = [int(t) for t in tokens]
        except ValueError:
            continue
        if len(row) == 9:
            grid.append(row)
        if len(grid) == 9:
            break
    while len(grid) < 9:
        grid.append([0] * 9)
    return phone, resolution, grid[:9]


def warp_with_outline(image: np.ndarray, corners: np.ndarray, size: int) -> np.ndarray:
    """4-point perspective warp using the parent-project helper, then resize.

    `perspective_transform` auto-computes the warped square size from
    the input contour (max of width / height of the source quad), so
    the natural output size varies per image. We resize to a fixed
    `size`×`size` afterward so every training cell is the same shape
    regardless of source resolution. INTER_AREA is the right
    downsample interpolation for the typical case (warp size > 450).
    """
    contour = corners.reshape(4, 1, 2).astype(np.float32)
    warped = perspective_transform(image, contour)
    return cv2.resize(warped, (size, size), interpolation=cv2.INTER_AREA)


def slice_into_cells(warped: np.ndarray) -> List[np.ndarray]:
    """Slice the warped grid into 81 cells via raw 1/9 splits.

    NO margin trim — the v2 agent owns the trim hyperparameter and
    needs the full cell area to experiment with different trim
    percentages downstream.
    """
    h, w = warped.shape[:2]
    cell_h = h // 9
    cell_w = w // 9
    cells = []
    for r in range(9):
        for c in range(9):
            cells.append(
                warped[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract OCR v2 training cells from Wicht V2 train set"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N train images (for smoke testing)",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Wipe data/train_cells/ before extraction (default: append/overwrite)",
    )
    args = parser.parse_args()

    if not TRAIN_MANIFEST.exists():
        raise SystemExit(
            f"{TRAIN_MANIFEST} not found — clone the Wicht dataset first"
        )

    train_names = load_train_names()
    if args.limit:
        train_names = train_names[: args.limit]

    outlines = load_outlines_combined()

    print("=" * 70)
    print(" OCR v2 — prep training cells")
    print("=" * 70)
    print(f"  V2 train images:                  {len(train_names)}")
    print(f"  Combined outlines (train subset): {len(outlines)}")

    missing = [n for n in train_names if n not in outlines]
    if missing:
        print()
        print(f"  WARNING: {len(missing)} train images have no outline:")
        for n in missing[:10]:
            print(f"    {n}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
        print()
        print("  Run `python -m ocr_v2.scripts.annotate_missing_outlines` first.")
        raise SystemExit(1)
    print(f"  Cells to extract:                 {len(train_names) * 81}")
    print()

    if args.clean and OUTPUT_CELLS_DIR.exists():
        print(f"  Cleaning {OUTPUT_CELLS_DIR}/ ...")
        shutil.rmtree(OUTPUT_CELLS_DIR)
    OUTPUT_CELLS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_records: List[Dict] = []
    n_cells_written = 0
    n_filled_total = 0
    n_empty_total = 0

    for i, name in enumerate(train_names, 1):
        img_path = IMAGES_DIR / name
        dat_path = IMAGES_DIR / name.replace(".jpg", ".dat")
        if not img_path.exists() or not dat_path.exists():
            print(f"  SKIP {name} — missing image or .dat", flush=True)
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  SKIP {name} — cv2 failed to load", flush=True)
            continue

        phone, resolution, gt_grid = parse_dat(dat_path)
        corners = outlines[name]
        warped = warp_with_outline(image, corners, WARP_SIZE)
        cells = slice_into_cells(warped)

        stem = name.replace(".jpg", "")
        n_filled_img = 0
        n_empty_img = 0
        for r in range(9):
            for c in range(9):
                idx = r * 9 + c
                gt_digit = gt_grid[r][c]
                cell_path = OUTPUT_CELLS_DIR / f"{stem}_r{r}_c{c}_gt{gt_digit}.png"
                cv2.imwrite(str(cell_path), cells[idx])
                n_cells_written += 1
                if gt_digit == 0:
                    n_empty_img += 1
                else:
                    n_filled_img += 1

        n_filled_total += n_filled_img
        n_empty_total += n_empty_img
        metadata_records.append(
            {
                "filename": name,
                "phone": phone,
                "resolution": resolution,
                "n_filled": n_filled_img,
                "n_empty": n_empty_img,
            }
        )

        if i % 20 == 0 or i == len(train_names):
            print(
                f"  [{i:>3d}/{len(train_names)}] "
                f"{n_cells_written} cells written so far",
                flush=True,
            )

    OUTPUT_METADATA.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_METADATA.open("w") as f:
        for rec in metadata_records:
            f.write(json.dumps(rec) + "\n")

    print()
    print(f"  Wrote {n_cells_written} cells to {OUTPUT_CELLS_DIR}")
    print(f"  Filled cells: {n_filled_total}")
    print(f"  Empty cells:  {n_empty_total}")
    print(f"  Metadata:     {OUTPUT_METADATA}")
    print()
    print("Done. Ready to copy ocr_v2/ to the isolated playground.")


if __name__ == "__main__":
    main()
