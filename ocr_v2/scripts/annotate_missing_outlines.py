"""Hand-annotate the 4-point grid outline for any V2 train image whose
filename does NOT appear in `research/wichtounet_dataset/outlines_sorted.csv`.

This script is a thin shim over the existing parent-tree picker
(`evaluation/annotate.py::pick_corners`). The picker collects 16
intersection points; we extract indices [0, 3, 15, 12] as the four
outer corners (TL, TR, BR, BL) and write them to
`ocr_v2/data/extra_outlines.csv` in Wicht's outlines_sorted.csv format
so the prep step can merge both files.

Run from the parent repo (the picker imports from `evaluation/`):

    cd /path/to/Sudoku-Solved
    python -m ocr_v2.scripts.annotate_missing_outlines

Run BEFORE `prep_training_data.py`. The output `extra_outlines.csv`
gets committed to the main repo (it's small) so the annotation work
isn't lost when the ocr_v2/ scaffold is copied to the isolated
playground.

Idempotency: re-running skips images that already have entries in
`extra_outlines.csv`. Use `--redo IMG` to force re-annotation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Set

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.annotate import pick_corners  # noqa: E402

WICHT_ROOT = PROJECT_ROOT / "research" / "wichtounet_dataset"
TRAIN_MANIFEST = WICHT_ROOT / "datasets" / "v2_train.desc"
OUTLINES_CSV = WICHT_ROOT / "outlines_sorted.csv"
IMAGES_DIR = WICHT_ROOT / "images"

EXTRA_OUTLINES_CSV = PROJECT_ROOT / "ocr_v2" / "data" / "extra_outlines.csv"

# Indices into the 16-point picker output that map to the four outer
# corners (TL, TR, BR, BL) of the Sudoku grid. This is the standard
# corners_16 outer-quad convention used throughout the parent project
# (see CLAUDE.md / evaluation/evaluate_ocr.py::GT_OUTER_INDICES).
OUTER_4 = [0, 3, 15, 12]

CSV_HEADER = [
    "filepath",
    "p1_x", "p1_y",
    "p2_x", "p2_y",
    "p3_x", "p3_y",
    "p4_x", "p4_y",
]


def load_train_names() -> List[str]:
    return [
        Path(line.strip()).name
        for line in TRAIN_MANIFEST.read_text().splitlines()
        if line.strip()
    ]


def load_outlined_names() -> Set[str]:
    """Names of images that already have an outline in outlines_sorted.csv."""
    names: Set[str] = set()
    with OUTLINES_CSV.open() as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if row:
                names.add(Path(row[0]).name)
    return names


def load_extra_outlines_existing() -> Set[str]:
    """Names already in extra_outlines.csv (so we can skip them on rerun)."""
    if not EXTRA_OUTLINES_CSV.exists():
        return set()
    names: Set[str] = set()
    with EXTRA_OUTLINES_CSV.open() as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return set()
        for row in reader:
            if row:
                names.add(Path(row[0]).name)
    return names


def append_extra_outline(filename: str, four_corners) -> None:
    """Append one record to ocr_v2/data/extra_outlines.csv (header on first write)."""
    EXTRA_OUTLINES_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not EXTRA_OUTLINES_CSV.exists() or EXTRA_OUTLINES_CSV.stat().st_size == 0
    with EXTRA_OUTLINES_CSV.open("a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        if is_new:
            writer.writerow(CSV_HEADER)
        # Match the format of outlines_sorted.csv: ./images/imageN.jpg
        rel_path = f"./images/{filename}"
        row = [rel_path]
        for x, y in four_corners:
            row.extend([int(round(x)), int(round(y))])
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate the 4-point outline for V2 train images "
                    "missing from outlines_sorted.csv",
    )
    parser.add_argument(
        "--redo", nargs="+", metavar="FILENAME",
        help="Force re-annotation for these specific image filenames "
             "(otherwise idempotent — already-annotated images are skipped)",
    )
    args = parser.parse_args()

    if not TRAIN_MANIFEST.exists():
        raise SystemExit(
            f"{TRAIN_MANIFEST} not found — clone the Wicht dataset first:\n"
            f"  git clone https://github.com/wichtounet/sudoku_dataset "
            f"{WICHT_ROOT}"
        )

    train_names = load_train_names()
    outlined = load_outlined_names()
    already_extra = load_extra_outlines_existing()

    missing = [n for n in train_names if n not in outlined]
    if args.redo:
        redo_set = set(args.redo)
        targets = [n for n in missing if n in redo_set]
        if not targets:
            raise SystemExit(
                f"None of {sorted(redo_set)} appear in the missing-outline list. "
                f"Missing: {missing}"
            )
    else:
        targets = [n for n in missing if n not in already_extra]

    print("=" * 70)
    print(" OCR v2 — annotate missing outlines")
    print("=" * 70)
    print(f"  V2 train images:                {len(train_names)}")
    print(f"  Already in outlines_sorted.csv: {len(train_names) - len(missing)}")
    print(f"  Missing outlines (raw):         {len(missing)}")
    print(f"  Already in extra_outlines.csv:  "
          f"{len([n for n in missing if n in already_extra])}")
    print(f"  To annotate this run:           {len(targets)}")
    print()

    if not targets:
        print("Nothing to do. All train images have outlines.")
        return

    print("Click 16 grid intersection points per image (4 rows × 4 cols of "
          "internal grid corners). Only the 4 outer corners are kept; the "
          "16-point flow is reused from evaluation/annotate.py.")
    print("Keys: 'u'=undo  'r'=reset  's'=skip  'q'=quit")
    print()

    for idx, name in enumerate(targets, 1):
        img_path = IMAGES_DIR / name
        if not img_path.exists():
            print(f"  [{idx}/{len(targets)}] SKIP {name} — image file not found")
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [{idx}/{len(targets)}] SKIP {name} — cv2 failed to load")
            continue

        print(f"  [{idx}/{len(targets)}] {name}")
        result = pick_corners(image, name)
        if result is None:
            print("  Quit. Progress saved.")
            return
        if result == "skip":
            print("  Skipped.")
            continue

        four = [result[i] for i in OUTER_4]
        append_extra_outline(name, four)
        print(f"  Saved 4 outer corners to {EXTRA_OUTLINES_CSV.name}")

    print()
    print(f"Done. {EXTRA_OUTLINES_CSV} now has all hand-annotated outlines.")
    print("Next: run `python -m ocr_v2.scripts.prep_training_data`")


if __name__ == "__main__":
    main()
