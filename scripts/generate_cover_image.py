"""Generate a cover image for the Kaggle labeled dataset page.

Reads ``data/labeled_dataset/data.jsonl`` (produced by
``scripts/build_labeled_dataset.py``), picks 16 "happy-path" sample
records — detectable, solvable, a realistic filled-count range, NOT
in the 38-image ground-truth benchmark — and renders a 4×4 grid of
the source JPEGs with the ``detect_grid`` quadrilateral overlaid in
green. Saves the result to ``data/labeled_dataset/cover.png`` at
1120×560 (2× Kaggle's 560×280 target crop so the auto-resize produces
a crisp cover).

Usage:
    python -m scripts.generate_cover_image
    python -m scripts.generate_cover_image --output custom_cover.png
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_DIR = PROJECT_ROOT / "data" / "labeled_dataset"
SOURCE_IMAGES_DIR = PROJECT_ROOT / "Examples" / "aug"
DEFAULT_OUTPUT = DATASET_DIR / "cover.png"

# Kaggle crops the cover to 560x280 (16:8 aspect). Render at 2x
# resolution so the auto-resize gives a crisp thumbnail. matplotlib
# figsize is in inches at dpi=100 → (11.2, 5.6) = 1120x560 pixels.
FIG_W_IN = 11.2
FIG_H_IN = 5.6
DPI = 100

# Sample selection criteria
N_SAMPLES = 16          # 4x4 grid
GRID_ROWS = 4
GRID_COLS = 4
FILLED_MIN = 20         # typical Sudoku has 17-35 clue cells
FILLED_MAX = 32
SEED = 42


def load_records() -> List[Dict[str, Any]]:
    path = DATASET_DIR / "data.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run "
            f"`python -m scripts.build_labeled_dataset` first"
        )
    with path.open() as f:
        return [json.loads(line) for line in f]


def pick_happy_path_samples(
    records: List[Dict[str, Any]],
    n: int = N_SAMPLES,
) -> List[Dict[str, Any]]:
    """Select `n` solvable, high-confidence, non-GT images.

    Priority: solvable + detectable + filled-count in [FILLED_MIN,
    FILLED_MAX] + NOT in the GT benchmark subset. Tie-break by mean
    confidence descending so the cover shows the cleanest examples.
    """
    candidates = [
        r for r in records
        if r["detectable"]
        and r["solvable"]
        and not r["has_ground_truth_benchmark"]
        and r["best_guess_filled_count"] is not None
        and FILLED_MIN <= r["best_guess_filled_count"] <= FILLED_MAX
    ]

    def mean_confidence(rec: Dict[str, Any]) -> float:
        conf = rec.get("best_guess_confidence")
        if conf is None:
            return 0.0
        flat = [c for row in conf for c in row]
        return float(np.mean(flat)) if flat else 0.0

    # Sort by mean confidence descending so the happiest paths rise
    candidates.sort(key=mean_confidence, reverse=True)

    # Take the top 3*n by confidence, then seed-pick n of them so the
    # selection is deterministic AND doesn't over-fit to the very top
    # (which tend to be near-empty grids that look boring on a cover).
    top_pool = candidates[: max(n * 3, n)]
    if len(top_pool) < n:
        return top_pool

    random.seed(SEED)
    return random.sample(top_pool, n)


def render_cover(
    samples: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Render the 4x4 grid and save as PNG."""
    fig, axes = plt.subplots(
        GRID_ROWS,
        GRID_COLS,
        figsize=(FIG_W_IN, FIG_H_IN),
        facecolor="white",
    )
    axes = np.asarray(axes).reshape(-1)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax, rec in zip(axes, samples):
        img_path = SOURCE_IMAGES_DIR / rec["filename"]
        if not img_path.exists():
            ax.text(
                0.5, 0.5, "(missing)",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            continue

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img, aspect="equal")

        corners = rec.get("detected_corners_4")
        if corners is not None:
            # Close the polygon by appending the first point
            cs = np.array(corners + [corners[0]], dtype=float)
            ax.plot(
                cs[:, 0], cs[:, 1],
                color="#39FF14",     # neon green for visibility
                linewidth=2.2,
                solid_capstyle="round",
                solid_joinstyle="round",
            )

    # Fill any unused axes (if we had fewer samples than slots)
    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.subplots_adjust(
        left=0.005, right=0.995,
        top=0.995, bottom=0.005,
        wspace=0.02, hspace=0.02,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(output_path),
        dpi=DPI,
        facecolor="white",
        bbox_inches=None,    # preserve the manual subplots_adjust
        pad_inches=0,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Kaggle cover image for labeled dataset"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    print(f"Loading records from {DATASET_DIR / 'data.jsonl'}")
    records = load_records()
    print(f"  {len(records)} total records")

    samples = pick_happy_path_samples(records, N_SAMPLES)
    print(f"Selected {len(samples)} happy-path samples for cover:")
    for rec in samples:
        fc = rec["best_guess_filled_count"]
        conf = rec.get("best_guess_confidence")
        mean_c = (
            float(np.mean([c for row in conf for c in row]))
            if conf else 0.0
        )
        print(f"  {rec['filename']:<26s}  "
              f"filled {fc:>3d}  mean_conf {mean_c:.3f}")

    print(f"\nRendering {GRID_ROWS}x{GRID_COLS} cover to {args.output}")
    render_cover(samples, args.output)

    size = args.output.stat().st_size
    print(
        f"  wrote {args.output} "
        f"({size / 1024:.1f} KB, {int(FIG_W_IN * DPI)}x"
        f"{int(FIG_H_IN * DPI)} px)"
    )
    print()
    print("Kaggle will auto-crop to 560x280 cover + 280x280 thumbnail.")


if __name__ == "__main__":
    main()
