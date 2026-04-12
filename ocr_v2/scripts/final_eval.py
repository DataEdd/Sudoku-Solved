"""GATED — official OCR v2 benchmark on Wicht V2 test set.

This script is the only thing in `ocr_v2/` that touches the test set.
The OCR v2 development agent must NOT run it. It is run by the user
in the parent repo, AFTER the agent has handed over a finalized model.

What it does:

1. Loads `ocr_v2/src/infer.py::recognize_cells` (the agent's final
   model interface — same shape as the parent project's recognizer).
2. Reads `research/wichtounet_dataset/datasets/v2_test.desc` (40
   images), the per-image .dat ground truth, and the 4-point
   outlines from `outlines_sorted.csv`.
3. For each test image: warps via the outline corners (same
   geometry the agent's prep script used for training), slices
   into 81 cells, hands them to `recognize_cells`, scores against
   the .dat GT.
4. Reports both metric families:
   - DataEdd format: filled / empty / wrong / missed / hallucinated
   - Wicht format: perfect-image rate, per-phone breakdown
5. Writes `ocr_v2/results/final_eval.json` with per-image + summary.
6. Writes `ocr_v2/results/final_eval.md` with a side-by-side
   comparison vs the v5.1 numbers from
   `docs/internal/wicht_test_results.json`.

Run from the parent repo:

    python -m ocr_v2.scripts.final_eval

Pre-requisite: the v2 model must exist at
`ocr_v2/checkpoints/v2.{pth,onnx}` and `ocr_v2/src/infer.py` must
expose a working `recognize_cells(cells)` function.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.extraction import perspective_transform  # noqa: E402

WICHT_ROOT = PROJECT_ROOT / "research" / "wichtounet_dataset"
TEST_MANIFEST = WICHT_ROOT / "datasets" / "v2_test.desc"
OUTLINES_CSV = WICHT_ROOT / "outlines_sorted.csv"
IMAGES_DIR = WICHT_ROOT / "images"

OCR_V2_ROOT = PROJECT_ROOT / "ocr_v2"
INFER_PATH = OCR_V2_ROOT / "src" / "infer.py"
RESULTS_DIR = OCR_V2_ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "final_eval.json"
RESULTS_MD = RESULTS_DIR / "final_eval.md"

# Match prep_training_data.py
WARP_SIZE = 450

# Reference v5.1 numbers from docs/internal/wicht_test_results.json on
# the same 40-image test set, for the side-by-side comparison.
V51_REFERENCE = {
    "filled_acc": 0.585,        # 676/1156
    "empty_acc": 0.992,         # 2067/2084
    "perfect_images": 5,        # out of 40 (includes the 2 detection failures)
    "wrong": 203,
    "missed": 277,
    "halluc": 17,
    "detection_rate": 1.0,      # detect_grid returned a quad on 40/40
}


def load_v2_recognizer():
    """Dynamically import the agent's recognize_cells from src/infer.py."""
    if not INFER_PATH.exists():
        raise SystemExit(
            f"{INFER_PATH} not found — the OCR v2 agent has not delivered "
            f"a final model yet. This script is gated and only runs after "
            f"the agent finishes."
        )
    spec = importlib.util.spec_from_file_location("ocr_v2_infer", INFER_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {INFER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "recognize_cells"):
        raise SystemExit(
            f"{INFER_PATH} must expose a `recognize_cells(cells)` function "
            f"matching the parent-project recognizer interface."
        )
    return module.recognize_cells


def load_test_names() -> List[str]:
    return [
        Path(line.strip()).name
        for line in TEST_MANIFEST.read_text().splitlines()
        if line.strip()
    ]


def load_outlines() -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with OUTLINES_CSV.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row:
                continue
            name = Path(row[0]).name
            try:
                coords = [float(x) for x in row[1:9]]
            except (ValueError, IndexError):
                continue
            out[name] = np.array(coords, dtype=np.float32).reshape(4, 2)
    return out


def parse_dat(path: Path) -> Tuple[str, str, List[List[int]]]:
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


def warp_image(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    contour = corners.reshape(4, 1, 2).astype(np.float32)
    warped = perspective_transform(image, contour)
    return cv2.resize(warped, (WARP_SIZE, WARP_SIZE), interpolation=cv2.INTER_AREA)


def slice_cells(warped: np.ndarray) -> List[np.ndarray]:
    h, w = warped.shape[:2]
    ch, cw = h // 9, w // 9
    return [
        warped[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw]
        for r in range(9)
        for c in range(9)
    ]


def score(
    pred_grid: List[List[int]], gt_grid: List[List[int]]
) -> Dict[str, Any]:
    stats = dict(
        filled_total=0, filled_correct=0,
        empty_total=0, empty_correct=0,
        wrong=0, missed=0, halluc=0, perfect=True,
    )
    for i in range(9):
        for j in range(9):
            g, p = gt_grid[i][j], pred_grid[i][j]
            if g != 0:
                stats["filled_total"] += 1
                if p == g:
                    stats["filled_correct"] += 1
                elif p == 0:
                    stats["missed"] += 1
                    stats["perfect"] = False
                else:
                    stats["wrong"] += 1
                    stats["perfect"] = False
            else:
                stats["empty_total"] += 1
                if p == 0:
                    stats["empty_correct"] += 1
                else:
                    stats["halluc"] += 1
                    stats["perfect"] = False
    return stats


def main() -> None:
    print("=" * 70)
    print(" OCR v2 — GATED final eval on Wicht V2 test set")
    print("=" * 70)
    print()

    recognize_cells = load_v2_recognizer()
    print(f"Loaded recognize_cells from {INFER_PATH}")

    test_names = load_test_names()
    outlines = load_outlines()
    print(f"Test images: {len(test_names)}")
    print(f"Outlines available: {sum(1 for n in test_names if n in outlines)}")
    print()

    per_image = []
    for i, name in enumerate(test_names, 1):
        img_path = IMAGES_DIR / name
        dat_path = IMAGES_DIR / name.replace(".jpg", ".dat")
        if not img_path.exists() or not dat_path.exists():
            print(f"  SKIP {name} — missing file")
            continue
        if name not in outlines:
            print(f"  SKIP {name} — no outline (cannot warp)")
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  SKIP {name} — cv2 failed to load")
            continue

        phone, resolution, gt_grid = parse_dat(dat_path)
        warped = warp_image(image, outlines[name])
        cells = slice_cells(warped)

        result = recognize_cells(cells)
        # Tolerate either (grid, conf) or just grid
        if isinstance(result, tuple) and len(result) == 2:
            pred_grid, _conf = result
        else:
            pred_grid = result
        pred_grid = [[int(v) for v in row] for row in pred_grid]

        stats = score(pred_grid, gt_grid)
        per_image.append({
            "filename": name,
            "phone": phone,
            "resolution": resolution,
            "gt_grid": gt_grid,
            "pred_grid": pred_grid,
            "stats": stats,
        })
        print(
            f"  [{i:>2d}/{len(test_names)}] {name:<18s}  "
            f"filled {stats['filled_correct']}/{stats['filled_total']}  "
            f"perfect={stats['perfect']}",
            flush=True,
        )

    # Aggregate
    n = len(per_image)
    n_filled_total = sum(r["stats"]["filled_total"] for r in per_image)
    n_filled_correct = sum(r["stats"]["filled_correct"] for r in per_image)
    n_empty_total = sum(r["stats"]["empty_total"] for r in per_image)
    n_empty_correct = sum(r["stats"]["empty_correct"] for r in per_image)
    n_wrong = sum(r["stats"]["wrong"] for r in per_image)
    n_missed = sum(r["stats"]["missed"] for r in per_image)
    n_halluc = sum(r["stats"]["halluc"] for r in per_image)
    n_perfect = sum(1 for r in per_image if r["stats"]["perfect"])

    summary = {
        "test_set": "wichtounet/sudoku_dataset V2 (40 images)",
        "images_scored": n,
        "filled_total": n_filled_total,
        "filled_correct": n_filled_correct,
        "filled_acc": round(n_filled_correct / max(n_filled_total, 1), 4),
        "empty_total": n_empty_total,
        "empty_correct": n_empty_correct,
        "empty_acc": round(n_empty_correct / max(n_empty_total, 1), 4),
        "wrong": n_wrong,
        "missed": n_missed,
        "halluc": n_halluc,
        "perfect_images": n_perfect,
        "perfect_rate": round(n_perfect / max(n, 1), 4),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(
        json.dumps({"summary": summary, "per_image": per_image}, indent=2)
    )

    print()
    print("=" * 70)
    print(" v2 vs v5.1 on the same V2 test set")
    print("=" * 70)
    print(f"{'Metric':<28s}  {'v5.1':>10s}  {'v2':>10s}  {'Δ':>8s}")
    print("-" * 60)

    def row(label: str, v51: float, v2: float, fmt: str = "{:.1%}") -> None:
        delta = v2 - v51
        print(f"{label:<28s}  {fmt.format(v51):>10s}  {fmt.format(v2):>10s}  "
              f"{('+' + fmt.format(delta)) if delta >= 0 else fmt.format(delta):>8s}")

    row("Filled-cell accuracy", V51_REFERENCE["filled_acc"], summary["filled_acc"])
    row("Empty-cell accuracy", V51_REFERENCE["empty_acc"], summary["empty_acc"])
    row("Perfect-image rate", V51_REFERENCE["perfect_images"] / 40,
        summary["perfect_rate"])
    print(f"{'Wrong (filled→other)':<28s}  {V51_REFERENCE['wrong']:>10d}  {n_wrong:>10d}  {n_wrong - V51_REFERENCE['wrong']:>+8d}")
    print(f"{'Missed (filled→empty)':<28s}  {V51_REFERENCE['missed']:>10d}  {n_missed:>10d}  {n_missed - V51_REFERENCE['missed']:>+8d}")
    print(f"{'Hallucinated (empty→digit)':<28s}  {V51_REFERENCE['halluc']:>10d}  {n_halluc:>10d}  {n_halluc - V51_REFERENCE['halluc']:>+8d}")

    # Markdown report
    lines = [
        "# OCR v2 — final evaluation on Wicht V2 test set",
        "",
        "Run via `python -m ocr_v2.scripts.final_eval` after the v2 agent",
        "delivers a finalized model. v5.1 reference numbers come from",
        "`docs/internal/wicht_test_results.json` (the parent project's",
        "zero-shot baseline on the same 40-image set).",
        "",
        "## Side-by-side",
        "",
        "| Metric | v5.1 (zero-shot) | v2 (in-distribution) | Δ |",
        "|---|---:|---:|---:|",
        f"| Filled-cell accuracy | {100 * V51_REFERENCE['filled_acc']:.1f}% | {100 * summary['filled_acc']:.1f}% | "
        f"{100 * (summary['filled_acc'] - V51_REFERENCE['filled_acc']):+.1f}% |",
        f"| Empty-cell accuracy | {100 * V51_REFERENCE['empty_acc']:.1f}% | {100 * summary['empty_acc']:.1f}% | "
        f"{100 * (summary['empty_acc'] - V51_REFERENCE['empty_acc']):+.1f}% |",
        f"| Perfect-image rate | {V51_REFERENCE['perfect_images']}/40 ({100 * V51_REFERENCE['perfect_images'] / 40:.1f}%) | "
        f"{n_perfect}/{n} ({100 * summary['perfect_rate']:.1f}%) | "
        f"{n_perfect - V51_REFERENCE['perfect_images']:+d} |",
        f"| Wrong (filled→other) | {V51_REFERENCE['wrong']} | {n_wrong} | {n_wrong - V51_REFERENCE['wrong']:+d} |",
        f"| Missed (filled→empty) | {V51_REFERENCE['missed']} | {n_missed} | {n_missed - V51_REFERENCE['missed']:+d} |",
        f"| Hallucinated (empty→digit) | {V51_REFERENCE['halluc']} | {n_halluc} | {n_halluc - V51_REFERENCE['halluc']:+d} |",
        "",
        "## Hypothesis verdict",
        "",
        "The hypothesis being tested: **in-distribution training matters more",
        "than architecture choice.** v5.1 was trained on synthetic data; v2",
        "trains from scratch on Wicht's V2 training set.",
        "",
        f"- v5.1 filled-cell: {100 * V51_REFERENCE['filled_acc']:.1f}%",
        f"- v2 filled-cell:   {100 * summary['filled_acc']:.1f}%",
        f"- Delta:            {100 * (summary['filled_acc'] - V51_REFERENCE['filled_acc']):+.1f} pts",
        "",
        ("**HYPOTHESIS CONFIRMED** — in-distribution training improves "
         "filled-cell accuracy on the V2 test set."
         if summary["filled_acc"] > V51_REFERENCE["filled_acc"]
         else "**HYPOTHESIS REFUTED** — even with in-distribution training, "
              "v2 does not beat v5.1 on filled-cell accuracy. Either the "
              "architecture is the bottleneck after all, or the v2 training "
              "approach has a different problem worth investigating."),
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines))

    print()
    print(f"Wrote {RESULTS_JSON}")
    print(f"Wrote {RESULTS_MD}")


if __name__ == "__main__":
    main()
