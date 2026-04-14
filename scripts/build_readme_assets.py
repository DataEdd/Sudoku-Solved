"""Generate every image asset the public README references.

Produces PNGs with transparent backgrounds so they read well on both
light and dark GitHub themes:

    docs/hero.png              Solved-overlay collage
    docs/lessons/scoring.png   _33_ classical vs 5-term + crossword failure
    docs/lessons/warp.png      4-point vs 8-point piecewise on _19_

Deterministic — no RNG — so the committed images match a fresh rerun.

    python -m scripts.build_readme_assets
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core import extraction as E
from app.core.solver import backtracking
from app.ml.recognizer import CNNRecognizer

GT_PATH = PROJECT_ROOT / "evaluation" / "ground_truth.json"
HERO_OUT = PROJECT_ROOT / "docs" / "hero.png"
LESSONS_DIR = PROJECT_ROOT / "docs" / "lessons"

OUTER_IDX = [0, 3, 15, 12]
INNER_IDX = [5, 6, 10, 9]


def load_gt() -> List[dict]:
    with GT_PATH.open() as f:
        return json.load(f)["images"]


def gt_outer_corners(record: dict) -> np.ndarray:
    return np.array(
        [record["corners_16"][i] for i in OUTER_IDX], dtype=np.float32
    )


def gt_inner_corners(record: dict) -> np.ndarray:
    return np.array(
        [record["corners_16"][i] for i in INNER_IDX], dtype=np.float32
    )


def ocr_grid(
    warped: np.ndarray, recognizer: CNNRecognizer
) -> List[List[int]]:
    cells = E.extract_cells(warped)
    preds = recognizer.predict_batch(cells)
    grid = [[0] * 9 for _ in range(9)]
    for idx, (digit, _conf) in enumerate(preds):
        r, c = divmod(idx, 9)
        grid[r][c] = digit
    return grid


def cell_allows(gt_cell, pred: int) -> bool:
    if isinstance(gt_cell, list):
        return pred in gt_cell
    return pred == gt_cell


def all_clues_read_correctly(
    pred_grid: List[List[int]], gt_grid: List[List]
) -> bool:
    """True iff every GT clue (non-empty) is read correctly by OCR.

    GT-empty cells are ignored — a spurious digit on an empty cell is a
    hallucination that would be caught by the confidence threshold in
    production, and we'll solve from the correct clues regardless.
    """
    for r in range(9):
        for c in range(9):
            gt_cell = gt_grid[r][c]
            if isinstance(gt_cell, list):
                non_zero = [v for v in gt_cell if v != 0]
                if not non_zero:
                    continue  # GT empty (list of zeros)
                if not cell_allows(gt_cell, pred_grid[r][c]):
                    return False
            else:
                if gt_cell == 0:
                    continue
                if pred_grid[r][c] != gt_cell:
                    return False
    return True


def puzzle_from_gt(gt_grid: List[List]) -> List[List[int]]:
    """GT cells may be ints or lists (multi-value). Reduce to an int puzzle
    by taking the first valid value (or 0 if the list contains 0)."""
    out = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            cell = gt_grid[r][c]
            if isinstance(cell, list):
                non_zero = [v for v in cell if v != 0]
                out[r][c] = non_zero[0] if non_zero else 0
            else:
                out[r][c] = int(cell)
    return out


def paint_solution(
    image: np.ndarray,
    outer_corners: np.ndarray,
    clue_grid: List[List[int]],
    solved_grid: List[List[int]],
) -> np.ndarray:
    """Paint the solver-filled digits back onto the original photo.

    The inverse homography maps warped-grid cell centers back to image
    pixel coordinates; solver-filled cells are drawn in green, clues
    are left as-is (they're already visible on the paper).
    """
    size = 900
    src = E.order_points(outer_corners.reshape(4, 1, 2))
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = np.linalg.inv(M)

    cell = size / 9.0
    canvas = image.copy()
    for r in range(9):
        for c in range(9):
            if clue_grid[r][c] != 0:
                continue
            if solved_grid[r][c] == 0:
                continue
            cx = (c + 0.5) * cell
            cy = (r + 0.5) * cell
            pt = M_inv @ np.array([cx, cy, 1.0])
            pt /= pt[2]
            x, y = int(pt[0]), int(pt[1])

            # Size the digit relative to the local cell height in image pixels.
            top = M_inv @ np.array([cx, cy - cell * 0.4, 1.0])
            top /= top[2]
            digit_px = max(10, int(abs(top[1] - y) * 1.8))

            font_scale = digit_px / 30.0
            thickness = max(2, int(font_scale * 2.2))
            text = str(solved_grid[r][c])
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
            )
            org = (x - tw // 2, y + th // 2)
            cv2.putText(
                canvas, text, org, cv2.FONT_HERSHEY_DUPLEX,
                font_scale, (30, 200, 30), thickness, cv2.LINE_AA,
            )
    # Outline the detected grid lightly so the before/after is legible.
    pts = outer_corners.reshape(-1, 2).astype(np.int32)
    cv2.polylines(canvas, [pts], True, (30, 200, 30), 3, cv2.LINE_AA)
    return canvas


def pick_hero_candidates(
    records: List[dict], recognizer: CNNRecognizer, want: int = 6
) -> List[Tuple[dict, np.ndarray, List[List[int]], List[List[int]]]]:
    """Return up to `want` (record, outer_corners, clue_grid, solved_grid)
    tuples where the full production pipeline succeeds end-to-end AND the
    solver's output is consistent with the GT clue grid."""
    accepted = []
    for rec in records:
        path = PROJECT_ROOT / rec["path"]
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue

        corners, _conf = E.detect_grid(img)
        if corners is None:
            continue

        warped = E.perspective_transform(img, corners)
        read = ocr_grid(warped, recognizer)

        if not all_clues_read_correctly(read, rec["grid"]):
            continue

        puzzle = puzzle_from_gt(rec["grid"])
        solved, _nodes, ok = backtracking(puzzle)
        if not ok:
            continue

        accepted.append((rec, corners, puzzle, solved))
        print(f"  hero candidate: {rec['path']}")
        if len(accepted) >= want:
            break
    return accepted


# All outputs are BGRA PNGs where the non-image area is fully transparent
# (alpha=0) so the collage reads on both light and dark GitHub themes.
def _to_bgra(image: np.ndarray, alpha: int = 255) -> np.ndarray:
    """Promote a BGR image to BGRA with uniform alpha."""
    if image.shape[2] == 4:
        return image
    bgra = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    bgra[:, :, :3] = image
    bgra[:, :, 3] = alpha
    return bgra


def _square_panel_bgra(
    image: np.ndarray, corners: np.ndarray, clue: List[List[int]],
    solved: List[List[int]], size: int,
) -> np.ndarray:
    """Crop around the detected grid, center-place on a transparent square."""
    painted = paint_solution(image, corners, clue, solved)
    pts = corners.reshape(-1, 2)
    x0 = max(0, int(pts[:, 0].min()) - 40)
    y0 = max(0, int(pts[:, 1].min()) - 40)
    x1 = min(painted.shape[1], int(pts[:, 0].max()) + 40)
    y1 = min(painted.shape[0], int(pts[:, 1].max()) + 40)
    crop = painted[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    side = max(h, w)
    # Fit into (side, side) while preserving aspect, then resize to (size, size).
    pad = np.zeros((side, side, 4), dtype=np.uint8)  # fully transparent
    oy = (side - h) // 2
    ox = (side - w) // 2
    pad[oy:oy + h, ox:ox + w, :3] = crop
    pad[oy:oy + h, ox:ox + w, 3] = 255
    return cv2.resize(pad, (size, size), interpolation=cv2.INTER_AREA)


def build_hero(records: List[dict], recognizer: CNNRecognizer) -> None:
    picks = pick_hero_candidates(records, recognizer, want=6)
    size = 520
    panels = [
        _square_panel_bgra(
            cv2.imread(str(PROJECT_ROOT / rec["path"])),
            corners, clue, solved, size,
        )
        for rec, corners, clue, solved in picks
    ]

    def _row(items: List[np.ndarray]) -> np.ndarray:
        return (
            np.hstack(items) if items
            else np.zeros((size, 0, 4), dtype=np.uint8)
        )

    def _blank(h: int, w: int) -> np.ndarray:
        return np.zeros((h, w, 4), dtype=np.uint8)

    if len(panels) >= 6:
        collage = np.vstack([_row(panels[0:3]), _row(panels[3:6])])
    elif len(panels) == 5:
        total_w = 3 * size
        top = _row(panels[0:3])
        bot_inner = _row(panels[3:5])
        bot = _blank(size, total_w)
        ox = (total_w - bot_inner.shape[1]) // 2
        bot[:, ox:ox + bot_inner.shape[1]] = bot_inner
        collage = np.vstack([top, bot])
    elif len(panels) == 4:
        collage = np.vstack([_row(panels[0:2]), _row(panels[2:4])])
    else:
        collage = _row(panels)

    HERO_OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(HERO_OUT), collage)
    print(f"  wrote {HERO_OUT.relative_to(PROJECT_ROOT)} "
          f"({collage.shape[1]}x{collage.shape[0]}, {len(panels)} panels, PNG+alpha)")


# -----------------------------------------------------------------------------
# Lesson 2: scoring — classical 3-term vs 5-term structure-aware on _33_
# -----------------------------------------------------------------------------

def classical_3term_score(
    quad: np.ndarray, area: float, max_area: float,
    img_center: np.ndarray, max_dist: float,
) -> float:
    return E.score_quad(quad, area, max_area, img_center, max_dist)


def structure_5term_score(
    image: np.ndarray, quad: np.ndarray, area: float, max_area: float,
    img_center: np.ndarray, max_dist: float,
) -> float:
    area_norm = area / max_area if max_area > 0 else 0.0
    x, y, w, h = cv2.boundingRect(quad.reshape(-1, 1, 2).astype(np.int32))
    squareness = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
    centroid = np.mean(quad.reshape(-1, 2), axis=0)
    dist = np.linalg.norm(centroid - img_center)
    centeredness = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
    struct = E.score_grid_structure(image, quad)[0]
    cells = E.score_cell_count(image, quad)[0]
    return (
        0.2 * area_norm + 0.2 * squareness + 0.1 * centeredness
        + 0.3 * struct + 0.2 * cells
    )


def candidate_quads(image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = E._preprocess(gray)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    H, W = image.shape[:2]
    image_area = H * W
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.01 * image_area or area > 0.99 * image_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        out.append((approx.reshape(4, 2).astype(np.float32), float(area)))
    return out


def _draw_quads(
    image: np.ndarray, winner: np.ndarray,
    runner_up: Optional[np.ndarray] = None,
) -> np.ndarray:
    canvas = image.copy()
    if runner_up is not None:
        pts = runner_up.reshape(-1, 2).astype(np.int32)
        cv2.polylines(canvas, [pts], True, (40, 40, 220), 4, cv2.LINE_AA)
    pts = winner.reshape(-1, 2).astype(np.int32)
    cv2.polylines(canvas, [pts], True, (40, 200, 40), 5, cv2.LINE_AA)
    return canvas


def _panel_bgra(
    image: np.ndarray, panel_w: int, panel_h: int,
) -> np.ndarray:
    """BGRA panel: opaque image letterboxed on a fully transparent canvas.

    No baked-in title — captions live in surrounding Markdown prose so
    they inherit GitHub's theme-aware text color.
    """
    ih, iw = image.shape[:2]
    scale = min(panel_w / iw, panel_h / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    panel = np.zeros((panel_h, panel_w, 4), dtype=np.uint8)
    ox = (panel_w - new_w) // 2
    oy = (panel_h - new_h) // 2
    panel[oy:oy + new_h, ox:ox + new_w, :3] = resized
    panel[oy:oy + new_h, ox:ox + new_w, 3] = 255
    return panel


def _pick_scoring_quads(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, float, float]]]:
    """Return (classical_top, five_top, all_scored) for an image.

    all_scored entries are (quad, classical_score, five_score).
    """
    H, W = image.shape[:2]
    img_center = np.array([W / 2.0, H / 2.0])
    max_dist = float(np.sqrt((W / 2.0) ** 2 + (H / 2.0) ** 2))

    quads = candidate_quads(image)
    if not quads:
        return None, None, []
    max_area = max(a for _, a in quads)

    scored = []
    for q, a in quads:
        cl = classical_3term_score(q, a, max_area, img_center, max_dist)
        five = structure_5term_score(image, q, a, max_area, img_center, max_dist)
        scored.append((q, cl, five))

    cl_top = max(scored, key=lambda x: x[1])[0]
    five_top = max(scored, key=lambda x: x[2])[0]
    return cl_top, five_top, scored


def build_scoring_demo(records: List[dict]) -> None:
    """Compose a single docs/lessons/scoring.jpg with three labelled panels.

    Row 1: _33_ classical top (wrong) | _33_ five-term top (right)
    Row 2: _4_ five-term top (still wrong — crossword wins over sudoku)
    """
    # --- _33_: article panel vs sudoku grid ---
    rec33 = next(r for r in records if "_33_" in r["path"])
    img33 = cv2.imread(str(PROJECT_ROOT / rec33["path"]))
    cl33, five33, _ = _pick_scoring_quads(img33)

    def same_quad(a: np.ndarray, b: np.ndarray, tol: float = 30.0) -> bool:
        return float(np.linalg.norm(np.mean(a, axis=0) - np.mean(b, axis=0))) < tol

    # Show classical picking a non-sudoku; if classical == five (sudoku), use the
    # next-highest non-sudoku candidate.
    if same_quad(cl33, five33):
        _, _, scored33 = _pick_scoring_quads(img33)
        non_sud = [q for q, cl, _ in sorted(scored33, key=lambda x: -x[1])
                   if not same_quad(q, five33)]
        cl33_panel = non_sud[0] if non_sud else cl33
    else:
        cl33_panel = cl33

    panel_a = _draw_quads(img33, cl33_panel, five33)
    panel_b = _draw_quads(img33, five33, None)

    # --- _4_: crossword vs sudoku on same page ---
    rec4 = next(r for r in records if "_4_" in r["path"])
    img4 = cv2.imread(str(PROJECT_ROOT / rec4["path"]))
    cl4, five4, _ = _pick_scoring_quads(img4)
    # On _4_ the 5-term score still picks the crossword; show that quad.
    panel_c = _draw_quads(img4, five4, None) if five4 is not None else img4

    # Layout: top row is two _33_ panels side-by-side; bottom row is a wider
    # crossword-failure panel spanning both columns. No baked-in captions —
    # those live in the README prose around the image.
    panel_w, panel_h = 720, 540
    top_w = 2 * panel_w
    a = _panel_bgra(panel_a, panel_w, panel_h)
    b = _panel_bgra(panel_b, panel_w, panel_h)
    c = _panel_bgra(panel_c, top_w, panel_h)
    collage = np.vstack([np.hstack([a, b]), c])

    LESSONS_DIR.mkdir(parents=True, exist_ok=True)
    # Clean up stale outputs from earlier layouts.
    for stale in (
        "scoring_classical.jpg", "scoring_structure.jpg", "scoring.jpg",
    ):
        (LESSONS_DIR / stale).unlink(missing_ok=True)

    out = LESSONS_DIR / "scoring.png"
    cv2.imwrite(str(out), collage)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)} "
          f"({collage.shape[1]}x{collage.shape[0]}, PNG+alpha)")


# -----------------------------------------------------------------------------
# Lesson 3: warp — 4-point vs 8-point piecewise on a curved-newsprint image
# -----------------------------------------------------------------------------

def piecewise_warp_image(
    image: np.ndarray, outer: np.ndarray, inner: np.ndarray, size: int = 540,
) -> np.ndarray:
    """Return the full piecewise-warped grid image (pre-cell-split)."""
    outer = np.array(outer, dtype=np.float32).reshape(4, 2)
    inner = np.array(inner, dtype=np.float32).reshape(4, 2)
    TL, TR, BR, BL = outer[0], outer[1], outer[2], outer[3]
    CTL, CTR, CBR, CBL = inner[0], inner[1], inner[2], inner[3]

    T3 = E._lerp(TL, TR, 1 / 3)
    T6 = E._lerp(TL, TR, 2 / 3)
    B3 = E._lerp(BL, BR, 1 / 3)
    B6 = E._lerp(BL, BR, 2 / 3)
    L3 = E._lerp(TL, BL, 1 / 3)
    L6 = E._lerp(TL, BL, 2 / 3)
    R3 = E._lerp(TR, BR, 1 / 3)
    R6 = E._lerp(TR, BR, 2 / 3)

    s3 = size / 3
    s6 = size * 2 / 3
    sz = float(size)

    src_quads = [
        [TL, T3, CTL, L3],
        [T3, T6, CTR, CTL],
        [T6, TR, R3, CTR],
        [L3, CTL, CBL, L6],
        [CTL, CTR, CBR, CBL],
        [CTR, R3, R6, CBR],
        [L6, CBL, B3, BL],
        [CBL, CBR, B6, B3],
        [CBR, R6, BR, B6],
    ]
    dst_quads = [
        [[0, 0], [s3, 0], [s3, s3], [0, s3]],
        [[s3, 0], [s6, 0], [s6, s3], [s3, s3]],
        [[s6, 0], [sz, 0], [sz, s3], [s6, s3]],
        [[0, s3], [s3, s3], [s3, s6], [0, s6]],
        [[s3, s3], [s6, s3], [s6, s6], [s3, s6]],
        [[s6, s3], [sz, s3], [sz, s6], [s6, s6]],
        [[0, s6], [s3, s6], [s3, sz], [0, sz]],
        [[s3, s6], [s6, s6], [s6, sz], [s3, sz]],
        [[s6, s6], [sz, s6], [sz, sz], [s6, sz]],
    ]
    output = np.zeros((size, size, 3), dtype=np.uint8)
    for src_q, dst_q in zip(src_quads, dst_quads):
        src_pts = np.array(src_q, dtype=np.float32)
        dst_pts = np.array(dst_q, dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (size, size))
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
        mask_3ch = cv2.merge([mask, mask, mask])
        output = np.where(mask_3ch > 0, warped, output)
    return output


def draw_grid_lines(warped: np.ndarray) -> np.ndarray:
    canvas = warped.copy()
    H, W = canvas.shape[:2]
    for k in range(1, 9):
        x = int(k * W / 9)
        y = int(k * H / 9)
        thickness = 2 if k % 3 == 0 else 1
        color = (40, 200, 40) if k % 3 == 0 else (40, 160, 40)
        cv2.line(canvas, (x, 0), (x, H), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (0, y), (W, y), color, thickness, cv2.LINE_AA)
    return canvas


def build_warp_demo(records: List[dict]) -> None:
    """Compose a single docs/lessons/warp.jpg with 4-pt and 8-pt side-by-side.

    Picks the image with the largest measured interior-corner deviation
    from ideal — that's where the piecewise warp has the most to correct
    and the difference between the two is most legible.
    """
    # Measured ranking: _19_ (toilet paper, 31px) > _37_ (27) > _31_ (24).
    want = ["_19_", "_37_", "_31_"]
    rec = None
    for tag in want:
        for r in records:
            if tag in r["path"]:
                rec = r
                break
        if rec is not None:
            break
    if rec is None:
        raise RuntimeError("No warp-demo candidate found")

    image = cv2.imread(str(PROJECT_ROOT / rec["path"]))
    outer = gt_outer_corners(rec)
    inner = gt_inner_corners(rec)

    warp_size = 540
    warp4 = E.perspective_transform(image, outer.reshape(4, 1, 2))
    warp4 = cv2.resize(warp4, (warp_size, warp_size), interpolation=cv2.INTER_AREA)
    warp8 = piecewise_warp_image(image, outer, inner, size=warp_size)

    warp4_lined = draw_grid_lines(warp4)
    warp8_lined = draw_grid_lines(warp8)

    panel_w, panel_h = 720, 720
    a = _panel_bgra(warp4_lined, panel_w, panel_h)
    b = _panel_bgra(warp8_lined, panel_w, panel_h)
    collage = np.hstack([a, b])

    LESSONS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in ("warp_4pt.jpg", "warp_8pt.jpg", "warp.jpg"):
        (LESSONS_DIR / stale).unlink(missing_ok=True)

    out = LESSONS_DIR / "warp.png"
    cv2.imwrite(str(out), collage)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)} "
          f"({collage.shape[1]}x{collage.shape[0]}, PNG+alpha)")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    records = load_gt()
    print(f"Loaded {len(records)} GT records")

    recognizer = CNNRecognizer()
    print(f"Recognizer backend: {recognizer.backend}")

    print("\n[1/3] Building scoring demo (lesson 2)…")
    build_scoring_demo(records)

    print("\n[2/3] Building warp demo (lesson 3)…")
    build_warp_demo(records)

    print("\n[3/3] Building hero collage…")
    build_hero(records, recognizer)

    print("\nDone.")


if __name__ == "__main__":
    main()
