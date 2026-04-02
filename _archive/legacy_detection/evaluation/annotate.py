"""
Interactive ground truth annotation tool (v2).

16-point corner annotation + multi-digit cell support.

Opens each image in an OpenCV window. Click 16 points — all intersections
of the thick grid lines (outer border + box separators), row by row:

    Row 0 (top):    P0  P1  P2  P3
    Row 3:          P4  P5  P6  P7
    Row 6:          P8  P9  P10 P11
    Row 9 (bottom): P12 P13 P14 P15

Then type the 9x9 grid in the terminal. Supports multi-digit cells (e.g. 1/7).
Saves to ground_truth_annotated.json.

Usage:
    cd Sudoku-Solved/
    python _archive/legacy_detection/evaluation/annotate.py

Controls:
    Left click  — place next point
    'u'         — undo last point
    'r'         — reset all points
    's'         — skip image
    'q'         — quit (progress is saved)
    Enter       — confirm all 16 points and proceed to grid entry
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
os.chdir(PROJECT_ROOT)

IMAGE_DIR = "Examples/Ground Example"
OUTPUT_PATH = "_archive/legacy_detection/evaluation/ground_truth_annotated.json"

# 16 point labels: row-major P0..P15
POINT_LABELS = [f"P{i}" for i in range(16)]
POINT_DESCRIPTIONS = []
for r_idx, row_label in enumerate(["row 0", "row 3", "row 6", "row 9"]):
    for c_idx, col_label in enumerate(["col 0", "col 3", "col 6", "col 9"]):
        POINT_DESCRIPTIONS.append(f"{row_label}, {col_label}")

# Colors per row of points
ROW_COLORS = [
    (0, 255, 0),    # row 0 = green
    (255, 255, 0),  # row 3 = cyan
    (0, 255, 255),  # row 6 = yellow
    (0, 0, 255),    # row 9 = red
]


class CornerPicker:
    """OpenCV mouse callback handler for picking 16 grid intersection points."""

    def __init__(self, image):
        self.original = image.copy()
        self.points = []
        self._last_click_time = 0.0

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        now = time.monotonic()
        if now - self._last_click_time < 0.3:
            return  # debounce
        self._last_click_time = now
        if len(self.points) < 16:
            self.points.append((x, y))

    def undo(self):
        if self.points:
            self.points.pop()

    def reset(self):
        self.points = []

    def draw(self):
        img = self.original.copy()

        # Draw placed points with row-based colors and connecting lines
        for i, (x, y) in enumerate(self.points):
            row = i // 4
            color = ROW_COLORS[row]
            cv2.circle(img, (x, y), 7, color, -1)
            cv2.circle(img, (x, y), 9, (255, 255, 255), 2)
            cv2.putText(img, POINT_LABELS[i], (x + 10, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw horizontal lines (within each row)
        for row in range(4):
            start = row * 4
            for col in range(3):
                idx_a = start + col
                idx_b = start + col + 1
                if idx_a < len(self.points) and idx_b < len(self.points):
                    cv2.line(img, self.points[idx_a], self.points[idx_b], ROW_COLORS[row], 2)

        # Draw vertical lines (between rows)
        for row in range(3):
            for col in range(4):
                idx_a = row * 4 + col
                idx_b = (row + 1) * 4 + col
                if idx_a < len(self.points) and idx_b < len(self.points):
                    color_a = ROW_COLORS[row]
                    color_b = ROW_COLORS[row + 1]
                    # Average color for vertical lines
                    avg_color = tuple((a + b) // 2 for a, b in zip(color_a, color_b))
                    cv2.line(img, self.points[idx_a], self.points[idx_b], avg_color, 2)

        # Instructions
        n = len(self.points)
        if n < 16:
            label = POINT_LABELS[n]
            desc = POINT_DESCRIPTIONS[n]
            text = f"Click {label} ({desc}) - {n}/16  |  'u'=undo  'r'=reset  's'=skip  'q'=quit"
        else:
            text = "All 16 points placed. Press ENTER to confirm  |  'u'=undo  'r'=reset  'q'=quit"

        # Draw text with outline for readability
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return img


def warp_grid_piecewise(image, points_16, size=450):
    """Piecewise perspective transform using 16 annotated grid intersection points.

    All 9 quads use real annotated corners — no interpolation needed.
    """
    pts = np.array(points_16, dtype=np.float32)

    s3 = size / 3
    s6 = size * 2 / 3

    # 9 source quads: each is [TL, TR, BR, BL] of a box region
    # Points are indexed row-major: row * 4 + col
    src_quads = [
        # Top row of boxes
        [pts[0], pts[1], pts[5], pts[4]],     # Box 0
        [pts[1], pts[2], pts[6], pts[5]],     # Box 1
        [pts[2], pts[3], pts[7], pts[6]],     # Box 2
        # Middle row
        [pts[4], pts[5], pts[9], pts[8]],     # Box 3
        [pts[5], pts[6], pts[10], pts[9]],    # Box 4
        [pts[6], pts[7], pts[11], pts[10]],   # Box 5
        # Bottom row
        [pts[8], pts[9], pts[13], pts[12]],   # Box 6
        [pts[9], pts[10], pts[14], pts[13]],  # Box 7
        [pts[10], pts[11], pts[15], pts[14]], # Box 8
    ]

    dst_quads = [
        [[0, 0], [s3, 0], [s3, s3], [0, s3]],
        [[s3, 0], [s6, 0], [s6, s3], [s3, s3]],
        [[s6, 0], [size, 0], [size, s3], [s6, s3]],
        [[0, s3], [s3, s3], [s3, s6], [0, s6]],
        [[s3, s3], [s6, s3], [s6, s6], [s3, s6]],
        [[s6, s3], [size, s3], [size, s6], [s6, s6]],
        [[0, s6], [s3, s6], [s3, size], [0, size]],
        [[s3, s6], [s6, s6], [s6, size], [s3, size]],
        [[s6, s6], [size, s6], [size, size], [s6, size]],
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


def draw_grid_overlay(warped):
    """Draw 9x9 grid lines on the warped image."""
    img = warped.copy()
    h, w = img.shape[:2]
    for i in range(10):
        y = int(i * h / 9)
        x = int(i * w / 9)
        thickness = 2 if i % 3 == 0 else 1
        color = (0, 255, 0) if i % 3 == 0 else (0, 180, 0)
        cv2.line(img, (0, y), (w, y), color, thickness)
        cv2.line(img, (x, 0), (x, h), color, thickness)
    return img


def pick_corners(image, title):
    """Open window for 16-point corner selection.

    Returns list of 16 (x, y) tuples, or None (quit), or "skip".
    """
    h, w = image.shape[:2]
    max_dim = 900
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        display_img = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        display_img = image.copy()

    picker = CornerPicker(display_img)
    win_name = f"Annotate: {title}"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, picker.mouse_callback)

    while True:
        cv2.imshow(win_name, picker.draw())
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            cv2.destroyWindow(win_name)
            return None

        if key == ord('u'):
            picker.undo()

        if key == ord('r'):
            picker.reset()

        if key == ord('s'):
            cv2.destroyWindow(win_name)
            return "skip"

        if key in (13, 10):  # Enter
            if len(picker.points) == 16:
                points = list(picker.points)
                cv2.destroyWindow(win_name)
                # Scale back to original image coordinates
                if scale != 1.0:
                    points = [(int(x / scale), int(y / scale)) for x, y in points]
                return points


def parse_cell(token):
    """Parse a single cell token.

    Returns int or list[int].
    '0' -> 0, '3' -> 3, '1/7' -> [1, 7], '3/8/9' -> [3, 8, 9]
    Raises ValueError on invalid input.
    """
    if '/' in token:
        parts = [int(p) for p in token.split('/')]
        if len(parts) < 2:
            raise ValueError(f"Multi-digit cell needs 2+ values: {token}")
        for v in parts:
            if not 1 <= v <= 9:
                raise ValueError(f"Multi-digit values must be 1-9: {token}")
        unique = sorted(set(parts))
        if len(unique) < 2:
            raise ValueError(f"Multi-digit cell needs 2+ unique values: {token}")
        return unique
    else:
        v = int(token)
        if not 0 <= v <= 9:
            raise ValueError(f"Value must be 0-9: {token}")
        return v


def format_cell(cell):
    """Format a cell value for display."""
    if isinstance(cell, list):
        return '/'.join(str(v) for v in cell)
    return str(cell) if cell else '.'


def enter_grid(warped):
    """Show warped grid and prompt user to type the 9x9 grid in terminal.

    Supports multi-digit cells with X/Y notation.
    """
    overlay = draw_grid_overlay(warped)
    display = cv2.resize(overlay, (600, 600), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Warped Grid (type grid in terminal)", display)
    cv2.waitKey(100)

    print("\n  Type the 9x9 grid (0 for empty cells).")
    print("  Enter one row per line, values separated by spaces.")
    print("  Use X/Y for ambiguous cells (e.g., 1/7 means either 1 or 7).")
    print("  Example: 5 0 9 1/7 0 0 0 2 0")
    print("  Type 'skip' to skip this image.\n")

    grid = []
    for row_idx in range(9):
        while True:
            try:
                line = input(f"  Row {row_idx + 1}: ").strip()
                if line.lower() == "skip":
                    cv2.destroyWindow("Warped Grid (type grid in terminal)")
                    return None

                # Split by whitespace
                tokens = line.split()

                # Support compact format: 9 chars with no spaces and no slashes
                if len(tokens) == 1 and len(line) == 9 and '/' not in line:
                    tokens = list(line)

                if len(tokens) != 9:
                    print(f"    Need 9 values, got {len(tokens)}. Try again.")
                    continue

                row = [parse_cell(t) for t in tokens]
                grid.append(row)
                break
            except ValueError as e:
                print(f"    Invalid: {e}. Try again.")
            except KeyboardInterrupt:
                print("    Use 'skip' to skip or 'q' in the window to quit.")

    cv2.destroyWindow("Warped Grid (type grid in terminal)")
    return grid


def load_annotations():
    """Load existing annotations if any."""
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            return json.load(f)
    return {"images": []}


def save_annotations(data):
    """Save annotations to disk."""
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_already_annotated(data):
    """Get set of already annotated image paths."""
    return {img["path"] for img in data["images"]}


def main():
    # Collect all images from Ground Example directory, sorted
    all_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpeg", ".jpg", ".png", ".webp"))
    ])
    image_paths = [f"{IMAGE_DIR}/{f}" for f in all_files]

    data = load_annotations()
    already_done = get_already_annotated(data)
    remaining = [p for p in image_paths if p not in already_done]
    done_count = len(image_paths) - len(remaining)

    print("=" * 65)
    print("  SUDOKU GROUND TRUTH ANNOTATION TOOL v2 (16-point)")
    print("=" * 65)
    print(f"\n  Source: {IMAGE_DIR}/")
    print(f"  Images: {len(image_paths)} total, {done_count} done, {len(remaining)} remaining")
    print(f"  Saving to: {OUTPUT_PATH}")
    print()
    print("  For each image:")
    print("    1. Click 16 grid intersection points (row by row, left to right)")
    print("       P0-P3 (top edge), P4-P7 (row 3), P8-P11 (row 6), P12-P15 (bottom)")
    print("    2. Press Enter to confirm")
    print("    3. Type the grid in the terminal (0=empty, X/Y for ambiguous)")
    print()
    print("  Keys: 'u'=undo  'r'=reset  's'=skip  'q'=quit")
    print("=" * 65)

    for idx, img_path in enumerate(remaining):
        img_name = Path(img_path).name
        print(f"\n[{done_count + idx + 1}/{len(image_paths)}] {img_name}")

        image = cv2.imread(img_path)
        if image is None:
            print(f"  ERROR: Could not load {img_path}")
            continue

        # Step 1: Pick 16 corners
        result = pick_corners(image, img_name)

        if result is None:
            print("  Quitting. Progress saved.")
            save_annotations(data)
            return

        if result == "skip":
            print("  Skipped.")
            continue

        points_16 = result
        print(f"  16 points placed.")

        # Step 2: Show piecewise-warped grid and enter values
        warped = warp_grid_piecewise(image, points_16)
        grid = enter_grid(warped)

        if grid is None:
            print("  Grid entry skipped.")
            continue

        entry = {
            "path": img_path,
            "corners_16": [list(pt) for pt in points_16],
            "grid": grid,
        }

        print("\n  Grid entered:")
        for row in grid:
            print("    " + " ".join(format_cell(v) for v in row))

        data["images"].append(entry)
        save_annotations(data)
        print(f"  Saved. ({done_count + idx + 1}/{len(image_paths)} done)")

    print(f"\nAll done! {len(data['images'])} annotations saved to {OUTPUT_PATH}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
