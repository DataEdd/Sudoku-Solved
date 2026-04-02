"""
Interactive ground truth annotation tool.

Opens each image in an OpenCV window. Click 8 corners in two phases:
  Phase 1: 4 outer corners (TL, TR, BR, BL) of the full grid
  Phase 2: 4 center-box corners (CTL, CTR, CBR, CBL) at intersections
           of rows 3,6 and columns 3,6

Then type the 9x9 grid in the terminal. Saves to ground_truth_annotated.json.

Usage:
    cd Sudoku-Solved/
    python evaluation/annotate.py

Controls:
    Left click  — place next corner point
    'u'         — undo last corner
    'r'         — reset all corners for current phase
    'q'         — quit (progress is saved)
    Enter       — confirm current phase / confirm all and proceed
"""

import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

OUTPUT_PATH = "evaluation/ground_truth_annotated.json"

OUTER_LABELS = ["TL", "TR", "BR", "BL"]
CENTER_LABELS = ["CTL", "CTR", "CBR", "CBL"]

OUTER_COLORS = [
    (0, 255, 0),    # TL = green
    (0, 255, 255),  # TR = yellow
    (0, 0, 255),    # BR = red
    (255, 0, 255),  # BL = magenta
]
CENTER_COLORS = [
    (255, 255, 0),  # CTL = cyan
    (255, 200, 0),  # CTR = light cyan
    (255, 150, 0),  # CBR = blue-ish
    (255, 100, 0),  # CBL = darker blue
]


class CornerPicker:
    """OpenCV mouse callback handler for picking 8 corners in two phases."""

    def __init__(self, image):
        self.original = image.copy()
        self.outer_corners = []
        self.center_corners = []
        self.phase = "outer"  # "outer" or "center"
        self.done = False

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.phase == "outer" and len(self.outer_corners) < 4:
            self.outer_corners.append((x, y))
        elif self.phase == "center" and len(self.center_corners) < 4:
            self.center_corners.append((x, y))

    def undo(self):
        if self.phase == "center" and self.center_corners:
            self.center_corners.pop()
        elif self.phase == "center" and not self.center_corners:
            # Go back to outer phase
            self.phase = "outer"
            if self.outer_corners:
                self.outer_corners.pop()
        elif self.phase == "outer" and self.outer_corners:
            self.outer_corners.pop()

    def reset(self):
        if self.phase == "center":
            self.center_corners = []
        else:
            self.outer_corners = []

    def draw(self):
        img = self.original.copy()

        # Draw outer corners and quad
        for i, (x, y) in enumerate(self.outer_corners):
            color = OUTER_COLORS[i]
            cv2.circle(img, (x, y), 8, color, -1)
            cv2.circle(img, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(img, OUTER_LABELS[i], (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if len(self.outer_corners) >= 2:
            for i in range(len(self.outer_corners) - 1):
                cv2.line(img, self.outer_corners[i], self.outer_corners[i + 1], (0, 255, 0), 2)
            if len(self.outer_corners) == 4:
                cv2.line(img, self.outer_corners[3], self.outer_corners[0], (0, 255, 0), 2)

        # Draw center corners and quad
        for i, (x, y) in enumerate(self.center_corners):
            color = CENTER_COLORS[i]
            cv2.circle(img, (x, y), 8, color, -1)
            cv2.circle(img, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(img, CENTER_LABELS[i], (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if len(self.center_corners) >= 2:
            for i in range(len(self.center_corners) - 1):
                cv2.line(img, self.center_corners[i], self.center_corners[i + 1], (255, 255, 0), 2)
            if len(self.center_corners) == 4:
                cv2.line(img, self.center_corners[3], self.center_corners[0], (255, 255, 0), 2)

        # Instructions
        if self.phase == "outer":
            if len(self.outer_corners) < 4:
                label = OUTER_LABELS[len(self.outer_corners)]
                text = f"OUTER: Click {label} ({len(self.outer_corners)}/4)  |  'u'=undo  'r'=reset  'q'=quit"
            else:
                text = "OUTER done. Press ENTER for center-box phase  |  'r'=reset  'q'=quit"
        else:
            if len(self.center_corners) < 4:
                label = CENTER_LABELS[len(self.center_corners)]
                text = f"CENTER BOX: Click {label} ({len(self.center_corners)}/4)  |  'u'=undo  'r'=reset  'q'=quit"
            else:
                text = "All 8 corners set. Press ENTER to confirm  |  'r'=reset  'q'=quit"

        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        return img


def warp_grid_piecewise(image, outer_corners, center_corners, size=450):
    """Piecewise perspective transform using 8 annotated corners.

    Divides the grid into 9 box-regions using outer + center corners
    plus interpolated boundary midpoints. Each region gets its own
    local homography, ensuring the center box is a perfect square
    and all grid lines are straight.

    Returns the composited square image of size x size.
    """
    outer = np.array(outer_corners, dtype=np.float32)
    center = np.array(center_corners, dtype=np.float32)

    # Outer corners: TL=0, TR=1, BR=2, BL=3
    TL, TR, BR, BL = outer[0], outer[1], outer[2], outer[3]
    # Center-box corners: CTL=0, CTR=1, CBR=2, CBL=3
    CTL, CTR, CBR, CBL = center[0], center[1], center[2], center[3]

    # Interpolate 8 boundary midpoints where 3rd/6th lines meet outer edges
    # Top edge: row 0
    T3 = TL + (TR - TL) / 3      # (row0, col3)
    T6 = TL + (TR - TL) * 2 / 3  # (row0, col6)
    # Bottom edge: row 9
    B3 = BL + (BR - BL) / 3      # (row9, col3)
    B6 = BL + (BR - BL) * 2 / 3  # (row9, col6)
    # Left edge: col 0
    L3 = TL + (BL - TL) / 3      # (row3, col0)
    L6 = TL + (BL - TL) * 2 / 3  # (row6, col0)
    # Right edge: col 9
    R3 = TR + (BR - TR) / 3      # (row3, col9)
    R6 = TR + (BR - TR) * 2 / 3  # (row6, col9)

    # Target positions in the output image (regular grid)
    s3 = size / 3
    s6 = size * 2 / 3

    # 16-point grid: source (image) and destination (output)
    # Layout (row, col) with positions:
    #   (0,0)=TL  (0,3)=T3  (0,6)=T6  (0,9)=TR
    #   (3,0)=L3  (3,3)=CTL (3,6)=CTR (3,9)=R3
    #   (6,0)=L6  (6,3)=CBL (6,6)=CBR (6,9)=R6
    #   (9,0)=BL  (9,3)=B3  (9,6)=B6  (9,9)=BR

    # Define 9 quads: each is [TL, TR, BR, BL] of the sub-region
    # Source quads (in original image coordinates)
    src_quads = [
        # Row 0: boxes 1-3
        [TL,  T3,  CTL, L3],   # box1 (top-left)
        [T3,  T6,  CTR, CTL],  # box2 (top-center)
        [T6,  TR,  R3,  CTR],  # box3 (top-right)
        # Row 1: boxes 4-6
        [L3,  CTL, CBL, L6],   # box4 (mid-left)
        [CTL, CTR, CBR, CBL],  # box5 (center)
        [CTR, R3,  R6,  CBR],  # box6 (mid-right)
        # Row 2: boxes 7-9
        [L6,  CBL, B3,  BL],   # box7 (bot-left)
        [CBL, CBR, B6,  B3],   # box8 (bot-center)
        [CBR, R6,  BR,  B6],   # box9 (bot-right)
    ]

    # Destination quads (regular grid positions in output)
    dst_quads = [
        # Row 0
        [np.array([0, 0]),  np.array([s3, 0]),  np.array([s3, s3]),  np.array([0, s3])],
        [np.array([s3, 0]), np.array([s6, 0]),  np.array([s6, s3]),  np.array([s3, s3])],
        [np.array([s6, 0]), np.array([size, 0]), np.array([size, s3]), np.array([s6, s3])],
        # Row 1
        [np.array([0, s3]),  np.array([s3, s3]),  np.array([s3, s6]),  np.array([0, s6])],
        [np.array([s3, s3]), np.array([s6, s3]),  np.array([s6, s6]),  np.array([s3, s6])],
        [np.array([s6, s3]), np.array([size, s3]), np.array([size, s6]), np.array([s6, s6])],
        # Row 2
        [np.array([0, s6]),  np.array([s3, s6]),  np.array([s3, size]),  np.array([0, size])],
        [np.array([s3, s6]), np.array([s6, s6]),  np.array([s6, size]),  np.array([s3, size])],
        [np.array([s6, s6]), np.array([size, s6]), np.array([size, size]), np.array([s6, size])],
    ]

    output = np.zeros((size, size, 3), dtype=np.uint8)

    for src_q, dst_q in zip(src_quads, dst_quads):
        src_pts = np.array(src_q, dtype=np.float32)
        dst_pts = np.array(dst_q, dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (size, size))

        # Create mask for this quad region in output space
        mask = np.zeros((size, size), dtype=np.uint8)
        roi = dst_pts.astype(np.int32)
        cv2.fillConvexPoly(mask, roi, 255)

        # Composite onto output
        mask_3ch = cv2.merge([mask, mask, mask])
        output = np.where(mask_3ch > 0, warped, output)

    return output


def warp_grid(image, corners, size=450):
    """Simple 4-corner perspective transform (fallback)."""
    pts = np.array(corners, dtype=np.float32)
    dst = np.array([
        [0, 0], [size - 1, 0],
        [size - 1, size - 1], [0, size - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (size, size))


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
    """Open window for 8-point corner selection (outer + center box).

    Returns (outer_corners, center_corners) or None (quit) or "skip".
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
            if picker.phase == "outer" and len(picker.outer_corners) == 4:
                # Transition to center phase
                picker.phase = "center"
            elif picker.phase == "center" and len(picker.center_corners) == 4:
                # All 8 corners confirmed
                outer = picker.outer_corners
                center = picker.center_corners
                cv2.destroyWindow(win_name)
                # Scale back to original coordinates
                if scale != 1.0:
                    outer = [(int(x / scale), int(y / scale)) for x, y in outer]
                    center = [(int(x / scale), int(y / scale)) for x, y in center]
                return outer, center


def enter_grid(warped):
    """Show warped grid and prompt user to type the 9x9 grid in terminal."""
    overlay = draw_grid_overlay(warped)
    display = cv2.resize(overlay, (600, 600), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Warped Grid (type grid in terminal)", display)
    cv2.waitKey(100)

    print("\n  Type the 9x9 grid (0 for empty cells).")
    print("  Enter one row per line, digits separated by spaces.")
    print("  Example: 0 3 0 0 7 0 5 0 6")
    print("  Type 'skip' to skip this image.\n")

    grid = []
    for row_idx in range(9):
        while True:
            try:
                line = input(f"  Row {row_idx + 1}: ").strip()
                if line.lower() == "skip":
                    cv2.destroyWindow("Warped Grid (type grid in terminal)")
                    return None

                if len(line.replace(" ", "").replace(",", "")) == 9 and " " not in line and "," not in line:
                    values = [int(c) for c in line.replace(" ", "").replace(",", "")]
                else:
                    values = [int(x) for x in line.replace(",", " ").split()]

                if len(values) != 9:
                    print(f"    Need 9 values, got {len(values)}. Try again.")
                    continue
                if not all(0 <= v <= 9 for v in values):
                    print("    Values must be 0-9. Try again.")
                    continue

                grid.append(values)
                break
            except (ValueError, KeyboardInterrupt):
                print("    Invalid input. Use digits 0-9 separated by spaces.")

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


def migrate_old_format(data):
    """Migrate old annotations that have 'corners' to 'outer_corners'."""
    changed = False
    for img in data["images"]:
        if "corners" in img and "outer_corners" not in img:
            img["outer_corners"] = img.pop("corners")
            img.setdefault("center_corners", None)
            changed = True
    return changed


def main():
    random.seed(42)
    all_files = [f for f in os.listdir("Examples/aug/") if f.endswith((".jpeg", ".jpg", ".png", ".webp"))]
    sample_files = random.sample(all_files, 20)
    image_paths = [f"Examples/aug/{f}" for f in sorted(sample_files)]

    data = load_annotations()

    # Migrate old format if needed
    if migrate_old_format(data):
        save_annotations(data)
        print("  (Migrated old 'corners' → 'outer_corners' format)")

    already_done = get_already_annotated(data)

    # Images needing center corners (have outer but no center)
    needs_center = {
        img["path"] for img in data["images"]
        if img.get("center_corners") is None and img.get("outer_corners") is not None
    }

    remaining = [p for p in image_paths if p not in already_done or p in needs_center]
    done_count = len(image_paths) - len(remaining)

    print("=" * 60)
    print("  SUDOKU GROUND TRUTH ANNOTATION TOOL (8-point)")
    print("=" * 60)
    print(f"\n  Images: {len(image_paths)} total, {done_count} fully done, {len(remaining)} remaining")
    if needs_center:
        print(f"  ({len(needs_center)} need center-box corners added)")
    print(f"  Saving to: {OUTPUT_PATH}")
    print()
    print("  For each image:")
    print("    1. Click 4 OUTER corners: TL → TR → BR → BL (green)")
    print("    2. Press Enter, then click 4 CENTER BOX corners: CTL → CTR → CBR → CBL (cyan)")
    print("       (intersections of 3rd/6th grid lines)")
    print("    3. Type the grid in the terminal (0 = empty)")
    print()
    print("  Keys: 'u'=undo  'r'=reset  's'=skip  'q'=quit")
    print("=" * 60)

    for idx, img_path in enumerate(remaining):
        img_name = Path(img_path).name
        print(f"\n[{done_count + idx + 1}/{len(image_paths)}] {img_name}")

        image = cv2.imread(img_path)
        if image is None:
            print(f"  ERROR: Could not load {img_path}")
            continue

        # Check if this image already has outer corners (just needs center)
        existing = None
        if img_path in needs_center:
            existing = next(img for img in data["images"] if img["path"] == img_path)
            print(f"  (Already has outer corners, needs center-box corners)")

        # Step 1: Pick corners
        result = pick_corners(image, img_name)

        if result is None:
            print("  Quitting. Progress saved.")
            save_annotations(data)
            return

        if result == "skip":
            print("  Skipped.")
            continue

        outer_corners, center_corners = result
        print(f"  Outer corners:  {outer_corners}")
        print(f"  Center corners: {center_corners}")

        # Step 2: Show piecewise-warped grid and enter values
        warped = warp_grid_piecewise(image, outer_corners, center_corners)
        grid = enter_grid(warped)

        if existing:
            # Update existing entry
            existing["outer_corners"] = outer_corners
            existing["center_corners"] = center_corners
            if grid is not None:
                existing["grid"] = grid
        else:
            entry = {
                "path": img_path,
                "outer_corners": outer_corners,
                "center_corners": center_corners,
                "grid": grid,
            }
            if grid is not None:
                print("\n  Grid entered:")
                for row in grid:
                    print("    " + " ".join(str(v) if v else "." for v in row))

            data["images"].append(entry)

        save_annotations(data)
        print(f"  Saved. ({done_count + idx + 1}/{len(image_paths)} done)")

    print(f"\nAll done! {len(data['images'])} annotations saved to {OUTPUT_PATH}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
