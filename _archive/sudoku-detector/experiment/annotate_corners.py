#!/usr/bin/env python3
"""
Ground Truth Corner Annotation Tool (OpenCV-based)

Usage:
    python annotate_corners.py

Instructions:
    - Click 4 corners in order: TL, TR, BR, BL
    - Press 'r' to reset
    - Press 'Enter' or 'Space' to confirm
    - Press 'n' to skip to next image
    - Press 'q' or 'Esc' to quit (saves progress)
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configuration
RANDOM_SEED = 42
N_SAMPLES = 6
GROUND_TRUTH_FILE = Path('../data/ground_truth_corners.json')
EXAMPLES_DIR = Path('../examples')

# Colors (BGR)
COLORS = [
    (0, 0, 255),    # Red - TL
    (0, 255, 0),    # Green - TR
    (255, 0, 0),    # Blue - BR
    (0, 255, 255),  # Yellow - BL
]
CORNER_LABELS = ['TL', 'TR', 'BR', 'BL']


def load_ground_truth(filepath: Path = GROUND_TRUTH_FILE) -> Dict:
    """Load existing ground truth from JSON file."""
    if not filepath.exists():
        return {}
    with open(filepath, 'r') as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith('_')}


def save_ground_truth(data: Dict, filepath: Path = GROUND_TRUTH_FILE):
    """Save ground truth to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} annotations to {filepath}")


def get_sample_images() -> List[Path]:
    """Get the same sample images used in the notebook."""
    random.seed(RANDOM_SEED)
    all_images = list(EXAMPLES_DIR.glob('*.jpeg'))
    return random.sample(all_images, N_SAMPLES)


class CornerAnnotator:
    """OpenCV-based corner annotation tool."""

    def __init__(self, img_path: Path):
        self.img_path = img_path
        self.original = cv2.imread(str(img_path))
        self.corners: List[Tuple[int, int]] = []
        self.confirmed = False
        self.skip = False
        self.quit = False

    def _draw(self) -> np.ndarray:
        """Draw current state on image."""
        img = self.original.copy()
        h, w = img.shape[:2]

        # Draw existing corners
        for i, (x, y) in enumerate(self.corners):
            cv2.circle(img, (x, y), 8, COLORS[i], -1)
            cv2.circle(img, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(img, CORNER_LABELS[i], (x + 15, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[i], 2)

        # Draw polygon if we have corners
        if len(self.corners) >= 2:
            pts = np.array(self.corners, dtype=np.int32)
            cv2.polylines(img, [pts], False, (255, 255, 0), 2)
        if len(self.corners) == 4:
            pts = np.array(self.corners, dtype=np.int32)
            cv2.polylines(img, [pts], True, (255, 255, 0), 2)

        # Draw instructions
        if len(self.corners) < 4:
            next_label = CORNER_LABELS[len(self.corners)]
            text = f"Click {next_label} corner ({len(self.corners)}/4)"
        else:
            text = "Press ENTER to confirm, 'r' to reset"

        # Draw text background
        cv2.rectangle(img, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw filename
        cv2.rectangle(img, (0, h - 35), (w, h), (0, 0, 0), -1)
        cv2.putText(img, f"{self.img_path.name} | r=reset, n=skip, q=quit",
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return img

    def _on_mouse(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) < 4:
            self.corners.append((x, y))
            print(f"  {CORNER_LABELS[len(self.corners)-1]}: ({x}, {y})")

    def annotate(self) -> Optional[np.ndarray]:
        """Run annotation. Returns corners array or None."""
        window_name = "Annotate Corners"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 800)
        cv2.setMouseCallback(window_name, self._on_mouse)

        while True:
            img = self._draw()
            cv2.imshow(window_name, img)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('r'):  # Reset
                print("  Reset")
                self.corners = []
            elif key == ord('n'):  # Skip
                self.skip = True
                break
            elif key == ord('q') or key == 27:  # Quit (q or Esc)
                self.quit = True
                break
            elif (key == 13 or key == 32) and len(self.corners) == 4:  # Enter or Space
                self.confirmed = True
                break

        cv2.destroyWindow(window_name)

        if self.confirmed:
            return np.array(self.corners, dtype=np.float32)
        return None


def main():
    print("=" * 60)
    print("GROUND TRUTH CORNER ANNOTATION")
    print("=" * 60)
    print()
    print("Click corners in order: TL → TR → BR → BL")
    print("Keys: r=reset, Enter=confirm, n=skip, q=quit")
    print()

    # Load existing
    ground_truth = load_ground_truth()
    print(f"Existing annotations: {len(ground_truth)}")

    # Get sample images
    sample_images = get_sample_images()
    need_annotation = [p for p in sample_images if p.name not in ground_truth]

    print(f"Images to annotate: {len(need_annotation)}")
    print()

    if not need_annotation:
        print("All images already annotated!")
        return

    # Annotate
    for i, img_path in enumerate(need_annotation):
        print(f"[{i+1}/{len(need_annotation)}] {img_path.name}")

        annotator = CornerAnnotator(img_path)
        corners = annotator.annotate()

        if annotator.quit:
            print("Quitting...")
            break

        if corners is not None:
            ground_truth[img_path.name] = corners.tolist()
            save_ground_truth(ground_truth)
            print(f"  ✓ Saved")
        else:
            print(f"  Skipped")

    # Summary
    print()
    print("=" * 60)
    print(f"Done! {len(ground_truth)} total annotations")
    for img in sample_images:
        status = "✓" if img.name in ground_truth else "✗"
        print(f"  {status} {img.name}")


if __name__ == '__main__':
    main()
