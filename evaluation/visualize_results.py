"""
Generate comparison visuals from benchmark results.

Creates:
1. Bar chart: detection rate per method per category
2. Timing comparison bar chart
3. Side-by-side detection images (one row per image, one col per method)
4. Summary table as an image

Usage:
    cd Sudoku-Solved/
    python -m evaluation.visualize_results
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, skipping chart generation")


VISUALS_DIR = os.path.join("evaluation", "visuals")
RESULTS_PATH = os.path.join("evaluation", "results.json")

# Color palette for methods
METHOD_COLORS = {
    "contour": "#2ecc71",
    "simple_baseline": "#3498db",
    "sobel_flood": "#9b59b6",
    "line_segment": "#e74c3c",
    "hough_standard": "#f39c12",
    "hough_polar": "#1abc9c",
    "sudoku_detector": "#e67e22",
    "generalized_hough": "#95a5a6",
}

METHOD_SHORT = {
    "contour": "Contour",
    "simple_baseline": "Baseline",
    "sobel_flood": "Sobel+Flood",
    "line_segment": "Line Seg",
    "hough_standard": "Hough Std",
    "hough_polar": "Hough Polar",
    "sudoku_detector": "Sudoku-Det",
    "generalized_hough": "GHT",
}


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def plot_detection_rate_by_category(data):
    """Bar chart: detection rate per method, grouped by category."""
    if not HAS_MPL:
        return

    summary = data["summary"]
    methods = list(summary["methods"].keys())
    categories = sorted(summary["by_category"].keys())

    fig, ax = plt.subplots(figsize=(14, 6))

    n_methods = len(methods)
    n_cats = len(categories)
    bar_width = 0.8 / n_methods
    x = np.arange(n_cats)

    for i, method in enumerate(methods):
        rates = []
        for cat in categories:
            if method in summary["by_category"][cat]:
                rates.append(summary["by_category"][cat][method]["detection_rate"] * 100)
            else:
                rates.append(0)
        offset = (i - n_methods / 2 + 0.5) * bar_width
        color = METHOD_COLORS.get(method, "#666")
        ax.bar(x + offset, rates, bar_width * 0.9,
               label=METHOD_SHORT.get(method, method), color=color, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title("Sudoku Grid Detection Rate by Category", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "detection_rate_by_category.png"), dpi=150)
    plt.close()
    print("Saved: detection_rate_by_category.png")


def plot_overall_comparison(data):
    """Horizontal bar chart: overall detection rate + timing."""
    if not HAS_MPL:
        return

    summary = data["summary"]
    methods = sorted(summary["methods"].keys(),
                     key=lambda m: summary["methods"][m]["detection_rate"],
                     reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Detection rate
    rates = [summary["methods"][m]["detection_rate"] * 100 for m in methods]
    colors = [METHOD_COLORS.get(m, "#666") for m in methods]
    labels = [METHOD_SHORT.get(m, m) for m in methods]

    bars1 = ax1.barh(labels, rates, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Detection Rate (%)")
    ax1.set_title("Detection Rate (higher = better)", fontweight="bold")
    ax1.set_xlim(0, 110)
    for bar, rate in zip(bars1, rates):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{rate:.0f}%", va="center", fontsize=10)

    # Timing (log scale)
    timings = [summary["methods"][m]["median_timing_ms"] for m in methods]
    bars2 = ax2.barh(labels, timings, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Median Time (ms, log scale)")
    ax2.set_title("Speed (lower = better)", fontweight="bold")
    ax2.set_xscale("log")
    for bar, t in zip(bars2, timings):
        ax2.text(bar.get_width() * 1.2, bar.get_y() + bar.get_height()/2,
                 f"{t:.0f}ms", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "overall_comparison.png"), dpi=150)
    plt.close()
    print("Saved: overall_comparison.png")


def create_detection_grid(data, max_images=8):
    """
    Side-by-side grid showing detected borders for each method on same images.
    Uses debug images if available, otherwise draws corners on original.
    """
    results = data["results"][:max_images]
    methods = list(data["summary"]["methods"].keys())

    # Cell dimensions
    cell_w, cell_h = 200, 200
    label_h = 30
    row_label_w = 140

    n_methods = len(methods)
    n_images = len(results)

    grid_w = row_label_w + n_methods * cell_w
    grid_h = label_h + n_images * cell_h

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40  # Dark gray bg

    # Column headers
    for j, method in enumerate(methods):
        x = row_label_w + j * cell_w + cell_w // 2
        label = METHOD_SHORT.get(method, method)
        cv2.putText(grid, label, (x - len(label) * 4, label_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Fill cells
    for i, img_result in enumerate(results):
        y_start = label_h + i * cell_h
        img_path = img_result["image_path"]
        img_name = Path(img_path).stem[:18]

        # Row label
        cv2.putText(grid, img_name, (5, y_start + cell_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # Load original image
        original = cv2.imread(img_path)
        if original is None:
            continue

        for j, method in enumerate(methods):
            x_start = row_label_w + j * cell_w

            mr = img_result["methods"].get(method)
            if mr is None:
                continue

            # Try to load debug image
            cell_img = None
            if mr.get("debug_image_path") and os.path.exists(mr["debug_image_path"]):
                cell_img = cv2.imread(mr["debug_image_path"])

            if cell_img is None:
                cell_img = original.copy()
                if mr["detected"] and mr.get("corners"):
                    pts = np.array(mr["corners"]).astype(np.int32)
                    cv2.polylines(cell_img, [pts], True, (0, 255, 0), 3)

            # Add status indicator
            if mr["detected"]:
                cv2.circle(cell_img, (20, 20), 10, (0, 255, 0), -1)
            else:
                cv2.circle(cell_img, (20, 20), 10, (0, 0, 255), -1)

            # Resize to cell
            cell_resized = cv2.resize(cell_img, (cell_w, cell_h))
            grid[y_start:y_start + cell_h, x_start:x_start + cell_w] = cell_resized

    out_path = os.path.join(VISUALS_DIR, "detection_grid.png")
    cv2.imwrite(out_path, grid)
    print(f"Saved: detection_grid.png ({n_images} images x {n_methods} methods)")


def create_hero_image(data):
    """Create a hero image showing input -> detection -> (future: solved)."""
    results = data["results"]

    # Find an image where contour detected successfully
    for r in results:
        if r["methods"].get("contour", {}).get("detected"):
            img_path = r["image_path"]
            break
    else:
        print("No successful contour detection for hero image")
        return

    original = cv2.imread(img_path)
    if original is None:
        return

    # Draw contour detection
    detected = original.copy()
    corners = r["methods"]["contour"].get("corners")
    if corners:
        pts = np.array(corners).astype(np.int32)
        cv2.polylines(detected, [pts], True, (0, 255, 0), 4)
        for pt in pts:
            cv2.circle(detected, tuple(pt), 10, (0, 0, 255), -1)

    # Resize both to same height
    target_h = 400
    def resize_h(img, h):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    orig_resized = resize_h(original, target_h)
    det_resized = resize_h(detected, target_h)

    # Arrow between them
    arrow_w = 60
    arrow = np.ones((target_h, arrow_w, 3), dtype=np.uint8) * 40
    mid_y = target_h // 2
    cv2.arrowedLine(arrow, (10, mid_y), (arrow_w - 10, mid_y),
                    (255, 255, 255), 3, tipLength=0.4)

    hero = np.hstack([orig_resized, arrow, det_resized])

    # Add title bar
    title_h = 50
    title_bar = np.ones((title_h, hero.shape[1], 3), dtype=np.uint8) * 30
    cv2.putText(title_bar, "Input Photo", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(title_bar, "Detected Grid", (orig_resized.shape[1] + arrow_w + 20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    hero = np.vstack([title_bar, hero])

    cv2.imwrite(os.path.join(VISUALS_DIR, "hero.png"), hero)
    print("Saved: hero.png")


def main():
    os.makedirs(VISUALS_DIR, exist_ok=True)
    data = load_results()

    print(f"Loaded results: {len(data['results'])} images, "
          f"{len(data['summary']['methods'])} methods\n")

    # Generate all visuals
    plot_detection_rate_by_category(data)
    plot_overall_comparison(data)
    create_detection_grid(data)
    create_hero_image(data)

    print(f"\nAll visuals saved to {VISUALS_DIR}/")


if __name__ == "__main__":
    main()
