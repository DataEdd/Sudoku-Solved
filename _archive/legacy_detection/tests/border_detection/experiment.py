"""
Interactive experiment script for testing preprocessing and border detection.

Define a list of instructions to execute in order on a single image.

Usage:
    python tests/border_detection/experiment.py

Instructions are defined as a list of tuples: (step_name, params_dict)
Available steps are from the pipeline module plus special detection steps.
"""

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.pipeline import PreprocessingPipeline, PipelineConfig
from app.core.border_detection import DetectorRegistry
from tests.border_detection.sampler import sample_images, load_image


class Experiment:
    """
    Run a sequence of instructions on an image.

    Example:
        exp = Experiment()
        exp.load_random_image(seed=42)

        exp.run_instructions([
            ("grayscale", {}),
            ("gaussian_blur", {"kernel_size": 5}),
            ("canny", {"low_threshold": 50, "high_threshold": 150}),
            ("detect_border", {"method": "line_segment"}),
        ])

        exp.show_results()
        exp.save_report("my_experiment.html")
    """

    def __init__(self):
        self.image: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        self.results: List[Dict[str, Any]] = []
        self.current: Optional[np.ndarray] = None
        self.detected_corners: Optional[np.ndarray] = None

    def load_random_image(self, seed: Optional[int] = None, aug_level: Optional[int] = None) -> str:
        """Load a random image from Examples/aug."""
        aug_levels = [aug_level] if aug_level is not None else None
        images = sample_images(1, aug_levels=aug_levels, seed=seed)

        if not images:
            raise FileNotFoundError("No images found in Examples/aug/")

        self.image_path = str(images[0])
        self.image = load_image(images[0])
        self.current = self.image.copy()

        self.results = [{
            "step": "original",
            "params": {"path": self.image_path},
            "output": self.image.copy(),
        }]

        print(f"Loaded: {images[0].name}")
        return self.image_path

    def load_image(self, path: str) -> None:
        """Load a specific image."""
        self.image_path = path
        self.image = cv2.imread(path)
        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        self.current = self.image.copy()
        self.results = [{
            "step": "original",
            "params": {"path": path},
            "output": self.image.copy(),
        }]

        print(f"Loaded: {path}")

    def run_instructions(self, instructions: List[Tuple[str, Dict[str, Any]]]) -> None:
        """
        Run a list of instructions in order.

        Args:
            instructions: List of (step_name, params) tuples

        Available steps:
            Preprocessing (operate on current image):
                - grayscale: {}
                - gaussian_blur: {kernel_size: 5, sigma: 0}
                - median_blur: {kernel_size: 5}
                - bilateral: {d: 9, sigma_color: 75, sigma_space: 75}
                - adaptive_threshold: {block_size: 11, c: 2}
                - binary_threshold: {threshold: 127, use_otsu: False, invert: False}
                - canny: {low_threshold: 50, high_threshold: 150}
                - sobel: {ksize: 3, normalize: True}
                - morphology: {operation: "dilate"|"erode"|"open"|"close", kernel_size: 3}
                - contrast: {method: "clahe"|"histogram", clip_limit: 2.0}
                - invert: {}

            Detection (uses current processed image):
                - detect_border: {method: "simple_baseline"|"sobel_flood"|"line_segment"}
                - find_contours: {mode: "external", approx: "simple"}
                - hough_lines: {threshold: 80, min_length: 50, max_gap: 10}

            Utility:
                - reset: {} - Reset to original image
                - resize: {scale: 0.5} or {width: 400, height: 400}
        """
        for step_name, params in instructions:
            print(f"  {step_name}: {params}")
            self._execute_step(step_name, params)

    def _execute_step(self, step_name: str, params: Dict[str, Any]) -> None:
        """Execute a single step."""

        # Special steps
        if step_name == "reset":
            self.current = self.image.copy()
            self.results.append({
                "step": "reset",
                "params": {},
                "output": self.current.copy(),
            })
            return

        if step_name == "resize":
            self._resize(params)
            return

        if step_name == "detect_border":
            self._detect_border(params)
            return

        if step_name == "find_contours":
            self._find_contours(params)
            return

        if step_name == "hough_lines":
            self._hough_lines(params)
            return

        # Pipeline steps
        try:
            config = PipelineConfig(name="single", steps=[(step_name, params)])
            pipeline = PreprocessingPipeline(config)
            results = pipeline.run(self.current)

            if results:
                self.current = results[0].output
                self.results.append({
                    "step": step_name,
                    "params": params,
                    "output": self.current.copy(),
                    "time_ms": results[0].execution_time_ms,
                })
        except ValueError as e:
            print(f"    Error: {e}")

    def _resize(self, params: Dict[str, Any]) -> None:
        """Resize the current image."""
        if "scale" in params:
            scale = params["scale"]
            self.current = cv2.resize(self.current, None, fx=scale, fy=scale)
        elif "width" in params and "height" in params:
            self.current = cv2.resize(self.current, (params["width"], params["height"]))

        self.results.append({
            "step": "resize",
            "params": params,
            "output": self.current.copy(),
        })

    def _detect_border(self, params: Dict[str, Any]) -> None:
        """Run border detection."""
        method = params.get("method", "simple_baseline")
        detector = DetectorRegistry.get(method)

        # Get method-specific params
        det_params = {k: v for k, v in params.items() if k != "method"}

        result = detector.detect(self.image, preprocessed=self.current, **det_params)

        self.detected_corners = result.corners

        # Draw result on original image
        detection_vis = result.draw_on_image(self.image)

        self.results.append({
            "step": f"detect_border ({method})",
            "params": params,
            "output": detection_vis,
            "success": result.success,
            "confidence": result.confidence,
            "corners": result.corners,
            "time_ms": result.execution_time_ms,
        })

        status = "SUCCESS" if result.success else "FAILED"
        print(f"    -> {status} (confidence: {result.confidence:.2f})")

    def _find_contours(self, params: Dict[str, Any]) -> None:
        """Find and draw contours."""
        mode = params.get("mode", "external")
        approx = params.get("approx", "simple")

        mode_map = {
            "external": cv2.RETR_EXTERNAL,
            "list": cv2.RETR_LIST,
            "tree": cv2.RETR_TREE,
        }
        approx_map = {
            "simple": cv2.CHAIN_APPROX_SIMPLE,
            "none": cv2.CHAIN_APPROX_NONE,
        }

        # Ensure binary image
        if len(self.current.shape) == 3:
            gray = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current

        contours, _ = cv2.findContours(
            gray, mode_map.get(mode, cv2.RETR_EXTERNAL),
            approx_map.get(approx, cv2.CHAIN_APPROX_SIMPLE)
        )

        # Draw on color image
        vis = self.image.copy()
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

        # Highlight largest
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(vis, [largest], -1, (0, 0, 255), 3)

        self.results.append({
            "step": "find_contours",
            "params": params,
            "output": vis,
            "num_contours": len(contours),
        })

        print(f"    -> Found {len(contours)} contours")

    def _hough_lines(self, params: Dict[str, Any]) -> None:
        """Detect and draw Hough lines."""
        threshold = params.get("threshold", 80)
        min_length = params.get("min_length", 50)
        max_gap = params.get("max_gap", 10)

        # Ensure we have edges
        if len(self.current.shape) == 3:
            gray = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current

        lines = cv2.HoughLinesP(
            gray, 1, np.pi/180, threshold,
            minLineLength=min_length, maxLineGap=max_gap
        )

        # Draw on color image
        vis = self.image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        num_lines = len(lines) if lines is not None else 0

        self.results.append({
            "step": "hough_lines",
            "params": params,
            "output": vis,
            "num_lines": num_lines,
        })

        print(f"    -> Found {num_lines} lines")

    def show_results(self, max_cols: int = 4) -> None:
        """Display results using matplotlib."""
        import matplotlib.pyplot as plt

        n = len(self.results)
        cols = min(n, max_cols)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for i, result in enumerate(self.results):
            ax = axes[i]
            img = result["output"]

            if len(img.shape) == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            title = result["step"]
            if "time_ms" in result:
                title += f"\n({result['time_ms']:.1f}ms)"
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        # Hide unused axes
        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    def save_report(self, path: str = "experiment_report.html") -> None:
        """Save results as HTML report."""
        import base64

        def encode_img(img):
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            _, buf = cv2.imencode(".jpg", img)
            return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"

        cards = []
        for i, result in enumerate(self.results):
            img_b64 = encode_img(result["output"])

            info_items = []
            for k, v in result.items():
                if k not in ("output", "step"):
                    if isinstance(v, float):
                        info_items.append(f"<li><b>{k}:</b> {v:.3f}</li>")
                    elif isinstance(v, np.ndarray):
                        info_items.append(f"<li><b>{k}:</b> array{v.shape}</li>")
                    else:
                        info_items.append(f"<li><b>{k}:</b> {v}</li>")

            info_html = "\n".join(info_items)

            cards.append(f'''
            <div class="card">
                <div class="step-num">{i}</div>
                <img src="{img_b64}" onclick="showLightbox(this)">
                <div class="step-name">{result["step"]}</div>
                <ul class="info">{info_html}</ul>
            </div>
            ''')

        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Experiment: {Path(self.image_path).name if self.image_path else "unknown"}</title>
    <style>
        body {{ font-family: system-ui; background: #f0f0f0; margin: 20px; }}
        h1 {{ margin-bottom: 20px; }}
        .grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
        .card {{
            background: white; border-radius: 8px; padding: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); position: relative;
        }}
        .card img {{ width: 250px; border-radius: 4px; cursor: pointer; }}
        .step-num {{
            position: absolute; top: 8px; left: 8px;
            background: #333; color: white; padding: 4px 8px;
            border-radius: 4px; font-size: 12px;
        }}
        .step-name {{ font-weight: bold; margin: 8px 0 4px 0; }}
        .info {{ margin: 0; padding-left: 20px; font-size: 12px; color: #666; }}
        .lightbox {{
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.9); z-index: 1000; cursor: pointer;
            align-items: center; justify-content: center;
        }}
        .lightbox.active {{ display: flex; }}
        .lightbox img {{ max-width: 90%; max-height: 90%; }}
    </style>
</head>
<body>
    <h1>Experiment: {Path(self.image_path).name if self.image_path else "unknown"}</h1>
    <div class="grid">
        {"".join(cards)}
    </div>
    <div class="lightbox" onclick="this.classList.remove('active')">
        <img src="">
    </div>
    <script>
        function showLightbox(img) {{
            const lb = document.querySelector('.lightbox');
            lb.querySelector('img').src = img.src;
            lb.classList.add('active');
        }}
    </script>
</body>
</html>'''

        Path(path).write_text(html)
        print(f"Saved: {path}")


# ============== Example Usage ==============

if __name__ == "__main__":
    exp = Experiment()

    # Load a random image
    exp.load_random_image(seed=42)

    # Define your instruction sequence here
    instructions = [
        # Preprocessing
        ("grayscale", {}),
        ("gaussian_blur", {"kernel_size": 5}),
        ("adaptive_threshold", {"block_size": 11, "c": 2}),

        # Try border detection
        ("detect_border", {"method": "simple_baseline"}),
    ]

    print("\nRunning instructions:")
    exp.run_instructions(instructions)

    # Save HTML report
    exp.save_report("reports/experiment.html")

    # Optionally show matplotlib window
    # exp.show_results()
