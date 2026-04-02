"""
Test runner for border detection evaluation.

Executes detection methods on sample images and collects results.
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.border_detection import DetectorRegistry, BorderDetectionResult
from .sampler import sample_images, load_image, get_image_info


@dataclass
class ImageTestResult:
    """Result for a single image test."""

    image_path: str
    image_name: str
    image_info: Dict[str, Any]
    detection_result: BorderDetectionResult
    original_image: np.ndarray
    detection_image: np.ndarray  # Original with detection overlaid


@dataclass
class TestRunResults:
    """Results from a complete test run."""

    method: str
    params: Dict[str, Any]
    image_results: List[ImageTestResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    timestamp: str = ""

    @property
    def total(self) -> int:
        return len(self.image_results)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.image_results if r.detection_result.success)

    @property
    def failure_count(self) -> int:
        return self.total - self.success_count

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total if self.total > 0 else 0.0

    @property
    def avg_time_ms(self) -> float:
        if not self.image_results:
            return 0.0
        return sum(
            r.detection_result.execution_time_ms for r in self.image_results
        ) / len(self.image_results)


class TestRunner:
    """
    Runs border detection tests on sample images.

    Example:
        runner = TestRunner(
            method="simple_baseline",
            sample_size=5,
            seed=42
        )
        results = runner.run()
        print(f"Success rate: {results.success_rate:.1%}")
    """

    def __init__(
        self,
        method: str = "simple_baseline",
        sample_size: Optional[int] = None,
        image_paths: Optional[List[str]] = None,
        output_path: str = "reports/border_detection_report.html",
        pipeline_config: Optional[str] = None,
        params: Optional[List[str]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        aug_levels: Optional[List[int]] = None,
    ):
        """
        Initialize test runner.

        Args:
            method: Detection method name or "all"
            sample_size: Number of images to sample
            image_paths: Specific image paths (overrides sample_size)
            output_path: Path for HTML report
            pipeline_config: Path to pipeline JSON config
            params: List of "key=value" parameter overrides
            verbose: Verbosity level (0-3)
            seed: Random seed for sampling
            aug_levels: Filter by augmentation levels
        """
        self.method = method
        self.sample_size = sample_size
        self.image_paths = image_paths
        self.output_path = Path(output_path)
        self.pipeline_config = pipeline_config
        self.params = self._parse_params(params or [])
        self.verbose = verbose
        self.seed = seed
        self.aug_levels = aug_levels

    def _parse_params(self, param_list: List[str]) -> Dict[str, Any]:
        """Parse 'key=value' parameter strings."""
        params = {}
        for param in param_list:
            if "=" in param:
                key, value = param.split("=", 1)
                # Try to parse as int, float, or keep as string
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                params[key] = value
        return params

    def _get_images(self) -> List[Path]:
        """Get list of images to test."""
        if self.image_paths:
            return [Path(p) for p in self.image_paths]

        if self.sample_size:
            return sample_images(
                n=self.sample_size,
                aug_levels=self.aug_levels,
                seed=self.seed
            )

        return []

    def _get_methods(self) -> List[str]:
        """Get list of methods to test."""
        if self.method == "all":
            return DetectorRegistry.list_all()
        return [self.method]

    def run(self) -> TestRunResults:
        """
        Run tests and return results.

        Returns:
            TestRunResults with all image results
        """
        from datetime import datetime

        images = self._get_images()
        methods = self._get_methods()

        if not images:
            print("No images to test!")
            return TestRunResults(
                method=self.method,
                params=self.params,
                timestamp=datetime.now().isoformat()
            )

        # For now, run first method (multi-method comparison in report)
        method_name = methods[0]

        if self.verbose:
            print(f"Running {method_name} on {len(images)} images...")

        detector = DetectorRegistry.get(method_name)
        merged_params = {**detector.get_default_params(), **self.params}

        results = TestRunResults(
            method=method_name,
            params=merged_params,
            timestamp=datetime.now().isoformat()
        )

        start_time = time.perf_counter()

        for i, img_path in enumerate(images):
            if self.verbose >= 2:
                print(f"  [{i+1}/{len(images)}] {img_path.name}")

            image = load_image(img_path)
            if image is None:
                if self.verbose:
                    print(f"    Failed to load image")
                continue

            # Run detection
            detection = detector.detect(image, **merged_params)

            # Create detection visualization
            detection_image = detection.draw_on_image(image)

            result = ImageTestResult(
                image_path=str(img_path),
                image_name=img_path.name,
                image_info=get_image_info(img_path),
                detection_result=detection,
                original_image=image,
                detection_image=detection_image,
            )

            results.image_results.append(result)

            if self.verbose >= 2:
                status = "OK" if detection.success else "FAIL"
                print(f"    {status} (conf={detection.confidence:.2f}, time={detection.execution_time_ms:.1f}ms)")

        results.total_time_ms = (time.perf_counter() - start_time) * 1000

        if self.verbose:
            print(f"\nResults: {results.success_count}/{results.total} succeeded ({results.success_rate:.1%})")
            print(f"Average time: {results.avg_time_ms:.1f}ms")

        return results

    def run_comparison(self) -> Dict[str, TestRunResults]:
        """
        Run tests with all methods for comparison.

        Returns:
            Dict mapping method name to TestRunResults
        """
        images = self._get_images()
        methods = self._get_methods()

        all_results = {}

        for method_name in methods:
            if self.verbose:
                print(f"\n=== {method_name} ===")

            detector = DetectorRegistry.get(method_name)
            merged_params = {**detector.get_default_params(), **self.params}

            results = TestRunResults(
                method=method_name,
                params=merged_params,
                timestamp=""
            )

            for img_path in images:
                image = load_image(img_path)
                if image is None:
                    continue

                detection = detector.detect(image, **merged_params)
                detection_image = detection.draw_on_image(image)

                result = ImageTestResult(
                    image_path=str(img_path),
                    image_name=img_path.name,
                    image_info=get_image_info(img_path),
                    detection_result=detection,
                    original_image=image,
                    detection_image=detection_image,
                )

                results.image_results.append(result)

            all_results[method_name] = results

            if self.verbose:
                print(f"  {results.success_rate:.1%} success rate")

        return all_results
