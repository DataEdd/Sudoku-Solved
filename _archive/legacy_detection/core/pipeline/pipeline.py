"""
Pipeline orchestrator for composable preprocessing.

This module provides the PreprocessingPipeline class that:
- Manages step registration and lookup
- Executes pipelines with intermediate result capture
- Supports JSON configuration for reproducibility
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np

from .base import PipelineConfig, PreprocessingStep, StepResult


class PreprocessingPipeline:
    """
    Composable, configurable preprocessing pipeline.

    Example usage:
        # Create pipeline from config
        config = PipelineConfig(
            name="my_pipeline",
            steps=[
                ("grayscale", {}),
                ("gaussian_blur", {"kernel_size": 5}),
                ("adaptive_threshold", {"block_size": 11}),
            ]
        )
        pipeline = PreprocessingPipeline(config)

        # Run and get all intermediate results
        results = pipeline.run(image)
        for r in results:
            print(f"{r.name}: {r.output.shape}")

        # Or just get final output
        final = pipeline.get_final_output(image)
    """

    # Class-level registry of available steps
    _step_registry: Dict[str, Type[PreprocessingStep]] = {}

    @classmethod
    def register_step(cls, step_class: Type[PreprocessingStep]) -> None:
        """
        Register a preprocessing step type.

        Args:
            step_class: PreprocessingStep subclass to register
        """
        # Instantiate to get the name property
        instance = step_class()
        cls._step_registry[instance.name] = step_class

    @classmethod
    def get_step_class(cls, name: str) -> Type[PreprocessingStep]:
        """
        Get a registered step class by name.

        Args:
            name: Step identifier

        Returns:
            PreprocessingStep subclass

        Raises:
            ValueError: If step not found
        """
        if name not in cls._step_registry:
            available = ", ".join(sorted(cls._step_registry.keys()))
            raise ValueError(f"Unknown step: {name}. Available: {available}")
        return cls._step_registry[name]

    @classmethod
    def list_steps(cls) -> List[str]:
        """List all registered step names."""
        return sorted(cls._step_registry.keys())

    @classmethod
    def get_step_info(cls) -> List[Dict[str, Any]]:
        """Get info about all registered steps."""
        info = []
        for name in sorted(cls._step_registry.keys()):
            step = cls._step_registry[name]()
            info.append({
                "name": name,
                "description": step.description,
                "default_params": step.get_default_params(),
            })
        return info

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline from configuration.

        Args:
            config: PipelineConfig specifying steps and parameters
        """
        self.config = config
        self.steps: List[tuple[PreprocessingStep, Dict[str, Any]]] = []
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Instantiate steps from config."""
        for step_name, params in self.config.steps:
            step_class = self.get_step_class(step_name)
            step = step_class()

            # Merge default params with user params
            merged_params = {**step.get_default_params(), **params}

            # Validate
            step.validate_params(merged_params)

            self.steps.append((step, merged_params))

    def run(self, image: np.ndarray) -> List[StepResult]:
        """
        Execute pipeline, returning all intermediate results.

        Args:
            image: Input image (typically BGR)

        Returns:
            List of StepResult, one per step
        """
        results = []
        current = image.copy()

        for step, params in self.steps:
            start = time.perf_counter()
            output = step.process(current, **params)
            elapsed_ms = (time.perf_counter() - start) * 1000

            results.append(StepResult(
                name=step.name,
                output=output,
                params=params,
                execution_time_ms=elapsed_ms,
            ))

            current = output

        return results

    def get_final_output(self, image: np.ndarray) -> np.ndarray:
        """
        Run pipeline and return only final output.

        Args:
            image: Input image

        Returns:
            Final processed image
        """
        results = self.run(image)
        return results[-1].output if results else image

    def get_step_descriptions(self) -> List[Dict[str, Any]]:
        """Get descriptions of all steps in this pipeline."""
        descriptions = []
        for step, params in self.steps:
            descriptions.append({
                "name": step.name,
                "description": step.description,
                "params": params,
            })
        return descriptions

    @classmethod
    def from_json(cls, json_path: Path) -> "PreprocessingPipeline":
        """
        Load pipeline from JSON configuration file.

        Args:
            json_path: Path to JSON config file

        Returns:
            Configured PreprocessingPipeline
        """
        with open(json_path) as f:
            data = json.load(f)
        config = PipelineConfig.from_dict(data)
        return cls(config)

    def to_json(self, json_path: Path) -> None:
        """
        Save pipeline configuration to JSON file.

        Args:
            json_path: Path to write JSON config
        """
        with open(json_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        step_names = [s.name for s, _ in self.steps]
        return f"PreprocessingPipeline(name={self.config.name!r}, steps={step_names})"


# Predefined pipeline configurations
STANDARD_HOUGH_CONFIG = PipelineConfig(
    name="standard_hough",
    steps=[
        ("grayscale", {}),
        ("gaussian_blur", {"kernel_size": 5}),
        ("adaptive_threshold", {"block_size": 11, "c": 2}),
    ]
)

CANNY_EDGE_CONFIG = PipelineConfig(
    name="canny_edge",
    steps=[
        ("grayscale", {}),
        ("gaussian_blur", {"kernel_size": 5}),
        ("canny", {"low_threshold": 50, "high_threshold": 150}),
    ]
)

SOBEL_GRADIENT_CONFIG = PipelineConfig(
    name="sobel_gradient",
    steps=[
        ("grayscale", {}),
        ("gaussian_blur", {"kernel_size": 5}),
        ("sobel", {"ksize": 3, "normalize": True}),
    ]
)

ROBUST_EDGE_CONFIG = PipelineConfig(
    name="robust_edge",
    steps=[
        ("grayscale", {}),
        ("bilateral", {"d": 9, "sigma_color": 75, "sigma_space": 75}),
        ("canny", {"low_threshold": 30, "high_threshold": 100}),
        ("morphology", {"operation": "dilate", "kernel_size": 3}),
    ]
)

# Dict of predefined configs for easy access
PREDEFINED_PIPELINES: Dict[str, PipelineConfig] = {
    "standard_hough": STANDARD_HOUGH_CONFIG,
    "canny_edge": CANNY_EDGE_CONFIG,
    "sobel_gradient": SOBEL_GRADIENT_CONFIG,
    "robust_edge": ROBUST_EDGE_CONFIG,
}
