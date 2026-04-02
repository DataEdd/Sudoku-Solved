"""
Base classes for modular preprocessing pipeline.

This module provides the foundation for building composable, configurable
preprocessing pipelines. Each step can be parameterized and reordered.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class StepResult:
    """Result from a single preprocessing step."""

    name: str                               # Step identifier (e.g., "gaussian_blur")
    output: np.ndarray                      # Resulting image
    params: Dict[str, Any]                  # Parameters used for this execution
    execution_time_ms: float                # Time taken in milliseconds
    metadata: Dict[str, Any] = field(default_factory=dict)  # Optional extra info

    def __repr__(self) -> str:
        shape = self.output.shape if self.output is not None else None
        return f"StepResult(name={self.name!r}, shape={shape}, time={self.execution_time_ms:.1f}ms)"


@dataclass
class PipelineConfig:
    """
    Configuration for a preprocessing pipeline.

    Attributes:
        name: Human-readable pipeline name
        steps: List of (step_name, params) tuples defining the pipeline

    Example:
        config = PipelineConfig(
            name="standard_hough",
            steps=[
                ("grayscale", {}),
                ("gaussian_blur", {"kernel_size": 5}),
                ("adaptive_threshold", {"block_size": 11, "c": 2}),
            ]
        )
    """

    name: str
    steps: List[Tuple[str, Dict[str, Any]]]

    def __repr__(self) -> str:
        step_names = [s[0] for s in self.steps]
        return f"PipelineConfig(name={self.name!r}, steps={step_names})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "steps": [{"name": name, "params": params} for name, params in self.steps]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create from dictionary."""
        steps = [(s["name"], s.get("params", {})) for s in data["steps"]]
        return cls(name=data["name"], steps=steps)


class PreprocessingStep(ABC):
    """
    Abstract base class for preprocessing steps.

    Each step represents a single image transformation that can be
    composed into a pipeline. Steps should be stateless and idempotent.

    To create a custom step:
        class MyStep(PreprocessingStep):
            name = "my_step"
            description = "Does something cool"

            def process(self, image: np.ndarray, my_param: int = 10) -> np.ndarray:
                # Transform image
                return result

            def get_default_params(self) -> Dict[str, Any]:
                return {"my_param": 10}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this step.

        Used in pipeline configuration and registry lookup.
        Should be lowercase with underscores (e.g., "gaussian_blur").
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of what this step does.

        Shown in reports and CLI help.
        """
        pass

    @abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply preprocessing step to image.

        Args:
            image: Input image (grayscale or BGR depending on step)
            **kwargs: Step-specific parameters

        Returns:
            Processed image as numpy array
        """
        pass

    def get_default_params(self) -> Dict[str, Any]:
        """
        Return default parameters for this step.

        Override this to specify default values for step parameters.
        These are merged with user-provided params at runtime.

        Returns:
            Dict of parameter_name -> default_value
        """
        return {}

    def validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validate parameters before processing.

        Override to add parameter validation. Raise ValueError if invalid.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
