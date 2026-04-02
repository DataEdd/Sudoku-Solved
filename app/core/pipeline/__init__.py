"""
Modular preprocessing pipeline for image processing.

This package provides a composable, configurable pipeline system
for preprocessing images before border detection.

Example:
    from app.core.pipeline import PreprocessingPipeline, PipelineConfig

    config = PipelineConfig(
        name="my_pipeline",
        steps=[
            ("grayscale", {}),
            ("gaussian_blur", {"kernel_size": 5}),
            ("canny", {"low_threshold": 50, "high_threshold": 150}),
        ]
    )

    pipeline = PreprocessingPipeline(config)
    results = pipeline.run(image)

    # Access intermediate results
    for result in results:
        print(f"{result.name}: {result.output.shape}")
"""

from .base import PipelineConfig, PreprocessingStep, StepResult
from .pipeline import (
    PREDEFINED_PIPELINES,
    PreprocessingPipeline,
)
from .steps import (
    BUILTIN_STEPS,
    AdaptiveThresholdStep,
    BilateralFilterStep,
    BinaryThresholdStep,
    CannyEdgeStep,
    ContrastStep,
    GaussianBlurStep,
    GrayscaleStep,
    InvertStep,
    MedianBlurStep,
    MorphologyStep,
    SobelGradientStep,
)

# Auto-register all built-in steps
for step_class in BUILTIN_STEPS:
    PreprocessingPipeline.register_step(step_class)


__all__ = [
    # Base classes
    "PreprocessingStep",
    "StepResult",
    "PipelineConfig",
    # Pipeline
    "PreprocessingPipeline",
    "PREDEFINED_PIPELINES",
    # Steps
    "GrayscaleStep",
    "GaussianBlurStep",
    "AdaptiveThresholdStep",
    "BinaryThresholdStep",
    "SobelGradientStep",
    "CannyEdgeStep",
    "MorphologyStep",
    "MedianBlurStep",
    "BilateralFilterStep",
    "ContrastStep",
    "InvertStep",
    "BUILTIN_STEPS",
]
