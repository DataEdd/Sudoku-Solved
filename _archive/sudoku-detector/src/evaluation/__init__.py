"""Evaluation module for sudoku detection.

This module provides tools for:
- Diagnosing why detection failed on specific images
- Mapping failures to parameter adjustments
- Tracking detection quality metrics
"""

from .failure_diagnoser import FailureDiagnoser, DiagnosticResult
from .parameter_hints import FAILURE_HINTS, get_parameter_hints

__all__ = [
    "FailureDiagnoser",
    "DiagnosticResult",
    "FAILURE_HINTS",
    "get_parameter_hints",
]
