"""Map failure categories to likely parameter adjustments.

This module provides guidance on which parameters to tune based on
the diagnosed failure mode.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from .failure_diagnoser import FailureCode


@dataclass
class ParameterHint:
    """A hint for adjusting a parameter."""
    parameter: str
    direction: str  # "increase", "decrease", or "try_values"
    reason: str
    suggested_values: Optional[List] = None


# Mapping from failure codes to parameter adjustment hints
FAILURE_HINTS: Dict[str, List[ParameterHint]] = {
    FailureCode.AREA_REJECTED_SMALL: [
        ParameterHint(
            parameter="min_area_ratio",
            direction="decrease",
            reason="Contours rejected for being too small",
            suggested_values=[0.03, 0.02, 0.01],
        ),
    ],

    FailureCode.AREA_REJECTED_LARGE: [
        ParameterHint(
            parameter="max_area_ratio",
            direction="increase",
            reason="Contours rejected for being too large",
            suggested_values=[0.97, 0.98, 0.99],
        ),
    ],

    FailureCode.ANGLE_REJECTED: [
        ParameterHint(
            parameter="min_interior_angle",
            direction="decrease",
            reason="Quadrilateral interior angles are too acute",
            suggested_values=[40.0, 35.0, 30.0],
        ),
        ParameterHint(
            parameter="max_interior_angle",
            direction="increase",
            reason="Quadrilateral interior angles are too obtuse",
            suggested_values=[140.0, 145.0, 150.0],
        ),
    ],

    FailureCode.ASPECT_REJECTED: [
        ParameterHint(
            parameter="min_aspect_ratio",
            direction="decrease",
            reason="Aspect ratio too narrow (tall/thin quadrilateral)",
            suggested_values=[0.4, 0.3, 0.25],
        ),
        ParameterHint(
            parameter="max_aspect_ratio",
            direction="increase",
            reason="Aspect ratio too wide (short/wide quadrilateral)",
            suggested_values=[2.5, 3.0, 4.0],
        ),
    ],

    FailureCode.NO_QUADRILATERALS: [
        ParameterHint(
            parameter="contour_epsilon_factor",
            direction="increase",
            reason="Contour approximation too strict, not forming quadrilaterals",
            suggested_values=[0.025, 0.03, 0.04],
        ),
        ParameterHint(
            parameter="adaptive_block_size",
            direction="try_values",
            reason="Thresholding may be causing fragmented contours",
            suggested_values=[9, 15, 21],
        ),
    ],

    FailureCode.NO_CONTOURS: [
        ParameterHint(
            parameter="adaptive_c",
            direction="try_values",
            reason="Adaptive threshold constant may need adjustment",
            suggested_values=[0, 1, 3, 5],
        ),
        ParameterHint(
            parameter="clahe_clip_limit",
            direction="increase",
            reason="Low contrast may be preventing contour detection",
            suggested_values=[3.0, 4.0, 5.0],
        ),
    ],

    FailureCode.HOUGH_NO_LINES: [
        ParameterHint(
            parameter="hough_threshold",
            direction="decrease",
            reason="Hough accumulator threshold too high",
            suggested_values=[60, 50, 40],
        ),
        ParameterHint(
            parameter="hough_min_line_length",
            direction="decrease",
            reason="Minimum line length too long",
            suggested_values=[40, 30, 20],
        ),
        ParameterHint(
            parameter="canny_low",
            direction="decrease",
            reason="Canny edge thresholds may be too high",
            suggested_values=[40, 30, 20],
        ),
    ],

    FailureCode.HOUGH_LINE_COUNT_LOW: [
        ParameterHint(
            parameter="min_line_count",
            direction="decrease",
            reason="Requiring too many lines for validation",
            suggested_values=[7, 6, 5],
        ),
        ParameterHint(
            parameter="line_cluster_divisor",
            direction="increase",
            reason="Clustering threshold too aggressive, merging distinct lines",
            suggested_values=[40, 45, 50],
        ),
        ParameterHint(
            parameter="hough_threshold",
            direction="decrease",
            reason="Not detecting enough lines",
            suggested_values=[60, 50, 40],
        ),
    ],

    FailureCode.HOUGH_LINE_COUNT_HIGH: [
        ParameterHint(
            parameter="max_line_count",
            direction="increase",
            reason="Allowing too few lines for validation",
            suggested_values=[13, 14, 15],
        ),
        ParameterHint(
            parameter="line_cluster_divisor",
            direction="decrease",
            reason="Clustering threshold too permissive, not merging nearby lines",
            suggested_values=[32, 28, 24],
        ),
    ],

    FailureCode.HOUGH_FEW_HORIZONTAL: [
        ParameterHint(
            parameter="hough_threshold",
            direction="decrease",
            reason="Not detecting enough horizontal lines",
            suggested_values=[60, 50, 40],
        ),
    ],

    FailureCode.HOUGH_FEW_VERTICAL: [
        ParameterHint(
            parameter="hough_threshold",
            direction="decrease",
            reason="Not detecting enough vertical lines",
            suggested_values=[60, 50, 40],
        ),
    ],

    FailureCode.BOUNDARY_OVERSHOT: [
        ParameterHint(
            parameter="contour_epsilon_factor",
            direction="decrease",
            reason="Contour approximation including extra area",
            suggested_values=[0.015, 0.01, 0.008],
        ),
    ],

    FailureCode.BOUNDARY_UNDERSHOT: [
        ParameterHint(
            parameter="contour_epsilon_factor",
            direction="increase",
            reason="Contour approximation cutting off grid edges",
            suggested_values=[0.025, 0.03, 0.035],
        ),
    ],

    FailureCode.ALL_QUADS_INVALID: [
        ParameterHint(
            parameter="min_interior_angle",
            direction="decrease",
            reason="All quadrilaterals rejected by angle validation",
            suggested_values=[40.0, 35.0, 30.0],
        ),
        ParameterHint(
            parameter="max_interior_angle",
            direction="increase",
            reason="All quadrilaterals rejected by angle validation",
            suggested_values=[140.0, 145.0, 150.0],
        ),
    ],
}


def get_parameter_hints(failure_code: str) -> List[ParameterHint]:
    """Get parameter adjustment hints for a failure code.

    Args:
        failure_code: The failure code from diagnosis.

    Returns:
        List of ParameterHint objects suggesting adjustments.
    """
    return FAILURE_HINTS.get(failure_code, [])


def get_all_hints_for_failures(failure_codes: List[str]) -> Dict[str, List[ParameterHint]]:
    """Get parameter hints for multiple failure codes.

    Args:
        failure_codes: List of failure codes.

    Returns:
        Dictionary mapping parameters to lists of hints.
    """
    all_hints = {}

    for code in failure_codes:
        hints = get_parameter_hints(code)
        for hint in hints:
            if hint.parameter not in all_hints:
                all_hints[hint.parameter] = []
            all_hints[hint.parameter].append(hint)

    return all_hints


def format_hints_report(failure_codes: List[str]) -> str:
    """Format a human-readable report of parameter hints.

    Args:
        failure_codes: List of failure codes.

    Returns:
        Formatted string report.
    """
    lines = ["Parameter Adjustment Hints", "=" * 50]

    hints_by_param = get_all_hints_for_failures(failure_codes)

    if not hints_by_param:
        lines.append("No specific parameter hints for these failure codes.")
        return "\n".join(lines)

    for param, hints in sorted(hints_by_param.items()):
        lines.append(f"\n{param}:")
        for hint in hints:
            lines.append(f"  Direction: {hint.direction}")
            lines.append(f"  Reason: {hint.reason}")
            if hint.suggested_values:
                lines.append(f"  Try values: {hint.suggested_values}")

    return "\n".join(lines)


def suggest_config_changes(
    failure_summary: Dict[str, int],
    top_n: int = 3,
) -> List[Dict]:
    """Suggest config changes based on failure summary.

    Args:
        failure_summary: Dictionary mapping failure codes to counts.
        top_n: Number of top failures to address.

    Returns:
        List of suggested config changes with priority.
    """
    # Sort failures by count
    sorted_failures = sorted(failure_summary.items(), key=lambda x: -x[1])

    suggestions = []
    seen_params = set()

    for code, count in sorted_failures[:top_n]:
        hints = get_parameter_hints(code)

        for hint in hints:
            if hint.parameter not in seen_params:
                suggestions.append({
                    "parameter": hint.parameter,
                    "direction": hint.direction,
                    "reason": hint.reason,
                    "suggested_values": hint.suggested_values,
                    "failure_code": code,
                    "failure_count": count,
                })
                seen_params.add(hint.parameter)

    return suggestions
