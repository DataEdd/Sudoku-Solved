"""
Solver latency benchmark.

Runs the production backtracking solver and the simulated-annealing solver
against the 38 ground-truth Sudoku grids shipped in ``ground_truth.json``.

Backtracking is fast and deterministic, so each puzzle is solved 10 times and
per-call latencies are aggregated. Simulated annealing is slow and stochastic,
so each puzzle is solved once and its success/latency is recorded.

Usage:
    python -m evaluation.benchmark_solver
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.solver import backtracking, simulated_annealing

GT_PATH = Path(__file__).parent / "ground_truth.json"
OUTPUT_PATH = Path(__file__).parent / "solver_benchmark_results.json"

BACKTRACKING_RUNS_PER_PUZZLE = 10
SA_RUNS_PER_PUZZLE = 1


def _coerce_cell(cell: Any) -> int:
    """Multi-value GT cells (lists) are acceptable alternatives for OCR.

    For the solver benchmark we need a single integer per cell, so we treat
    any list-valued cell as empty (0). The solver never sees OCR predictions,
    only the clue digits that the photographer-provided puzzle contained.
    """
    if isinstance(cell, list):
        return 0
    return int(cell)


def _coerce_grid(grid: List[List[Any]]) -> List[List[int]]:
    return [[_coerce_cell(c) for c in row] for row in grid]


def _load_puzzles() -> List[Dict[str, Any]]:
    with open(GT_PATH) as f:
        data = json.load(f)

    puzzles: List[Dict[str, Any]] = []
    for entry in data["images"]:
        grid = _coerce_grid(entry["grid"])
        puzzles.append({"path": entry["path"], "grid": grid})
    return puzzles


def _percentile(values: List[float], pct: float) -> float:
    """Linear-interpolation percentile, matches numpy.percentile default."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _summarize(values: List[float]) -> Dict[str, float]:
    return {
        "min_ms": round(min(values), 4),
        "median_ms": round(statistics.median(values), 4),
        "mean_ms": round(statistics.mean(values), 4),
        "p95_ms": round(_percentile(values, 95), 4),
        "max_ms": round(max(values), 4),
    }


def benchmark_backtracking(puzzles: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_times_ms: List[float] = []
    solvable_times_ms: List[float] = []
    per_puzzle: List[Dict[str, Any]] = []
    failures: List[str] = []

    for entry in puzzles:
        runs_ms: List[float] = []
        all_successful = True
        for _ in range(BACKTRACKING_RUNS_PER_PUZZLE):
            t0 = time.perf_counter()
            solution, nodes, success = backtracking(entry["grid"])
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            runs_ms.append(elapsed_ms)
            if not success:
                all_successful = False

        if not all_successful:
            failures.append(entry["path"])
        else:
            solvable_times_ms.extend(runs_ms)

        all_times_ms.extend(runs_ms)
        per_puzzle.append(
            {
                "path": entry["path"],
                "runs_ms": [round(x, 4) for x in runs_ms],
                "median_ms": round(statistics.median(runs_ms), 4),
                "success": all_successful,
            }
        )

    summary = {
        "runs_per_puzzle": BACKTRACKING_RUNS_PER_PUZZLE,
        "total_runs": len(all_times_ms),
        "all_successful": not failures,
        "failed_paths": failures,
        "puzzles_total": len(puzzles),
        "puzzles_solvable": len(puzzles) - len(failures),
        "all_runs": _summarize(all_times_ms),
        "solvable_runs": _summarize(solvable_times_ms) if solvable_times_ms else None,
        "per_puzzle": per_puzzle,
    }
    return summary


def benchmark_simulated_annealing(puzzles: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_puzzle: List[Dict[str, Any]] = []
    success_times_ms: List[float] = []
    all_times_ms: List[float] = []
    successes = 0

    for entry in puzzles:
        t0 = time.perf_counter()
        solution, iters, success = simulated_annealing(entry["grid"])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        all_times_ms.append(elapsed_ms)
        if success:
            successes += 1
            success_times_ms.append(elapsed_ms)
        per_puzzle.append(
            {
                "path": entry["path"],
                "success": success,
                "iterations": iters,
                "time_ms": round(elapsed_ms, 4),
            }
        )

    # Summarize over successful runs if we have any; otherwise over everything.
    summary_values = success_times_ms if success_times_ms else all_times_ms
    summary = _summarize(summary_values) if summary_values else {
        "min_ms": 0.0,
        "median_ms": 0.0,
        "mean_ms": 0.0,
        "p95_ms": 0.0,
        "max_ms": 0.0,
    }
    summary.update(
        {
            "runs_per_puzzle": SA_RUNS_PER_PUZZLE,
            "total_runs": len(puzzles),
            "successes": successes,
            "stats_over": "successful_runs" if success_times_ms else "all_runs",
            "per_puzzle": per_puzzle,
        }
    )
    return summary


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _print_stats(label: str, stats: Dict[str, float]) -> None:
    print(
        f"  {label:<14}  min={stats['min_ms']:.4f}  "
        f"median={stats['median_ms']:.4f}  mean={stats['mean_ms']:.4f}  "
        f"p95={stats['p95_ms']:.4f}  max={stats['max_ms']:.4f}"
    )


def _print_backtracking_summary(summary: Dict[str, Any]) -> None:
    print("\n[backtracking]")
    print(f"  runs_per_puzzle : {summary['runs_per_puzzle']}")
    print(f"  total_runs      : {summary['total_runs']}")
    print(f"  puzzles_total   : {summary['puzzles_total']}")
    print(f"  puzzles_solvable: {summary['puzzles_solvable']}")
    print(f"  all_successful  : {summary['all_successful']}")
    if summary["failed_paths"]:
        print("  failed_paths    :")
        for p in summary["failed_paths"]:
            print(f"    - {p}")
    print()
    _print_stats("all_runs (ms)", summary["all_runs"])
    if summary["solvable_runs"] is not None:
        _print_stats("solvable (ms)", summary["solvable_runs"])


def _print_sa_summary(summary: Dict[str, Any]) -> None:
    print("\n[simulated_annealing]")
    print(f"  runs_per_puzzle : {summary['runs_per_puzzle']}")
    print(f"  total_runs      : {summary['total_runs']}")
    print(f"  successes       : {summary['successes']}/{summary['total_runs']}")
    print(f"  stats_over      : {summary['stats_over']}")
    print(
        f"  ms: min={summary['min_ms']:.4f}  median={summary['median_ms']:.4f}  "
        f"mean={summary['mean_ms']:.4f}  p95={summary['p95_ms']:.4f}  "
        f"max={summary['max_ms']:.4f}"
    )


def main() -> None:
    puzzles = _load_puzzles()
    print(f"Loaded {len(puzzles)} puzzles from {GT_PATH.name}")

    _print_header("Backtracking benchmark")
    bt = benchmark_backtracking(puzzles)
    _print_backtracking_summary(bt)

    _print_header("Simulated annealing benchmark")
    sa = benchmark_simulated_annealing(puzzles)
    _print_sa_summary(sa)

    _print_header("Headline")
    headline_stats = bt["solvable_runs"] if bt["solvable_runs"] is not None else bt["all_runs"]
    print(
        f"  >> Backtracking median latency (solvable subset, "
        f"{bt['puzzles_solvable']}/{bt['puzzles_total']} puzzles): "
        f"{headline_stats['median_ms']:.4f} ms"
    )
    print(
        f"  >> Backtracking median latency (full 38/38, including fast-fail invalid GT): "
        f"{bt['all_runs']['median_ms']:.4f} ms"
    )
    print(
        f"  >> SA success rate: {sa['successes']}/{sa['total_runs']} "
        f"(median {sa['median_ms']:.1f} ms over {sa['stats_over']})"
    )

    payload = {
        "command": "python -m evaluation.benchmark_solver",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": (
            f"Backtracking run {BACKTRACKING_RUNS_PER_PUZZLE}x per puzzle "
            f"({BACKTRACKING_RUNS_PER_PUZZLE * len(puzzles)} total); "
            f"simulated annealing run {SA_RUNS_PER_PUZZLE}x per puzzle. "
            "Multi-value GT cells (lists) are treated as empty (0) for the solver, "
            "since they represent OCR ambiguity not puzzle clues. "
            "Backtracking reports two summaries: 'all_runs' covers every puzzle, "
            "'solvable_runs' excludes GT puzzles whose transcribed clues violate "
            "Sudoku constraints (duplicate digits in a row/col/box) — the solver "
            "correctly rejects those as unsolvable. The solvable-run median is the "
            "headline number since it reflects solver performance on well-formed "
            "inputs."
        ),
        "backtracking": bt,
        "simulated_annealing": sa,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
