"""
Sudoku solver using simulated annealing.

The algorithm works by:
1. Initializing each 3x3 box with the missing numbers (randomly placed)
2. Swapping non-fixed cells within boxes to minimize row/column conflicts
3. Using simulated annealing to escape local minima
"""

import asyncio
import math
import random
from typing import List, Tuple

import numpy as np

from app.config import get_settings


def calculate_energy(grid: np.ndarray, fixed: np.ndarray) -> int:
    """
    Calculate the number of conflicts (energy) in the grid.

    Only counts row and column conflicts since boxes are handled
    by initialization and swap constraints.
    """
    conflicts = 0

    # Row conflicts
    for i in range(9):
        row = grid[i, :]
        conflicts += 9 - len(set(row))

    # Column conflicts
    for j in range(9):
        col = grid[:, j]
        conflicts += 9 - len(set(col))

    return conflicts


def initialize_grid(puzzle: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """
    Initialize grid by filling empty cells with missing numbers in each 3x3 box.

    Each box will contain all digits 1-9 (no box conflicts).
    """
    grid = puzzle.copy()

    for box_row in range(3):
        for box_col in range(3):
            r_start = box_row * 3
            c_start = box_col * 3

            # Get existing numbers in this box
            existing = set()
            for i in range(3):
                for j in range(3):
                    val = grid[r_start + i, c_start + j]
                    if val != 0:
                        existing.add(val)

            # Find missing numbers
            missing = [x for x in range(1, 10) if x not in existing]
            random.shuffle(missing)

            # Fill empty cells
            idx = 0
            for i in range(3):
                for j in range(3):
                    if grid[r_start + i, c_start + j] == 0:
                        grid[r_start + i, c_start + j] = missing[idx]
                        idx += 1

    return grid


def get_neighbor(grid: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Generate a neighbor by swapping two non-fixed cells in a random 3x3 box."""
    neighbor = grid.copy()

    # Pick a random 3x3 box
    box_row = random.randint(0, 2)
    box_col = random.randint(0, 2)
    r_start = box_row * 3
    c_start = box_col * 3

    # Find non-fixed cells in this box
    non_fixed = []
    for i in range(3):
        for j in range(3):
            r, c = r_start + i, c_start + j
            if not fixed[r, c]:
                non_fixed.append((r, c))

    # Need at least 2 non-fixed cells to swap
    if len(non_fixed) < 2:
        return neighbor

    # Pick two random cells and swap
    idx1, idx2 = random.sample(range(len(non_fixed)), 2)
    r1, c1 = non_fixed[idx1]
    r2, c2 = non_fixed[idx2]

    neighbor[r1, c1], neighbor[r2, c2] = neighbor[r2, c2], neighbor[r1, c1]

    return neighbor


def simulated_annealing(
    puzzle: List[List[int]],
    initial_temp: float = 1.0,
    cooling_rate: float = 0.99999,
    max_iterations: int = 500000,
) -> Tuple[List[List[int]], int, bool]:
    """
    Solve Sudoku using simulated annealing.

    Args:
        puzzle: 9x9 grid with 0 for empty cells
        initial_temp: Starting temperature
        cooling_rate: Temperature decay rate per iteration
        max_iterations: Maximum iterations before stopping

    Returns:
        Tuple of (solution_grid, iterations, success)
    """
    puzzle_arr = np.array(puzzle)
    fixed = puzzle_arr != 0

    # Initialize
    current = initialize_grid(puzzle_arr, fixed)
    current_energy = calculate_energy(current, fixed)

    best = current.copy()
    best_energy = current_energy

    temp = initial_temp

    for iteration in range(max_iterations):
        if current_energy == 0:
            return current.tolist(), iteration, True

        neighbor = get_neighbor(current, fixed)
        neighbor_energy = calculate_energy(neighbor, fixed)

        delta = neighbor_energy - current_energy

        # Accept better solutions or probabilistically accept worse ones
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = neighbor
            current_energy = neighbor_energy

            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy

        temp *= cooling_rate

        # Reheat if stuck
        if iteration % 100000 == 0 and iteration > 0 and best_energy > 0:
            temp = initial_temp * 0.5

    return best.tolist(), max_iterations, best_energy == 0


async def solve_sudoku_async(
    puzzle: List[List[int]],
) -> Tuple[List[List[int]], int, bool]:
    """
    Async wrapper for simulated annealing solver.

    Uses settings from configuration.
    """
    settings = get_settings()
    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(
        None,
        simulated_annealing,
        puzzle,
        settings.initial_temp,
        settings.cooling_rate,
        settings.max_iterations,
    )
