"""
Sudoku solver: MRV-ordered backtracking with per-cell domain restriction.

The algorithm is the classical CSP approach described in Kamal, Chawla &
Goel, "Detection of Sudoku Puzzle using Image Processing and Solving by
Backtracking, Simulated Annealing and Genetic Algorithms: A Comparative
Analysis" (ICIIP 2015), extended with the Minimum Remaining Value (MRV)
heuristic so that each recursive step expands the empty cell with the
smallest candidate set first. That shrinks the average branching factor
and prunes dead-end branches earlier than naive row-major expansion.
"""

import asyncio
import time
from typing import List, Tuple


def backtracking(
    puzzle: List[List[int]],
) -> Tuple[List[List[int]], int, bool]:
    """Solve a Sudoku puzzle via MRV-ordered backtracking.

    At each recursive step the solver picks the empty cell with the
    fewest remaining candidates, tries each candidate in turn, and
    recurses. On a dead end (a cell whose candidate set is empty) it
    returns False so the parent frame can try the next value.

    Args:
        puzzle: 9x9 grid with 0 for empty cells.

    Returns:
        (solution_grid, nodes_explored, success)
    """
    grid = [row[:] for row in puzzle]
    nodes = [0]

    def candidates(r: int, c: int) -> set:
        used = set(grid[r])
        used |= {grid[i][c] for i in range(9)}
        br, bc = 3 * (r // 3), 3 * (c // 3)
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                used.add(grid[i][j])
        return set(range(1, 10)) - used

    def solve() -> bool:
        # MRV: pick the empty cell with the fewest candidates
        best, best_cands = None, None
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    cands = candidates(i, j)
                    if not cands:
                        return False  # dead end
                    if best is None or len(cands) < len(best_cands):
                        best = (i, j)
                        best_cands = cands
                        if len(cands) == 1:
                            break  # forced move, no need to keep scanning
            else:
                continue
            break

        if best is None:
            return True  # grid is full — puzzle solved

        r, c = best
        for val in best_cands:
            nodes[0] += 1
            grid[r][c] = val
            if solve():
                return True
            grid[r][c] = 0  # backtrack

        return False

    success = solve()
    return grid, nodes[0], success


def solve(
    puzzle: List[List[int]],
) -> Tuple[List[List[int]], int, bool, float]:
    """Solve a Sudoku puzzle and return timing.

    Args:
        puzzle: 9x9 grid with 0 for empty cells.

    Returns:
        (solution, nodes_explored, success, elapsed_ms)
    """
    t0 = time.time()
    solution, nodes, success = backtracking(puzzle)
    elapsed_ms = (time.time() - t0) * 1000
    return solution, nodes, success, elapsed_ms


async def solve_sudoku_async(
    puzzle: List[List[int]],
) -> Tuple[List[List[int]], int, bool, float]:
    """Async wrapper for solve()."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, solve, puzzle)
