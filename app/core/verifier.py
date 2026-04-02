"""
Sudoku puzzle verification.

Validates puzzle rules and checks solution correctness.
"""

from typing import List, Tuple


def validate_puzzle(grid: List[List[int]]) -> Tuple[bool, List[str]]:
    """Check a puzzle for rule violations (duplicate digits).

    Only checks filled cells — zeros (empty) are ignored.

    Returns:
        (is_valid, list of error descriptions)
    """
    errors = []

    # Row duplicates
    for i in range(9):
        seen = {}
        for j in range(9):
            v = grid[i][j]
            if v == 0:
                continue
            if v in seen:
                errors.append(
                    f"Row {i + 1}: duplicate {v} at columns {seen[v] + 1} and {j + 1}"
                )
            else:
                seen[v] = j

    # Column duplicates
    for j in range(9):
        seen = {}
        for i in range(9):
            v = grid[i][j]
            if v == 0:
                continue
            if v in seen:
                errors.append(
                    f"Col {j + 1}: duplicate {v} at rows {seen[v] + 1} and {i + 1}"
                )
            else:
                seen[v] = i

    # Box duplicates
    for br in range(3):
        for bc in range(3):
            seen = {}
            for i in range(3):
                for j in range(3):
                    r, c = br * 3 + i, bc * 3 + j
                    v = grid[r][c]
                    if v == 0:
                        continue
                    if v in seen:
                        errors.append(f"Box ({br + 1},{bc + 1}): duplicate {v}")
                    else:
                        seen[v] = (r, c)

    return len(errors) == 0, errors


def verify_solution(solution: List[List[int]]) -> bool:
    """Check that a completed grid is a valid Sudoku solution.

    Every row, column, and 3x3 box must contain digits 1-9 exactly once.
    """
    target = set(range(1, 10))

    for i in range(9):
        if set(solution[i]) != target:
            return False
        if {solution[r][i] for r in range(9)} != target:
            return False

    for br in range(3):
        for bc in range(3):
            box = set()
            for i in range(3):
                for j in range(3):
                    box.add(solution[br * 3 + i][bc * 3 + j])
            if box != target:
                return False

    return True
