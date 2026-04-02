from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SudokuGrid(BaseModel):
    """Input model for a Sudoku grid."""

    grid: List[List[int]] = Field(..., min_length=9, max_length=9)

    @field_validator("grid")
    @classmethod
    def validate_grid(cls, v: List[List[int]]) -> List[List[int]]:
        if len(v) != 9:
            raise ValueError("Grid must have exactly 9 rows")
        for i, row in enumerate(v):
            if len(row) != 9:
                raise ValueError(f"Row {i} must have exactly 9 columns")
            for j, val in enumerate(row):
                if not isinstance(val, int) or val < 0 or val > 9:
                    raise ValueError(f"Cell [{i}][{j}] must be an integer 0-9")
        return v


class SolveRequest(SudokuGrid):
    """Request model for solving — grid + solver method."""

    method: Literal["backtracking", "simulated_annealing"] = "backtracking"


class ExtractResponse(BaseModel):
    """Response model for grid extraction."""

    success: bool
    grid: Optional[List[List[int]]] = None
    confidence_map: Optional[List[List[float]]] = None
    message: str = ""


class SolveResponse(BaseModel):
    """Response model for solving a puzzle."""

    success: bool
    solution: Optional[List[List[int]]] = None
    iterations: int = 0
    method: str = "backtracking"
    solve_time_ms: float = 0.0
    message: str = ""
