"""Sudoku extraction and solving API endpoints."""

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile

from app.core.extraction import (
    extract_cells,
    find_grid_contour,
    perspective_transform,
    preprocess_image,
    recognize_cells,
)
from app.core.solver import solve
from app.core.verifier import validate_puzzle, verify_solution
from app.models.schemas import ExtractResponse, SolveRequest, SolveResponse

router = APIRouter()


@router.post("/extract", response_model=ExtractResponse)
async def extract_grid(file: UploadFile = File(...)):
    """Extract Sudoku grid from uploaded image using CNN OCR."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return ExtractResponse(
                success=False,
                message="Invalid image file",
            )

        # Detection
        thresh = preprocess_image(image)
        contour = find_grid_contour(thresh)

        if contour is None:
            return ExtractResponse(
                success=False,
                message="Could not detect Sudoku grid in image.",
            )

        # Cell extraction + OCR
        warped = perspective_transform(image, contour)
        cells = extract_cells(warped)
        grid, confidence_map = recognize_cells(cells)

        return ExtractResponse(
            success=True,
            grid=grid,
            confidence_map=confidence_map,
            message="Grid extracted successfully",
        )

    except Exception as e:
        return ExtractResponse(
            success=False,
            message=f"Error processing image: {str(e)}",
        )


@router.post("/solve", response_model=SolveResponse)
async def solve_sudoku(data: SolveRequest):
    """Solve a Sudoku puzzle.

    Supports backtracking (default, deterministic) and
    simulated_annealing (probabilistic) methods.
    """
    grid = data.grid
    method = data.method

    # Validate puzzle before solving
    valid, errors = validate_puzzle(grid)
    if not valid:
        return SolveResponse(
            success=False,
            message=f"Invalid puzzle: {'; '.join(errors)}",
        )

    solution, iterations, success, elapsed_ms = solve(grid, method)

    if success and verify_solution(solution):
        return SolveResponse(
            success=True,
            solution=solution,
            iterations=iterations,
            method=method,
            solve_time_ms=round(elapsed_ms, 1),
            message=f"Solved in {iterations} steps ({elapsed_ms:.1f}ms)",
        )
    else:
        return SolveResponse(
            success=False,
            solution=solution,
            iterations=iterations,
            method=method,
            solve_time_ms=round(elapsed_ms, 1),
            message="Could not find valid solution.",
        )
