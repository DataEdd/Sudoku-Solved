import io
import base64
import random
import math
from typing import List, Optional

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

app = FastAPI(title="Sudoku Solver")
templates = Jinja2Templates(directory="templates")


# ============== Models ==============

class SudokuGrid(BaseModel):
    grid: List[List[int]]


class SolveResponse(BaseModel):
    success: bool
    solution: Optional[List[List[int]]] = None
    iterations: int = 0
    message: str = ""


class ExtractResponse(BaseModel):
    success: bool
    grid: Optional[List[List[int]]] = None
    message: str = ""


# ============== Image Processing ==============

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert image to binary for grid detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh


def find_grid_contour(thresh: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest square contour (the Sudoku grid)."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Sort by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Looking for a quadrilateral
        if len(approx) == 4:
            return approx

    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    pts = pts.reshape(4, 2)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def perspective_transform(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Apply perspective transform to get a top-down view of the grid."""
    pts = order_points(contour)

    # Compute dimensions of the new image
    width = max(
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[2] - pts[3])
    )
    height = max(
        np.linalg.norm(pts[0] - pts[3]),
        np.linalg.norm(pts[1] - pts[2])
    )

    # Make it square
    size = max(int(width), int(height))

    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, matrix, (size, size))

    return warped


def extract_cells(grid_image: np.ndarray) -> List[np.ndarray]:
    """Split the grid image into 81 cells."""
    cells = []
    height, width = grid_image.shape[:2]
    cell_h = height // 9
    cell_w = width // 9

    for i in range(9):
        for j in range(9):
            y1 = i * cell_h
            y2 = (i + 1) * cell_h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w

            cell = grid_image[y1:y2, x1:x2]
            cells.append(cell)

    return cells


def preprocess_cell(cell: np.ndarray) -> np.ndarray:
    """Preprocess a cell for OCR."""
    # Convert to grayscale if needed
    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # Resize for better OCR
    cell = cv2.resize(cell, (50, 50))

    # Crop center to remove grid lines
    margin = 5
    cell = cell[margin:-margin, margin:-margin]

    # Threshold
    _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Add border
    cell = cv2.copyMakeBorder(cell, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

    return cell


def recognize_digit(cell: np.ndarray) -> int:
    """Recognize a digit in a cell using OCR."""
    processed = preprocess_cell(cell)

    # Check if cell is mostly empty
    white_pixels = np.sum(processed == 255)
    total_pixels = processed.size
    if white_pixels / total_pixels < 0.03:
        return 0

    # OCR configuration for single digit
    config = '--psm 10 --oem 3 -c tessedit_char_whitelist=123456789'

    try:
        text = pytesseract.image_to_string(processed, config=config).strip()
        if text and text.isdigit() and 1 <= int(text) <= 9:
            return int(text)
    except Exception:
        pass

    return 0


def extract_grid_from_image(image: np.ndarray) -> Optional[List[List[int]]]:
    """Extract Sudoku grid from image."""
    thresh = preprocess_image(image)
    contour = find_grid_contour(thresh)

    if contour is None:
        return None

    warped = perspective_transform(image, contour)
    cells = extract_cells(warped)

    grid = []
    for i in range(9):
        row = []
        for j in range(9):
            digit = recognize_digit(cells[i * 9 + j])
            row.append(digit)
        grid.append(row)

    return grid


# ============== Simulated Annealing Solver ==============

def calculate_energy(grid: np.ndarray, fixed: np.ndarray) -> int:
    """Calculate the number of conflicts (energy) in the grid."""
    conflicts = 0

    # Row conflicts
    for i in range(9):
        row = grid[i, :]
        conflicts += 9 - len(set(row))

    # Column conflicts
    for j in range(9):
        col = grid[:, j]
        conflicts += 9 - len(set(col))

    # 3x3 box conflicts (already handled by initialization and swaps within boxes)
    # Boxes should have no duplicates by construction

    return conflicts


def initialize_grid(puzzle: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Initialize grid by filling empty cells with missing numbers in each 3x3 box."""
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


def simulated_annealing(puzzle: List[List[int]],
                        initial_temp: float = 1.0,
                        cooling_rate: float = 0.99999,
                        max_iterations: int = 500000) -> tuple:
    """Solve Sudoku using simulated annealing."""
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


# ============== API Endpoints ==============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/extract", response_model=ExtractResponse)
async def extract_grid(file: UploadFile = File(...)):
    """Extract Sudoku grid from uploaded image."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        grid = extract_grid_from_image(image)

        if grid is None:
            return ExtractResponse(
                success=False,
                message="Could not detect Sudoku grid in image. Please ensure the grid is clearly visible."
            )

        return ExtractResponse(success=True, grid=grid, message="Grid extracted successfully")

    except Exception as e:
        return ExtractResponse(success=False, message=f"Error processing image: {str(e)}")


@app.post("/api/solve", response_model=SolveResponse)
async def solve_sudoku(data: SudokuGrid):
    """Solve a Sudoku puzzle using simulated annealing."""
    grid = data.grid

    # Validate grid
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        return SolveResponse(success=False, message="Invalid grid size. Must be 9x9.")

    for row in grid:
        for val in row:
            if not isinstance(val, int) or val < 0 or val > 9:
                return SolveResponse(success=False, message="Invalid cell value. Must be 0-9.")

    # Solve
    solution, iterations, success = simulated_annealing(grid)

    if success:
        return SolveResponse(
            success=True,
            solution=solution,
            iterations=iterations,
            message=f"Solved in {iterations} iterations"
        )
    else:
        return SolveResponse(
            success=False,
            solution=solution,
            iterations=iterations,
            message="Could not find perfect solution. Showing best attempt."
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
