export type SudokuGrid = number[][];

export function isValid(
  grid: SudokuGrid,
  row: number,
  col: number,
  num: number
): boolean {
  // Check row
  for (let x = 0; x < 9; x++) {
    if (grid[row][x] === num) return false;
  }

  // Check column
  for (let x = 0; x < 9; x++) {
    if (grid[x][col] === num) return false;
  }

  // Check 3x3 box
  const startRow = row - (row % 3);
  const startCol = col - (col % 3);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      if (grid[i + startRow][j + startCol] === num) return false;
    }
  }

  return true;
}

export function solveSudoku(grid: SudokuGrid): boolean {
  for (let row = 0; row < 9; row++) {
    for (let col = 0; col < 9; col++) {
      if (grid[row][col] === 0) {
        for (let num = 1; num <= 9; num++) {
          if (isValid(grid, row, col, num)) {
            grid[row][col] = num;

            if (solveSudoku(grid)) {
              return true;
            }

            grid[row][col] = 0;
          }
        }
        return false;
      }
    }
  }
  return true;
}

export function isValidSudoku(grid: SudokuGrid): boolean {
  // Check if the puzzle has at least 17 clues (minimum for unique solution)
  const clues = grid.flat().filter((cell) => cell !== 0).length;
  if (clues < 17) return false;

  // Check each row, column, and box for duplicates
  for (let i = 0; i < 9; i++) {
    const rowSet = new Set<number>();
    const colSet = new Set<number>();

    for (let j = 0; j < 9; j++) {
      // Check row
      if (grid[i][j] !== 0) {
        if (rowSet.has(grid[i][j])) return false;
        rowSet.add(grid[i][j]);
      }

      // Check column
      if (grid[j][i] !== 0) {
        if (colSet.has(grid[j][i])) return false;
        colSet.add(grid[j][i]);
      }
    }
  }

  // Check 3x3 boxes
  for (let boxRow = 0; boxRow < 3; boxRow++) {
    for (let boxCol = 0; boxCol < 3; boxCol++) {
      const boxSet = new Set<number>();
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          const row = boxRow * 3 + i;
          const col = boxCol * 3 + j;
          if (grid[row][col] !== 0) {
            if (boxSet.has(grid[row][col])) return false;
            boxSet.add(grid[row][col]);
          }
        }
      }
    }
  }

  return true;
}

export function deepCopyGrid(grid: SudokuGrid): SudokuGrid {
  return grid.map((row) => [...row]);
}

// Simulate OCR extraction from image
export function extractPuzzleFromImage(): SudokuGrid {
  // This simulates OCR extraction with a sample puzzle
  // In a real implementation, this would use OCR libraries
  return [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
  ];
}
