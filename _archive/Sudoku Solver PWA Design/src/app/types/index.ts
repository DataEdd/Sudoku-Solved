export type SudokuGrid = number[][];

export interface CellInfo {
  value: number;
  isOriginal: boolean;
  isSolved: boolean;
  hasError: boolean;
  confidence?: number; // 0-1 for OCR confidence
}

export interface PuzzleSolve {
  id: string;
  timestamp: number;
  puzzle: SudokuGrid;
  solution: SudokuGrid;
  source: "camera" | "upload" | "manual";
  difficulty?: "easy" | "medium" | "hard" | "expert";
  solveTime: number; // milliseconds
  isFavorite: boolean;
}

export type Screen =
  | "camera"
  | "processing"
  | "review"
  | "solving"
  | "solution"
  | "history"
  | "manual";
