import { motion } from "motion/react";
import { cn } from "./ui/utils";
import { SudokuGrid } from "../types";

interface EnhancedSudokuGridProps {
  grid: SudokuGrid;
  originalGrid?: SudokuGrid;
  confidenceMap?: number[][]; // 0-1 for OCR confidence
  errorCells?: Set<string>; // "row-col"
  selectedCell?: { row: number; col: number } | null;
  onCellClick?: (row: number, col: number) => void;
  animateSolution?: boolean;
  editable?: boolean;
}

export function EnhancedSudokuGrid({
  grid,
  originalGrid,
  confidenceMap,
  errorCells,
  selectedCell,
  onCellClick,
  animateSolution = false,
  editable = false,
}: EnhancedSudokuGridProps) {
  const isOriginalCell = (row: number, col: number) => {
    return originalGrid && originalGrid[row][col] !== 0;
  };

  const isSolvedCell = (row: number, col: number) => {
    return (
      originalGrid &&
      originalGrid[row][col] === 0 &&
      grid[row][col] !== 0
    );
  };

  const hasError = (row: number, col: number) => {
    return errorCells?.has(`${row}-${col}`) || false;
  };

  const hasLowConfidence = (row: number, col: number) => {
    return confidenceMap && confidenceMap[row][col] < 0.7;
  };

  const isSelected = (row: number, col: number) => {
    return selectedCell?.row === row && selectedCell?.col === col;
  };

  const getCellClasses = (row: number, col: number) => {
    const classes = [];

    if (isOriginalCell(row, col)) {
      classes.push("text-cyan-400 font-bold");
    } else if (isSolvedCell(row, col)) {
      classes.push("text-green-400 font-semibold");
    } else {
      classes.push("text-gray-600");
    }

    if (hasError(row, col)) {
      classes.push("bg-red-500/20 border-red-500");
    } else if (isSelected(row, col)) {
      classes.push("bg-cyan-500/20 border-cyan-400");
    } else {
      classes.push("border-gray-700/30");
    }

    if (editable && !isOriginalCell(row, col)) {
      classes.push("cursor-pointer hover:bg-white/5");
    }

    return cn(classes);
  };

  const getBoxClasses = (row: number, col: number) => {
    const classes = ["border"];

    // Thick borders for 3x3 boxes
    if (row % 3 === 0) classes.push("border-t-2 border-t-gray-600");
    if (col % 3 === 0) classes.push("border-l-2 border-l-gray-600");
    if (row === 8) classes.push("border-b-2 border-b-gray-600");
    if (col === 8) classes.push("border-r-2 border-r-gray-600");

    return cn(classes);
  };

  const getAnimationDelay = (row: number, col: number) => {
    if (!animateSolution) return 0;
    const boxIndex = Math.floor(row / 3) * 3 + Math.floor(col / 3);
    return boxIndex * 0.08 + (row % 3) * 0.02 + (col % 3) * 0.02;
  };

  return (
    <div className="inline-block bg-gray-900/50 backdrop-blur-sm rounded-2xl shadow-2xl p-4 border border-gray-700/50">
      <div className="grid grid-cols-9 gap-0 aspect-square w-full max-w-[min(100vw-3rem,450px)]">
        {grid.map((row, rowIndex) =>
          row.map((cell, colIndex) => (
            <div
              key={`${rowIndex}-${colIndex}`}
              className={getBoxClasses(rowIndex, colIndex)}
            >
              <motion.div
                initial={animateSolution && isSolvedCell(rowIndex, colIndex) ? { opacity: 0, scale: 0.5 } : false}
                animate={{ opacity: 1, scale: 1 }}
                transition={{
                  duration: 0.3,
                  delay: getAnimationDelay(rowIndex, colIndex),
                }}
                onClick={() => onCellClick?.(rowIndex, colIndex)}
                className={cn(
                  "w-full h-full flex items-center justify-center text-lg relative",
                  getCellClasses(rowIndex, colIndex)
                )}
              >
                {cell !== 0 && cell}
                {hasLowConfidence(rowIndex, colIndex) && (
                  <div className="absolute top-0.5 right-0.5 w-1.5 h-1.5 bg-orange-500 rounded-full" />
                )}
              </motion.div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
