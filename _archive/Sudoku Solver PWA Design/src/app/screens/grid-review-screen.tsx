import { useState } from "react";
import { ArrowLeft, AlertCircle } from "lucide-react";
import { GlassButton } from "../components/glass-button";
import { EnhancedSudokuGrid } from "../components/enhanced-sudoku-grid";
import { NumberPicker } from "../components/number-picker";
import { SudokuGrid } from "../types";
import { motion } from "motion/react";

interface GridReviewScreenProps {
  grid: SudokuGrid;
  confidenceMap: number[][];
  onRetake: () => void;
  onSolve: (grid: SudokuGrid) => void;
}

export function GridReviewScreen({
  grid: initialGrid,
  confidenceMap,
  onRetake,
  onSolve,
}: GridReviewScreenProps) {
  const [grid, setGrid] = useState<SudokuGrid>(initialGrid);
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | null>(null);
  const [showPicker, setShowPicker] = useState(false);
  const [errorCells, setErrorCells] = useState<Set<string>>(new Set());

  const validateGrid = () => {
    const errors = new Set<string>();

    // Check rows
    for (let row = 0; row < 9; row++) {
      const seen = new Set<number>();
      for (let col = 0; col < 9; col++) {
        const val = grid[row][col];
        if (val !== 0) {
          if (seen.has(val)) {
            errors.add(`${row}-${col}`);
          }
          seen.add(val);
        }
      }
    }

    // Check columns
    for (let col = 0; col < 9; col++) {
      const seen = new Set<number>();
      for (let row = 0; row < 9; row++) {
        const val = grid[row][col];
        if (val !== 0) {
          if (seen.has(val)) {
            errors.add(`${row}-${col}`);
          }
          seen.add(val);
        }
      }
    }

    // Check 3x3 boxes
    for (let boxRow = 0; boxRow < 3; boxRow++) {
      for (let boxCol = 0; boxCol < 3; boxCol++) {
        const seen = new Set<number>();
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) {
            const row = boxRow * 3 + i;
            const col = boxCol * 3 + j;
            const val = grid[row][col];
            if (val !== 0) {
              if (seen.has(val)) {
                errors.add(`${row}-${col}`);
              }
              seen.add(val);
            }
          }
        }
      }
    }

    setErrorCells(errors);
    return errors.size === 0;
  };

  const handleCellClick = (row: number, col: number) => {
    setSelectedCell({ row, col });
    setShowPicker(true);
  };

  const handleNumberSelect = (num: number) => {
    if (!selectedCell) return;

    const newGrid = grid.map((row, i) =>
      i === selectedCell.row
        ? row.map((cell, j) => (j === selectedCell.col ? num : cell))
        : [...row]
    );

    setGrid(newGrid);
    setSelectedCell(null);
  };

  const handleSolve = () => {
    if (validateGrid()) {
      onSolve(grid);
    }
  };

  const lowConfidenceCount = confidenceMap
    .flat()
    .filter((conf) => conf < 0.7).length;

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 overflow-auto">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-gray-900/80 backdrop-blur-md border-b border-gray-700/50 p-4">
        <div className="max-w-2xl mx-auto flex items-center gap-4">
          <GlassButton variant="ghost" size="icon" onClick={onRetake}>
            <ArrowLeft className="size-5" />
          </GlassButton>
          <div>
            <h1 className="text-white font-bold text-lg">Review Grid</h1>
            <p className="text-white/60 text-sm">
              Tap cells to edit extracted numbers
            </p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-2xl mx-auto p-4 space-y-6 pb-32">
        {/* Warnings */}
        {lowConfidenceCount > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-orange-500/10 border border-orange-500/30 rounded-2xl p-4 flex items-start gap-3"
          >
            <AlertCircle className="size-5 text-orange-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-orange-200 text-sm font-medium">
                Low confidence detected
              </p>
              <p className="text-orange-300/70 text-xs mt-1">
                {lowConfidenceCount} cells marked with orange dots need review
              </p>
            </div>
          </motion.div>
        )}

        {errorCells.size > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-red-500/10 border border-red-500/30 rounded-2xl p-4 flex items-start gap-3"
          >
            <AlertCircle className="size-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-red-200 text-sm font-medium">
                Invalid puzzle detected
              </p>
              <p className="text-red-300/70 text-xs mt-1">
                Duplicate numbers found in rows, columns, or boxes
              </p>
            </div>
          </motion.div>
        )}

        {/* Grid */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex justify-center"
        >
          <EnhancedSudokuGrid
            grid={grid}
            originalGrid={initialGrid}
            confidenceMap={confidenceMap}
            errorCells={errorCells}
            selectedCell={selectedCell}
            onCellClick={handleCellClick}
            editable
          />
        </motion.div>

        {/* Legend */}
        <div className="bg-gray-800/50 rounded-2xl p-4 space-y-2">
          <p className="text-white/60 text-xs font-medium mb-3">LEGEND</p>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-cyan-400 font-bold">●</span>
            <span className="text-white/80">Detected numbers</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <div className="w-3 h-3 bg-orange-500 rounded-full" />
            <span className="text-white/80">Low confidence - verify</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <div className="w-3 h-3 bg-red-500 rounded-full" />
            <span className="text-white/80">Error - duplicates found</span>
          </div>
        </div>
      </div>

      {/* Bottom Actions */}
      <div className="fixed bottom-0 left-0 right-0 bg-gray-900/80 backdrop-blur-md border-t border-gray-700/50 p-4 pb-8">
        <div className="max-w-2xl mx-auto grid grid-cols-2 gap-3">
          <GlassButton variant="secondary" onClick={onRetake}>
            Retake Photo
          </GlassButton>
          <GlassButton
            variant="primary"
            onClick={handleSolve}
            disabled={errorCells.size > 0}
          >
            Solve Puzzle
          </GlassButton>
        </div>
      </div>

      {/* Number Picker */}
      <NumberPicker
        isOpen={showPicker}
        onClose={() => setShowPicker(false)}
        onSelect={handleNumberSelect}
        selectedCell={selectedCell}
      />
    </div>
  );
}
