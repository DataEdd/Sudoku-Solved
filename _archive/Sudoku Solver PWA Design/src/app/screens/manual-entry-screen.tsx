import { useState } from "react";
import { ArrowLeft, Eraser } from "lucide-react";
import { GlassButton } from "../components/glass-button";
import { SudokuGrid } from "../types";
import { motion } from "motion/react";

interface ManualEntryScreenProps {
  onBack: () => void;
  onSolve: (grid: SudokuGrid) => void;
}

export function ManualEntryScreen({ onBack, onSolve }: ManualEntryScreenProps) {
  const [grid, setGrid] = useState<SudokuGrid>(
    Array(9)
      .fill(null)
      .map(() => Array(9).fill(0))
  );
  const [selectedCell, setSelectedCell] = useState<{
    row: number;
    col: number;
  } | null>(null);

  const handleNumberClick = (num: number) => {
    if (!selectedCell) return;

    const newGrid = grid.map((row, i) =>
      i === selectedCell.row
        ? row.map((cell, j) => (j === selectedCell.col ? num : cell))
        : [...row]
    );

    setGrid(newGrid);

    // Auto-advance to next cell
    if (selectedCell.col < 8) {
      setSelectedCell({ row: selectedCell.row, col: selectedCell.col + 1 });
    } else if (selectedCell.row < 8) {
      setSelectedCell({ row: selectedCell.row + 1, col: 0 });
    }
  };

  const handleCellClick = (row: number, col: number) => {
    setSelectedCell({ row, col });
  };

  const handleClear = () => {
    setGrid(
      Array(9)
        .fill(null)
        .map(() => Array(9).fill(0))
    );
    setSelectedCell(null);
  };

  const handleSolve = () => {
    const filledCells = grid.flat().filter((c) => c !== 0).length;
    if (filledCells < 17) {
      alert("Please enter at least 17 numbers for a valid puzzle");
      return;
    }
    onSolve(grid);
  };

  const filledCount = grid.flat().filter((c) => c !== 0).length;

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 overflow-auto">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-gray-900/80 backdrop-blur-md border-b border-gray-700/50 p-4">
        <div className="max-w-2xl mx-auto flex items-center gap-4">
          <GlassButton variant="ghost" size="icon" onClick={onBack}>
            <ArrowLeft className="size-5" />
          </GlassButton>
          <div className="flex-1">
            <h1 className="text-white font-bold text-lg">Manual Entry</h1>
            <p className="text-white/60 text-sm">
              {filledCount}/81 cells filled
            </p>
          </div>
          <GlassButton variant="secondary" size="sm" onClick={handleClear}>
            <Eraser className="size-4 mr-2" />
            Clear
          </GlassButton>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-2xl mx-auto p-4 space-y-6 pb-32">
        {/* Grid */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex justify-center"
        >
          <div className="inline-block bg-gray-900/50 backdrop-blur-sm rounded-2xl shadow-2xl p-4 border border-gray-700/50">
            <div className="grid grid-cols-9 gap-0 aspect-square w-full max-w-[min(100vw-3rem,450px)]">
              {grid.map((row, rowIndex) =>
                row.map((cell, colIndex) => {
                  const isSelected =
                    selectedCell?.row === rowIndex &&
                    selectedCell?.col === colIndex;
                  const hasBorderTop = rowIndex % 3 === 0;
                  const hasBorderLeft = colIndex % 3 === 0;
                  const hasBorderBottom = rowIndex === 8;
                  const hasBorderRight = colIndex === 8;

                  return (
                    <div
                      key={`${rowIndex}-${colIndex}`}
                      className={`border border-gray-700/30 ${
                        hasBorderTop ? "border-t-2 border-t-gray-600" : ""
                      } ${hasBorderLeft ? "border-l-2 border-l-gray-600" : ""} ${
                        hasBorderBottom ? "border-b-2 border-b-gray-600" : ""
                      } ${hasBorderRight ? "border-r-2 border-r-gray-600" : ""}`}
                    >
                      <button
                        onClick={() => handleCellClick(rowIndex, colIndex)}
                        className={`w-full h-full flex items-center justify-center text-lg font-semibold transition-all ${
                          isSelected
                            ? "bg-cyan-500/30 text-white border-2 border-cyan-400"
                            : cell !== 0
                            ? "bg-gray-800/50 text-cyan-400"
                            : "bg-transparent text-gray-600 hover:bg-white/5"
                        }`}
                      >
                        {cell !== 0 && cell}
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </motion.div>

        {/* Number Pad */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-4 border border-gray-700/50"
        >
          <div className="grid grid-cols-3 gap-3">
            {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((num) => (
              <motion.button
                key={num}
                whileTap={{ scale: 0.95 }}
                onClick={() => handleNumberClick(num)}
                disabled={!selectedCell}
                className="aspect-square bg-gradient-to-br from-cyan-500 to-blue-600 rounded-2xl text-white text-3xl font-bold shadow-lg hover:shadow-cyan-500/50 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
              >
                {num}
              </motion.button>
            ))}
          </div>

          <button
            onClick={() => handleNumberClick(0)}
            disabled={!selectedCell}
            className="w-full mt-3 py-4 bg-gray-700/50 rounded-2xl text-white font-medium hover:bg-gray-700 transition-colors disabled:opacity-30"
          >
            Delete
          </button>
        </motion.div>

        {/* Hint */}
        <div className="bg-blue-500/10 border border-blue-500/30 rounded-2xl p-4">
          <p className="text-blue-200 text-sm">
            💡 Tap a cell, then select a number from the keypad below. You need
            at least 17 numbers for a valid puzzle.
          </p>
        </div>
      </div>

      {/* Bottom Action */}
      <div className="fixed bottom-0 left-0 right-0 bg-gray-900/80 backdrop-blur-md border-t border-gray-700/50 p-4 pb-8">
        <div className="max-w-2xl mx-auto">
          <GlassButton
            variant="primary"
            onClick={handleSolve}
            disabled={filledCount < 17}
            className="w-full"
          >
            Solve Puzzle ({filledCount}/17 minimum)
          </GlassButton>
        </div>
      </div>
    </div>
  );
}
