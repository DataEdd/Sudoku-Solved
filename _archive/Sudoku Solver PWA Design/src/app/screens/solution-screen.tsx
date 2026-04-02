import { motion } from "motion/react";
import { CheckCircle2, Camera, Share2, Download } from "lucide-react";
import { GlassButton } from "../components/glass-button";
import { EnhancedSudokuGrid } from "../components/enhanced-sudoku-grid";
import { SudokuGrid } from "../types";

interface SolutionScreenProps {
  originalGrid: SudokuGrid;
  solvedGrid: SudokuGrid;
  solveTime: number;
  onNewPuzzle: () => void;
}

export function SolutionScreen({
  originalGrid,
  solvedGrid,
  solveTime,
  onNewPuzzle,
}: SolutionScreenProps) {
  const difficulty = "Medium"; // Could calculate based on puzzle complexity
  const filledCells = originalGrid.flat().filter((c) => c !== 0).length;

  const handleShare = () => {
    // Would implement sharing functionality
    alert("Share functionality would be implemented here");
  };

  const handleSave = () => {
    // Would save to history
    alert("Saved to history!");
  };

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 overflow-auto">
      {/* Success Banner */}
      <motion.div
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ type: "spring", damping: 20 }}
        className="bg-gradient-to-r from-green-500 to-emerald-600 p-6"
      >
        <div className="max-w-2xl mx-auto flex items-center gap-4">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.3, type: "spring" }}
          >
            <CheckCircle2 className="size-12 text-white" />
          </motion.div>
          <div className="flex-1">
            <h1 className="text-white font-bold text-xl">Puzzle Solved!</h1>
            <p className="text-white/90 text-sm">
              Great work on this {difficulty.toLowerCase()} puzzle
            </p>
          </div>
        </div>
      </motion.div>

      {/* Content */}
      <div className="max-w-2xl mx-auto p-4 space-y-6 pb-32">
        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-3 gap-3"
        >
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-4 text-center border border-gray-700/50">
            <div className="text-2xl font-bold text-cyan-400">
              {(solveTime / 1000).toFixed(1)}s
            </div>
            <div className="text-white/60 text-xs mt-1">Solve Time</div>
          </div>
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-4 text-center border border-gray-700/50">
            <div className="text-2xl font-bold text-cyan-400">{filledCells}</div>
            <div className="text-white/60 text-xs mt-1">Given Clues</div>
          </div>
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-4 text-center border border-gray-700/50">
            <div className="text-2xl font-bold text-cyan-400">{difficulty}</div>
            <div className="text-white/60 text-xs mt-1">Difficulty</div>
          </div>
        </motion.div>

        {/* Grid */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="flex justify-center"
        >
          <EnhancedSudokuGrid
            grid={solvedGrid}
            originalGrid={originalGrid}
            animateSolution
          />
        </motion.div>

        {/* Legend */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-gray-800/50 rounded-2xl p-4 space-y-2"
        >
          <div className="flex items-center gap-2 text-sm">
            <span className="text-cyan-400 font-bold">●</span>
            <span className="text-white/80">Original numbers (given clues)</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-green-400 font-bold">●</span>
            <span className="text-white/80">Solved numbers (AI solution)</span>
          </div>
        </motion.div>
      </div>

      {/* Bottom Actions */}
      <div className="fixed bottom-0 left-0 right-0 bg-gray-900/80 backdrop-blur-md border-t border-gray-700/50 p-4 pb-8">
        <div className="max-w-2xl mx-auto space-y-3">
          <GlassButton
            variant="primary"
            onClick={onNewPuzzle}
            className="w-full flex items-center justify-center gap-2"
          >
            <Camera className="size-5" />
            Solve Another Puzzle
          </GlassButton>

          <div className="grid grid-cols-2 gap-3">
            <GlassButton
              variant="secondary"
              onClick={handleShare}
              className="flex items-center justify-center gap-2"
            >
              <Share2 className="size-4" />
              Share
            </GlassButton>
            <GlassButton
              variant="secondary"
              onClick={handleSave}
              className="flex items-center justify-center gap-2"
            >
              <Download className="size-4" />
              Save
            </GlassButton>
          </div>
        </div>
      </div>
    </div>
  );
}
