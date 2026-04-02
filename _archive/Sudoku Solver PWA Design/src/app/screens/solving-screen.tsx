import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { EnhancedSudokuGrid } from "../components/enhanced-sudoku-grid";
import { SudokuGrid } from "../types";

interface SolvingScreenProps {
  grid: SudokuGrid;
}

export function SolvingScreen({ grid }: SolvingScreenProps) {
  const [iterations, setIterations] = useState(0);

  useEffect(() => {
    // Simulate iteration counter
    const interval = setInterval(() => {
      setIterations((prev) => Math.min(prev + Math.floor(Math.random() * 50) + 20, 500));
    }, 100);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 flex items-center justify-center p-4 overflow-hidden">
      {/* Animated background gradient */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-purple-500/20"
        animate={{
          backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "linear",
        }}
        style={{ backgroundSize: "200% 200%" }}
      />

      <div className="relative z-10 max-w-lg w-full space-y-8">
        {/* Grid with sweeping gradient overlay */}
        <div className="relative">
          <EnhancedSudokuGrid grid={grid} originalGrid={grid} />

          {/* Sweeping gradient overlay */}
          <motion.div
            className="absolute inset-0 pointer-events-none"
            style={{
              background:
                "linear-gradient(90deg, transparent 0%, rgba(6, 182, 212, 0.3) 50%, transparent 100%)",
            }}
            animate={{
              x: ["-100%", "200%"],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "linear",
            }}
          />
        </div>

        {/* Status */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-4"
        >
          <div>
            <motion.div
              key={iterations}
              initial={{ scale: 1.2, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="text-4xl font-bold text-cyan-400 mb-2"
            >
              {iterations.toLocaleString()}
            </motion.div>
            <p className="text-white/60 text-sm">iterations</p>
          </div>

          <div className="flex items-center justify-center gap-2">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "linear",
              }}
              className="w-5 h-5 border-2 border-cyan-400 border-t-transparent rounded-full"
            />
            <span className="text-white text-lg font-medium">
              Finding solution...
            </span>
          </div>

          <p className="text-white/40 text-xs">
            Using backtracking algorithm
          </p>
        </motion.div>
      </div>
    </div>
  );
}
