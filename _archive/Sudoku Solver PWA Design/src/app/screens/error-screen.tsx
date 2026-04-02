import { motion } from "motion/react";
import { AlertCircle, Camera, XCircle } from "lucide-react";
import { GlassButton } from "../components/glass-button";

interface ErrorScreenProps {
  type: "no-grid" | "invalid-puzzle" | "no-solution";
  onRetry: () => void;
  onBack: () => void;
}

export function ErrorScreen({ type, onRetry, onBack }: ErrorScreenProps) {
  const errorContent = {
    "no-grid": {
      icon: Camera,
      title: "No Grid Detected",
      description:
        "We couldn't find a Sudoku puzzle in the image. Try taking a clearer photo with better lighting.",
      tips: [
        "Ensure the puzzle is well-lit",
        "Hold your device steady",
        "Frame the entire puzzle within the guides",
      ],
    },
    "invalid-puzzle": {
      icon: AlertCircle,
      title: "Invalid Puzzle",
      description:
        "This puzzle has duplicate numbers in rows, columns, or boxes. Please review and fix the conflicts.",
      tips: [
        "Check for duplicate numbers in red",
        "Each row must have unique numbers 1-9",
        "Each column must have unique numbers 1-9",
        "Each 3×3 box must have unique numbers 1-9",
      ],
    },
    "no-solution": {
      icon: XCircle,
      title: "No Solution Found",
      description:
        "This puzzle has no valid solution. It may contain errors from extraction or be unsolvable.",
      tips: [
        "Review the extracted numbers",
        "Check for OCR errors",
        "Verify the original puzzle is valid",
      ],
    },
  };

  const content = errorContent[type];
  const Icon = content.icon;

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="max-w-md w-full space-y-6"
      >
        {/* Error Icon */}
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: "spring" }}
          className="w-20 h-20 mx-auto bg-red-500/20 rounded-full flex items-center justify-center border-4 border-red-500/30"
        >
          <Icon className="size-10 text-red-400" />
        </motion.div>

        {/* Error Message */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="text-center space-y-2"
        >
          <h2 className="text-2xl font-bold text-white">{content.title}</h2>
          <p className="text-white/70">{content.description}</p>
        </motion.div>

        {/* Tips */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-4 border border-gray-700/50"
        >
          <p className="text-white/60 text-xs font-medium mb-3">💡 TIPS</p>
          <ul className="space-y-2">
            {content.tips.map((tip, index) => (
              <li key={index} className="flex items-start gap-2 text-sm">
                <span className="text-cyan-400 mt-0.5">•</span>
                <span className="text-white/80">{tip}</span>
              </li>
            ))}
          </ul>
        </motion.div>

        {/* Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="space-y-3"
        >
          <GlassButton variant="primary" onClick={onRetry} className="w-full">
            Try Again
          </GlassButton>
          <GlassButton variant="secondary" onClick={onBack} className="w-full">
            Go Back
          </GlassButton>
        </motion.div>
      </motion.div>
    </div>
  );
}
