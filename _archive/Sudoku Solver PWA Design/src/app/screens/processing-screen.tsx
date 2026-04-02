import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

interface ProcessingScreenProps {
  imageUrl: string;
}

export function ProcessingScreen({ imageUrl }: ProcessingScreenProps) {
  const [gridLines, setGridLines] = useState<number>(0);

  useEffect(() => {
    // Animate grid lines drawing
    const interval = setInterval(() => {
      setGridLines((prev) => {
        if (prev >= 18) {
          clearInterval(interval);
          return prev;
        }
        return prev + 1;
      });
    }, 80);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 flex items-center justify-center p-4">
      <div className="max-w-lg w-full space-y-6">
        {/* Image with grid overlay */}
        <div className="relative rounded-2xl overflow-hidden shadow-2xl">
          <img
            src={imageUrl}
            alt="Captured puzzle"
            className="w-full h-auto opacity-80"
          />

          {/* Animated grid lines */}
          <svg
            className="absolute inset-0 w-full h-full"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
          >
            {/* Horizontal lines */}
            {Array.from({ length: 10 }).map((_, i) => (
              <motion.line
                key={`h-${i}`}
                x1="0"
                y1={i * 11.11}
                x2="100"
                y2={i * 11.11}
                stroke="cyan"
                strokeWidth={i % 3 === 0 ? "0.5" : "0.2"}
                initial={{ pathLength: 0 }}
                animate={{
                  pathLength: gridLines >= i ? 1 : 0,
                }}
                transition={{ duration: 0.3 }}
              />
            ))}

            {/* Vertical lines */}
            {Array.from({ length: 10 }).map((_, i) => (
              <motion.line
                key={`v-${i}`}
                x1={i * 11.11}
                y1="0"
                x2={i * 11.11}
                y2="100"
                stroke="cyan"
                strokeWidth={i % 3 === 0 ? "0.5" : "0.2"}
                initial={{ pathLength: 0 }}
                animate={{
                  pathLength: gridLines >= i + 9 ? 1 : 0,
                }}
                transition={{ duration: 0.3 }}
              />
            ))}
          </svg>
        </div>

        {/* Status */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-3"
        >
          <div className="flex items-center justify-center gap-3">
            <Loader2 className="size-6 text-cyan-400 animate-spin" />
            <span className="text-white text-lg font-medium">
              Detecting grid...
            </span>
          </div>
          <p className="text-white/60 text-sm">
            Analyzing puzzle structure and extracting numbers
          </p>
        </motion.div>
      </div>
    </div>
  );
}
