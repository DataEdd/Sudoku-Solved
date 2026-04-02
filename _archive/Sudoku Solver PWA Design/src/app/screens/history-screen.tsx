import { useState } from "react";
import { ArrowLeft, Camera, Upload, Edit, Star, Trash2 } from "lucide-react";
import { GlassButton } from "../components/glass-button";
import { PuzzleSolve } from "../types";
import { motion } from "motion/react";

interface HistoryScreenProps {
  onBack: () => void;
  history: PuzzleSolve[];
  onDelete: (id: string) => void;
  onToggleFavorite: (id: string) => void;
}

type FilterTab = "all" | "camera" | "uploaded" | "favorites";

export function HistoryScreen({
  onBack,
  history,
  onDelete,
  onToggleFavorite,
}: HistoryScreenProps) {
  const [activeTab, setActiveTab] = useState<FilterTab>("all");

  const filteredHistory = history.filter((item) => {
    if (activeTab === "all") return true;
    if (activeTab === "favorites") return item.isFavorite;
    if (activeTab === "camera") return item.source === "camera";
    if (activeTab === "uploaded") return item.source === "upload";
    return true;
  });

  const getDifficultyColor = (difficulty?: string) => {
    switch (difficulty) {
      case "easy":
        return "bg-green-500/20 text-green-400 border-green-500/30";
      case "medium":
        return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      case "hard":
        return "bg-orange-500/20 text-orange-400 border-orange-500/30";
      case "expert":
        return "bg-red-500/20 text-red-400 border-red-500/30";
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30";
    }
  };

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 overflow-auto">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-gray-900/80 backdrop-blur-md border-b border-gray-700/50 p-4">
        <div className="max-w-2xl mx-auto flex items-center gap-4">
          <GlassButton variant="ghost" size="icon" onClick={onBack}>
            <ArrowLeft className="size-5" />
          </GlassButton>
          <div>
            <h1 className="text-white font-bold text-lg">History</h1>
            <p className="text-white/60 text-sm">
              {filteredHistory.length} solved puzzle{filteredHistory.length !== 1 ? "s" : ""}
            </p>
          </div>
        </div>

        {/* Filter Tabs */}
        <div className="max-w-2xl mx-auto mt-4 flex gap-2 overflow-x-auto pb-2">
          {[
            { id: "all" as FilterTab, label: "All" },
            { id: "camera" as FilterTab, label: "Camera" },
            { id: "uploaded" as FilterTab, label: "Uploaded" },
            { id: "favorites" as FilterTab, label: "Favorites" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all whitespace-nowrap ${
                activeTab === tab.id
                  ? "bg-cyan-500 text-white"
                  : "bg-gray-800/50 text-white/60 hover:bg-gray-800"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="max-w-2xl mx-auto p-4 pb-20">
        {filteredHistory.length === 0 ? (
          <div className="text-center py-16">
            <div className="w-16 h-16 bg-gray-800/50 rounded-full flex items-center justify-center mx-auto mb-4">
              <Camera className="size-8 text-gray-600" />
            </div>
            <p className="text-white/60 text-sm">No puzzles found</p>
            <p className="text-white/40 text-xs mt-1">
              Solve your first puzzle to see it here
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3">
            {filteredHistory.map((item, index) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className="bg-gray-800/50 backdrop-blur-sm rounded-2xl border border-gray-700/50 overflow-hidden"
              >
                {/* Puzzle Preview */}
                <div className="aspect-square bg-gray-900/50 p-3 relative">
                  <div className="grid grid-cols-9 gap-[1px] h-full bg-gray-700/30">
                    {item.solution.map((row, i) =>
                      row.map((cell, j) => (
                        <div
                          key={`${i}-${j}`}
                          className={`bg-gray-800 flex items-center justify-center ${
                            i % 3 === 0 ? "border-t border-t-gray-600" : ""
                          } ${j % 3 === 0 ? "border-l border-l-gray-600" : ""}`}
                        >
                          <span
                            className={`text-[6px] ${
                              item.puzzle[i][j] !== 0
                                ? "text-cyan-400 font-bold"
                                : "text-green-400"
                            }`}
                          >
                            {cell || ""}
                          </span>
                        </div>
                      ))
                    )}
                  </div>

                  {/* Source Icon */}
                  <div className="absolute top-2 left-2 w-6 h-6 bg-gray-900/80 backdrop-blur-sm rounded-full flex items-center justify-center">
                    {item.source === "camera" ? (
                      <Camera className="size-3 text-white/60" />
                    ) : item.source === "upload" ? (
                      <Upload className="size-3 text-white/60" />
                    ) : (
                      <Edit className="size-3 text-white/60" />
                    )}
                  </div>

                  {/* Favorite */}
                  <button
                    onClick={() => onToggleFavorite(item.id)}
                    className="absolute top-2 right-2 w-6 h-6 bg-gray-900/80 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-gray-900 transition-colors"
                  >
                    <Star
                      className={`size-3 ${
                        item.isFavorite
                          ? "fill-yellow-400 text-yellow-400"
                          : "text-white/60"
                      }`}
                    />
                  </button>
                </div>

                {/* Info */}
                <div className="p-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-white/80 text-xs">
                      {formatDate(item.timestamp)}
                    </span>
                    {item.difficulty && (
                      <span
                        className={`text-[10px] px-2 py-0.5 rounded-full border ${getDifficultyColor(
                          item.difficulty
                        )}`}
                      >
                        {item.difficulty}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-white/60">
                      {(item.solveTime / 1000).toFixed(1)}s
                    </span>
                    <button
                      onClick={() => onDelete(item.id)}
                      className="text-red-400/60 hover:text-red-400 transition-colors"
                    >
                      <Trash2 className="size-3" />
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
