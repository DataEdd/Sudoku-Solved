import { useState, useEffect } from "react";
import { Screen, SudokuGrid, PuzzleSolve } from "./types";
import { CameraScreen } from "./screens/camera-screen";
import { ProcessingScreen } from "./screens/processing-screen";
import { GridReviewScreen } from "./screens/grid-review-screen";
import { SolvingScreen } from "./screens/solving-screen";
import { SolutionScreen } from "./screens/solution-screen";
import { HistoryScreen } from "./screens/history-screen";
import { ManualEntryScreen } from "./screens/manual-entry-screen";
import { SplashScreen } from "./screens/splash-screen";
import { PWAInstallPrompt } from "./components/pwa-install-prompt";
import {
  extractPuzzleFromImage,
  solveSudoku,
  deepCopyGrid,
} from "./utils/sudoku-solver";
import {
  saveToHistory,
  getHistory,
  deleteFromHistory,
  toggleFavorite as toggleFavoriteInStorage,
} from "./utils/storage";
import { toast } from "sonner";
import { Toaster } from "./components/ui/sonner";

export default function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [screen, setScreen] = useState<Screen>("camera");
  const [flashEnabled, setFlashEnabled] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string>("");
  const [currentGrid, setCurrentGrid] = useState<SudokuGrid>([]);
  const [originalGrid, setOriginalGrid] = useState<SudokuGrid>([]);
  const [solvedGrid, setSolvedGrid] = useState<SudokuGrid>([]);
  const [confidenceMap, setConfidenceMap] = useState<number[][]>([]);
  const [history, setHistory] = useState<PuzzleSolve[]>([]);
  const [solveStartTime, setSolveStartTime] = useState<number>(0);

  useEffect(() => {
    // Load history on mount
    setHistory(getHistory());
  }, []);

  const generateConfidenceMap = (): number[][] => {
    // Simulate OCR confidence scores
    return Array(9)
      .fill(null)
      .map(() =>
        Array(9)
          .fill(null)
          .map(() => Math.random() * 0.5 + 0.5)
      );
  };

  const handleCapture = (imageUrl: string) => {
    setCapturedImage(imageUrl);
    setScreen("processing");

    // Simulate processing delay
    setTimeout(() => {
      const extracted = extractPuzzleFromImage();
      setCurrentGrid(extracted);
      setOriginalGrid(deepCopyGrid(extracted));
      setConfidenceMap(generateConfidenceMap());
      setScreen("review");
    }, 2500);
  };

  const handleRetake = () => {
    setScreen("camera");
    setCapturedImage("");
  };

  const handleSolve = (grid: SudokuGrid) => {
    setCurrentGrid(grid);
    setOriginalGrid(deepCopyGrid(grid));
    setSolveStartTime(Date.now());
    setScreen("solving");

    // Simulate solving delay
    setTimeout(() => {
      const gridCopy = deepCopyGrid(grid);
      const solved = solveSudoku(gridCopy);

      if (solved) {
        setSolvedGrid(gridCopy);
        const solveTime = Date.now() - solveStartTime;

        // Save to history
        const puzzleSolve: PuzzleSolve = {
          id: Date.now().toString(),
          timestamp: Date.now(),
          puzzle: grid,
          solution: gridCopy,
          source: screen === "manual" ? "manual" : "camera",
          difficulty: calculateDifficulty(grid),
          solveTime,
          isFavorite: false,
        };

        saveToHistory(puzzleSolve);
        setHistory(getHistory());
        setScreen("solution");
      } else {
        toast.error("No solution found! Please check the puzzle.");
        setScreen("review");
      }
    }, 1500);
  };

  const calculateDifficulty = (
    grid: SudokuGrid
  ): "easy" | "medium" | "hard" | "expert" => {
    const clues = grid.flat().filter((c) => c !== 0).length;
    if (clues >= 40) return "easy";
    if (clues >= 32) return "medium";
    if (clues >= 24) return "hard";
    return "expert";
  };

  const handleNewPuzzle = () => {
    setScreen("camera");
    setCapturedImage("");
    setCurrentGrid([]);
    setOriginalGrid([]);
    setSolvedGrid([]);
  };

  const handleDeleteFromHistory = (id: string) => {
    deleteFromHistory(id);
    setHistory(getHistory());
    toast.success("Puzzle deleted from history");
  };

  const handleToggleFavorite = (id: string) => {
    toggleFavoriteInStorage(id);
    setHistory(getHistory());
  };

  return (
    <>
      <Toaster position="top-center" theme="dark" />

      {showSplash && <SplashScreen onDismiss={() => setShowSplash(false)} />}

      {screen === "camera" && (
        <CameraScreen
          onCapture={handleCapture}
          onGalleryClick={handleCapture}
          onHistoryClick={() => setScreen("history")}
          onManualClick={() => setScreen("manual")}
          flashEnabled={flashEnabled}
          onFlashToggle={() => setFlashEnabled(!flashEnabled)}
        />
      )}

      {screen === "processing" && (
        <ProcessingScreen imageUrl={capturedImage} />
      )}

      {screen === "review" && (
        <GridReviewScreen
          grid={currentGrid}
          confidenceMap={confidenceMap}
          onRetake={handleRetake}
          onSolve={handleSolve}
        />
      )}

      {screen === "solving" && <SolvingScreen grid={currentGrid} />}

      {screen === "solution" && (
        <SolutionScreen
          originalGrid={originalGrid}
          solvedGrid={solvedGrid}
          solveTime={Date.now() - solveStartTime}
          onNewPuzzle={handleNewPuzzle}
        />
      )}

      {screen === "history" && (
        <HistoryScreen
          onBack={() => setScreen("camera")}
          history={history}
          onDelete={handleDeleteFromHistory}
          onToggleFavorite={handleToggleFavorite}
        />
      )}

      {screen === "manual" && (
        <ManualEntryScreen
          onBack={() => setScreen("camera")}
          onSolve={handleSolve}
        />
      )}

      <PWAInstallPrompt />
    </>
  );
}