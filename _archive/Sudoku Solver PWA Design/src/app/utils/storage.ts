import { PuzzleSolve } from "../types";

const HISTORY_KEY = "sudoku_solver_history";

export function saveToHistory(solve: PuzzleSolve): void {
  const history = getHistory();
  history.unshift(solve); // Add to beginning
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

export function getHistory(): PuzzleSolve[] {
  const stored = localStorage.getItem(HISTORY_KEY);
  return stored ? JSON.parse(stored) : [];
}

export function deleteFromHistory(id: string): void {
  const history = getHistory().filter((item) => item.id !== id);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

export function toggleFavorite(id: string): void {
  const history = getHistory();
  const index = history.findIndex((item) => item.id === id);
  if (index !== -1) {
    history[index].isFavorite = !history[index].isFavorite;
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  }
}
