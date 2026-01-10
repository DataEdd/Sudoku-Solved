// utils.js - Utility functions

const Utils = {
    // Generate unique ID
    generateId() {
        return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    },

    // Format date for display
    formatDate(timestamp) {
        const d = new Date(timestamp);
        return d.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    },

    // Calculate puzzle difficulty based on given cells count
    calculateDifficulty(grid) {
        const filledCells = grid.flat().filter(v => v !== 0).length;
        if (filledCells >= 36) return 'Easy';
        if (filledCells >= 28) return 'Medium';
        if (filledCells >= 22) return 'Hard';
        return 'Expert';
    },

    // Validate Sudoku grid for conflicts
    validateGrid(grid) {
        const errors = [];

        // Check rows
        for (let i = 0; i < 9; i++) {
            const row = grid[i].filter(v => v !== 0);
            const seen = new Set();
            for (const val of row) {
                if (seen.has(val)) {
                    errors.push({ type: 'row', index: i });
                    break;
                }
                seen.add(val);
            }
        }

        // Check columns
        for (let j = 0; j < 9; j++) {
            const col = grid.map(row => row[j]).filter(v => v !== 0);
            const seen = new Set();
            for (const val of col) {
                if (seen.has(val)) {
                    errors.push({ type: 'col', index: j });
                    break;
                }
                seen.add(val);
            }
        }

        // Check 3x3 boxes
        for (let boxRow = 0; boxRow < 3; boxRow++) {
            for (let boxCol = 0; boxCol < 3; boxCol++) {
                const box = [];
                for (let i = 0; i < 3; i++) {
                    for (let j = 0; j < 3; j++) {
                        const val = grid[boxRow * 3 + i][boxCol * 3 + j];
                        if (val !== 0) box.push(val);
                    }
                }
                const seen = new Set();
                for (const val of box) {
                    if (seen.has(val)) {
                        errors.push({ type: 'box', row: boxRow, col: boxCol });
                        break;
                    }
                    seen.add(val);
                }
            }
        }

        return errors;
    },

    // Deep clone grid
    cloneGrid(grid) {
        return grid.map(row => [...row]);
    },

    // Create empty grid
    createEmptyGrid() {
        return Array(9).fill(null).map(() => Array(9).fill(0));
    },

    // Check if grid has minimum clues
    hasMinimumClues(grid, min = 17) {
        const filledCells = grid.flat().filter(v => v !== 0).length;
        return filledCells >= min;
    },

    // Debounce function
    debounce(fn, delay) {
        let timeoutId;
        return (...args) => {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => fn(...args), delay);
        };
    },

    // Format number with commas
    formatNumber(num) {
        return num.toLocaleString();
    },

    // Sleep helper for animations
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
};
