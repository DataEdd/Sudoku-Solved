// grid.js - Sudoku grid management

const Grid = {
    // Track selected cell
    selectedCell: null,

    // Callback for when a cell value changes
    onCellChange: null,

    // Create grid DOM structure
    create(containerId, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return null;

        container.innerHTML = '';
        container.className = 'sudoku-grid';

        if (options.solving) {
            container.classList.add('sudoku-grid--solving');
        }

        for (let i = 0; i < 9; i++) {
            for (let j = 0; j < 9; j++) {
                const cell = document.createElement('div');
                cell.className = 'sudoku-cell';
                cell.dataset.row = i;
                cell.dataset.col = j;

                if (options.editable) {
                    cell.addEventListener('click', () => this.selectCell(containerId, i, j));
                }

                container.appendChild(cell);
            }
        }

        return container;
    },

    // Select a cell
    selectCell(containerId, row, col) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Clear previous selection
        container.querySelectorAll('.sudoku-cell--selected').forEach(
            cell => cell.classList.remove('sudoku-cell--selected')
        );

        const cell = container.querySelector(
            `.sudoku-cell[data-row="${row}"][data-col="${col}"]`
        );

        if (cell && !cell.classList.contains('sudoku-cell--fixed')) {
            cell.classList.add('sudoku-cell--selected');
            this.selectedCell = { row, col, element: cell, containerId };
        } else {
            this.selectedCell = null;
        }
    },

    // Set value in selected cell
    setSelectedValue(value) {
        if (!this.selectedCell) return;

        const { element } = this.selectedCell;
        element.textContent = value === 0 ? '' : value;
        element.classList.remove('sudoku-cell--error');

        if (this.onCellChange) {
            this.onCellChange(this.selectedCell.row, this.selectedCell.col, value);
        }
    },

    // Populate grid with values
    populate(containerId, values, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const cells = container.querySelectorAll('.sudoku-cell');
        const confidenceMap = options.confidenceMap || null;

        values.flat().forEach((val, idx) => {
            const cell = cells[idx];
            if (!cell) return;

            cell.textContent = val === 0 ? '' : val;
            cell.className = 'sudoku-cell';

            if (val !== 0 && options.markFixed) {
                cell.classList.add('sudoku-cell--fixed');
            }

            // Color-code cells by OCR confidence level
            if (confidenceMap && val !== 0) {
                const row = Math.floor(idx / 9);
                const col = idx % 9;
                const conf = confidenceMap[row][col];
                if (conf >= 0.8) {
                    cell.classList.add('confidence-high');
                } else if (conf >= 0.5) {
                    cell.classList.add('confidence-medium');
                    cell.title = `Confidence: ${Math.round(conf * 100)}%`;
                } else {
                    cell.classList.add('confidence-low');
                    cell.title = `Confidence: ${Math.round(conf * 100)}%`;
                }
            }
        });
    },

    // Get current grid values
    getValues(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return Utils.createEmptyGrid();

        const cells = container.querySelectorAll('.sudoku-cell');
        const grid = [];

        for (let i = 0; i < 9; i++) {
            grid[i] = [];
            for (let j = 0; j < 9; j++) {
                const cell = cells[i * 9 + j];
                const val = parseInt(cell?.textContent) || 0;
                grid[i][j] = val;
            }
        }

        return grid;
    },

    // Highlight errors
    highlightErrors(containerId, errors) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Clear existing errors
        container.querySelectorAll('.sudoku-cell--error').forEach(
            cell => cell.classList.remove('sudoku-cell--error')
        );

        errors.forEach(error => {
            if (error.type === 'row') {
                container.querySelectorAll(`[data-row="${error.index}"]`).forEach(
                    cell => cell.classList.add('sudoku-cell--error')
                );
            } else if (error.type === 'col') {
                container.querySelectorAll(`[data-col="${error.index}"]`).forEach(
                    cell => cell.classList.add('sudoku-cell--error')
                );
            } else if (error.type === 'box') {
                const startRow = error.row * 3;
                const startCol = error.col * 3;
                for (let i = 0; i < 3; i++) {
                    for (let j = 0; j < 3; j++) {
                        const cell = container.querySelector(
                            `[data-row="${startRow + i}"][data-col="${startCol + j}"]`
                        );
                        if (cell) cell.classList.add('sudoku-cell--error');
                    }
                }
            }
        });

        if (errors.length > 0) {
            container.classList.add('sudoku-grid--error');
            setTimeout(() => container.classList.remove('sudoku-grid--error'), 500);
        }
    },

    // Clear errors
    clearErrors(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.querySelectorAll('.sudoku-cell--error').forEach(
            cell => cell.classList.remove('sudoku-cell--error')
        );
    },

    // Animate solution reveal
    async revealSolution(containerId, originalGrid, solvedGrid) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const cells = container.querySelectorAll('.sudoku-cell');

        // Reveal by 3x3 boxes for nicer effect
        const boxOrder = [0, 1, 2, 3, 4, 5, 6, 7, 8];

        for (const boxIdx of boxOrder) {
            const boxRow = Math.floor(boxIdx / 3);
            const boxCol = boxIdx % 3;
            const startRow = boxRow * 3;
            const startCol = boxCol * 3;

            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++) {
                    const row = startRow + i;
                    const col = startCol + j;
                    const cellIdx = row * 9 + col;
                    const cell = cells[cellIdx];

                    if (originalGrid[row][col] === 0) {
                        cell.textContent = solvedGrid[row][col];
                        cell.classList.add('sudoku-cell--solved', 'sudoku-cell--revealing');
                        await Utils.sleep(20);
                    }
                }
            }
            await Utils.sleep(50);
        }
    },

    // Render mini grid for history thumbnails
    renderMiniGrid(grid) {
        let html = '';
        for (let i = 0; i < 9; i++) {
            for (let j = 0; j < 9; j++) {
                const val = grid[i][j];
                const filled = val !== 0 ? 'filled' : '';
                html += `<span class="mini-cell ${filled}">${val || ''}</span>`;
            }
        }
        return html;
    },

    // Clear selection
    clearSelection() {
        if (this.selectedCell) {
            this.selectedCell.element.classList.remove('sudoku-cell--selected');
            this.selectedCell = null;
        }
    }
};
