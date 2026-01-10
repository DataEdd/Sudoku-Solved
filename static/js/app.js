// app.js - Main application controller

const App = {
    state: {
        currentGrid: null,
        originalGrid: null,
        solvedGrid: null,
        source: null, // 'camera' | 'uploaded' | 'manual'
        iterations: 0
    },

    // Initialize app
    async init() {
        // Initialize first screen
        Screens.navigate('camera');

        // Bind all events
        this.bindEvents();

        // Register service worker
        this.registerServiceWorker();

        // Check for URL params (for PWA shortcuts)
        this.handleUrlParams();
    },

    // Register service worker for PWA
    async registerServiceWorker() {
        if ('serviceWorker' in navigator) {
            try {
                const registration = await navigator.serviceWorker.register('/sw.js');
                console.log('Service worker registered:', registration.scope);
            } catch (err) {
                console.error('Service worker registration failed:', err);
            }
        }
    },

    // Handle URL parameters
    handleUrlParams() {
        const params = new URLSearchParams(window.location.search);
        const screen = params.get('screen');
        if (screen && Screens.screens.includes(screen)) {
            Screens.navigate(screen);
        }
    },

    // Bind all event listeners
    bindEvents() {
        // Camera screen
        document.getElementById('btn-capture')?.addEventListener('click', () => this.captureImage());
        document.getElementById('btn-gallery')?.addEventListener('click', () => {
            document.getElementById('file-input')?.click();
        });
        document.getElementById('file-input')?.addEventListener('change', (e) => {
            if (e.target.files[0]) {
                this.uploadImage(e.target.files[0]);
                e.target.value = ''; // Reset for same file selection
            }
        });
        document.getElementById('btn-history')?.addEventListener('click', () => Screens.navigate('history'));
        document.getElementById('btn-flash')?.addEventListener('click', () => this.toggleFlash());
        document.getElementById('btn-manual')?.addEventListener('click', () => Screens.navigate('manual'));

        // Review screen
        document.getElementById('btn-retake')?.addEventListener('click', () => {
            Camera.stop();
            Screens.navigate('camera');
        });
        document.getElementById('btn-solve')?.addEventListener('click', () => this.solvePuzzle('review-grid'));

        // Solution screen
        document.getElementById('btn-new-puzzle')?.addEventListener('click', () => this.newPuzzle());
        document.getElementById('btn-share')?.addEventListener('click', () => this.shareSolution());
        document.getElementById('btn-save')?.addEventListener('click', () => this.saveSolution());

        // History screen
        document.getElementById('btn-history-back')?.addEventListener('click', () => Screens.navigate('camera'));
        document.querySelector('.filter-tabs')?.addEventListener('click', (e) => {
            if (e.target.classList.contains('tab')) {
                const filter = e.target.dataset.filter;
                Screens.setActiveTab(filter);
                Screens.renderHistory(filter);
            }
        });

        // Manual entry screen
        document.getElementById('btn-manual-back')?.addEventListener('click', () => Screens.navigate('camera'));
        document.getElementById('btn-clear-manual')?.addEventListener('click', () => {
            Grid.create('manual-grid', { editable: true });
        });
        document.getElementById('btn-solve-manual')?.addEventListener('click', () => this.solveManualEntry());

        // Number pad
        document.querySelector('.number-pad')?.addEventListener('click', (e) => {
            const btn = e.target.closest('.numpad-btn');
            if (btn) {
                const value = parseInt(btn.dataset.value);
                Grid.setSelectedValue(value);
            }
        });

        // Error modal
        document.getElementById('btn-error-dismiss')?.addEventListener('click', () => Screens.hideError());
        document.querySelector('.modal__backdrop')?.addEventListener('click', () => Screens.hideError());

        // Keyboard navigation
        document.addEventListener('keydown', (e) => this.handleKeydown(e));
    },

    // Handle keyboard input
    handleKeydown(e) {
        // Number input for manual entry
        if (Screens.current === 'manual' || Screens.current === 'review') {
            if (e.key >= '0' && e.key <= '9') {
                Grid.setSelectedValue(parseInt(e.key));
            } else if (e.key === 'Backspace' || e.key === 'Delete') {
                Grid.setSelectedValue(0);
            }
        }
    },

    // Toggle flash
    async toggleFlash() {
        const isOn = await Camera.toggleFlash();
        const btn = document.getElementById('btn-flash');
        if (btn) {
            btn.classList.toggle('active', isOn);
        }
    },

    // Capture image from camera
    async captureImage() {
        const blob = await Camera.capture();
        if (blob) {
            this.state.source = 'camera';
            const dataUrl = Camera.getImageDataUrl();
            await this.processImage(blob, dataUrl);
        }
    },

    // Upload image from file
    async uploadImage(file) {
        if (!file) return;

        this.state.source = 'uploaded';

        // Read file as data URL for preview
        const dataUrl = await new Promise(resolve => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.readAsDataURL(file);
        });

        await this.processImage(file, dataUrl);
    },

    // Process image for grid extraction
    async processImage(blob, dataUrl) {
        // Show processing screen
        Screens.navigate('processing');
        Screens.setProcessingImage(dataUrl);
        Screens.drawGridOverlay();
        Screens.updateProcessingText('Detecting grid...');

        // Stop camera to save resources
        Camera.stop();

        try {
            const result = await API.extractGrid(blob);

            if (result.success && result.grid) {
                this.state.currentGrid = result.grid;
                this.state.originalGrid = Utils.cloneGrid(result.grid);

                // Show review screen
                Screens.navigate('review');
                Grid.create('review-grid', { editable: true });
                Grid.populate('review-grid', result.grid, { markFixed: true });
            } else {
                Screens.showError(
                    'No Grid Detected',
                    result.message || 'Could not find a Sudoku puzzle in the image. Please try again with a clearer photo.'
                );
                Screens.navigate('camera');
            }
        } catch (err) {
            console.error('Extract error:', err);
            Screens.showError('Error', 'Failed to process image. Please check your connection and try again.');
            Screens.navigate('camera');
        }
    },

    // Solve puzzle from a grid
    async solvePuzzle(gridId) {
        // Get current grid values (user may have edited)
        const grid = Grid.getValues(gridId);

        // Validate
        const errors = Utils.validateGrid(grid);
        if (errors.length > 0) {
            Grid.highlightErrors(gridId, errors);
            Screens.showError('Invalid Puzzle', 'The puzzle has conflicting numbers. Please fix the highlighted cells.');
            return;
        }

        // Check minimum clues
        if (!Utils.hasMinimumClues(grid)) {
            Screens.showError('Not Enough Clues', 'Please enter at least 17 numbers for a valid Sudoku puzzle.');
            return;
        }

        this.state.originalGrid = Utils.cloneGrid(grid);

        // Show solving screen
        Screens.navigate('solving');
        Grid.create('solving-grid', { solving: true });
        Grid.populate('solving-grid', grid, { markFixed: true });

        // Animate iteration counter
        let displayIterations = 0;
        const counterInterval = setInterval(() => {
            displayIterations += Math.floor(Math.random() * 1000) + 500;
            Screens.updateIterationCounter(displayIterations);
        }, 100);

        try {
            const result = await API.solvePuzzle(grid);
            clearInterval(counterInterval);

            if (result.success && result.solution) {
                this.state.solvedGrid = result.solution;
                this.state.iterations = result.iterations;

                // Show solution screen
                Screens.navigate('solution');
                Screens.updateSolveStats(result.iterations);

                Grid.create('solution-grid');
                Grid.populate('solution-grid', this.state.originalGrid, { markFixed: true });
                await Grid.revealSolution('solution-grid', this.state.originalGrid, result.solution);
            } else {
                Screens.showError(
                    'No Solution Found',
                    result.message || 'This puzzle may not have a valid solution. Please check the numbers and try again.'
                );
                Screens.navigate('review');
            }
        } catch (err) {
            clearInterval(counterInterval);
            console.error('Solve error:', err);
            Screens.showError('Error', 'Failed to solve puzzle. Please check your connection and try again.');
            Screens.navigate('review');
        }
    },

    // Solve manual entry puzzle
    solveManualEntry() {
        this.state.source = 'manual';

        // Navigate to review for confirmation/editing
        const grid = Grid.getValues('manual-grid');
        this.state.currentGrid = grid;

        Screens.navigate('review');
        Grid.create('review-grid', { editable: true });
        Grid.populate('review-grid', grid, { markFixed: true });
    },

    // Start new puzzle
    newPuzzle() {
        this.state = {
            currentGrid: null,
            originalGrid: null,
            solvedGrid: null,
            source: null,
            iterations: 0
        };
        Screens.navigate('camera');
    },

    // Share solution
    async shareSolution() {
        if (!navigator.share) {
            // Fallback: copy to clipboard
            const text = `I solved a ${Utils.calculateDifficulty(this.state.originalGrid)} Sudoku in ${Utils.formatNumber(this.state.iterations)} iterations!`;
            try {
                await navigator.clipboard.writeText(text);
                this.showToast('Copied to clipboard!');
            } catch {
                console.log('Share not available');
            }
            return;
        }

        try {
            await navigator.share({
                title: 'Sudoku Solved!',
                text: `I solved a ${Utils.calculateDifficulty(this.state.originalGrid)} Sudoku in ${Utils.formatNumber(this.state.iterations)} iterations!`,
                url: window.location.origin
            });
        } catch (err) {
            if (err.name !== 'AbortError') {
                console.error('Share error:', err);
            }
        }
    },

    // Save solution to history
    saveSolution() {
        if (!this.state.originalGrid || !this.state.solvedGrid) return;

        History.add({
            source: this.state.source,
            originalGrid: this.state.originalGrid,
            solvedGrid: this.state.solvedGrid,
            iterations: this.state.iterations
        });

        // Visual feedback
        const btn = document.getElementById('btn-save');
        if (btn) {
            const originalHTML = btn.innerHTML;
            btn.innerHTML = '<svg class="icon icon--sm" viewBox="0 0 24 24"><path fill="currentColor" d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg> Saved!';
            btn.disabled = true;
            setTimeout(() => {
                btn.innerHTML = originalHTML;
                btn.disabled = false;
            }, 2000);
        }
    },

    // Show toast notification (simple implementation)
    showToast(message) {
        // Create toast element
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 100px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 12px 24px;
            border-radius: 8px;
            border: 1px solid var(--glass-border);
            z-index: 1000;
            animation: fadeIn 0.3s ease-out;
        `;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'fadeOut 0.3s ease-out forwards';
            setTimeout(() => toast.remove(), 300);
        }, 2000);
    }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => App.init());
