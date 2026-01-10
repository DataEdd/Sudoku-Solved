// screens.js - Screen management and rendering

const Screens = {
    current: null,
    screens: ['camera', 'processing', 'review', 'solving', 'solution', 'history', 'manual'],

    // Navigate to screen
    navigate(screenName) {
        if (!this.screens.includes(screenName)) {
            console.error('Invalid screen:', screenName);
            return;
        }

        // Exit current screen
        if (this.current) {
            const currentScreen = document.getElementById(`screen-${this.current}`);
            if (currentScreen) {
                currentScreen.classList.remove('screen--active');
            }
            this.onExit(this.current);
        }

        // Enter new screen
        const newScreen = document.getElementById(`screen-${screenName}`);
        if (newScreen) {
            newScreen.classList.add('screen--active');
        }

        this.current = screenName;
        this.onEnter(screenName);
    },

    // Screen enter callbacks
    onEnter(screenName) {
        switch (screenName) {
            case 'camera':
                Camera.initialize('camera-feed', 'camera-canvas');
                break;
            case 'history':
                this.renderHistory('all');
                break;
            case 'manual':
                Grid.create('manual-grid', { editable: true });
                break;
        }
    },

    // Screen exit callbacks
    onExit(screenName) {
        switch (screenName) {
            case 'camera':
                // Keep camera running unless we're going to a different flow
                break;
            case 'manual':
                Grid.clearSelection();
                break;
            case 'review':
                Grid.clearSelection();
                break;
        }
    },

    // Render history cards
    renderHistory(filter = 'all') {
        const container = document.getElementById('history-grid');
        if (!container) return;

        const entries = History.filter(filter);

        if (entries.length === 0) {
            container.innerHTML = `
                <div class="history-empty">
                    <p>No puzzles yet</p>
                    <p>Solve some puzzles to see them here!</p>
                </div>
            `;
            return;
        }

        container.innerHTML = entries.map(entry => `
            <article class="history-card" data-id="${entry.id}">
                <div class="history-card__thumbnail">
                    <div class="mini-grid">${Grid.renderMiniGrid(entry.originalGrid)}</div>
                </div>
                <div class="history-card__info">
                    <time class="history-card__date">${Utils.formatDate(entry.timestamp)}</time>
                    <span class="difficulty-badge difficulty-badge--${entry.difficulty.toLowerCase()}">${entry.difficulty}</span>
                </div>
            </article>
        `).join('');
    },

    // Show error modal
    showError(title, message) {
        const modal = document.getElementById('error-modal');
        const titleEl = document.getElementById('error-title');
        const messageEl = document.getElementById('error-message');

        if (modal && titleEl && messageEl) {
            titleEl.textContent = title;
            messageEl.textContent = message;
            modal.hidden = false;
        }
    },

    // Hide error modal
    hideError() {
        const modal = document.getElementById('error-modal');
        if (modal) {
            modal.hidden = true;
        }
    },

    // Update processing text
    updateProcessingText(text) {
        const el = document.querySelector('.processing__text');
        if (el) el.textContent = text;
    },

    // Update iteration counter
    updateIterationCounter(count) {
        const el = document.getElementById('iteration-count');
        if (el) el.textContent = Utils.formatNumber(count);
    },

    // Update solve stats
    updateSolveStats(iterations) {
        const el = document.getElementById('solve-stats');
        if (el) el.textContent = `Solved in ${Utils.formatNumber(iterations)} iterations`;
    },

    // Set processing image
    setProcessingImage(dataUrl) {
        const img = document.getElementById('processing-image');
        if (img) img.src = dataUrl;
    },

    // Draw grid overlay animation
    drawGridOverlay() {
        const svg = document.querySelector('.processing__grid-overlay');
        if (!svg) return;

        // Clear existing lines
        svg.innerHTML = '';

        // Create grid lines
        for (let i = 1; i < 9; i++) {
            const pos = (i / 9) * 100;

            // Horizontal line
            const hLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            hLine.setAttribute('x1', '0');
            hLine.setAttribute('y1', pos);
            hLine.setAttribute('x2', '100');
            hLine.setAttribute('y2', pos);
            svg.appendChild(hLine);

            // Vertical line
            const vLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            vLine.setAttribute('x1', pos);
            vLine.setAttribute('y1', '0');
            vLine.setAttribute('x2', pos);
            vLine.setAttribute('y2', '100');
            svg.appendChild(vLine);
        }
    },

    // Update filter tabs
    setActiveTab(filter) {
        document.querySelectorAll('.filter-tabs .tab').forEach(tab => {
            tab.classList.toggle('tab--active', tab.dataset.filter === filter);
        });
    }
};
