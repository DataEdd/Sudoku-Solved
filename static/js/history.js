// history.js - History storage and management using LocalStorage

const History = {
    STORAGE_KEY: 'sudoku_history',
    MAX_ENTRIES: 50,

    // Get all history entries
    getAll() {
        try {
            const data = localStorage.getItem(this.STORAGE_KEY);
            return data ? JSON.parse(data) : [];
        } catch (e) {
            console.error('Error reading history:', e);
            return [];
        }
    },

    // Save history to storage
    _save(history) {
        try {
            localStorage.setItem(this.STORAGE_KEY, JSON.stringify(history));
        } catch (e) {
            console.error('Error saving history:', e);
        }
    },

    // Add new entry
    add(entry) {
        const history = this.getAll();
        const newEntry = {
            id: Utils.generateId(),
            timestamp: Date.now(),
            source: entry.source, // 'camera' | 'uploaded' | 'manual'
            originalGrid: entry.originalGrid,
            solvedGrid: entry.solvedGrid,
            iterations: entry.iterations,
            difficulty: Utils.calculateDifficulty(entry.originalGrid),
            isFavorite: false
        };

        history.unshift(newEntry);

        // Limit history size
        if (history.length > this.MAX_ENTRIES) {
            history.pop();
        }

        this._save(history);
        return newEntry;
    },

    // Toggle favorite
    toggleFavorite(id) {
        const history = this.getAll();
        const entry = history.find(e => e.id === id);
        if (entry) {
            entry.isFavorite = !entry.isFavorite;
            this._save(history);
        }
        return entry;
    },

    // Delete entry
    delete(id) {
        const history = this.getAll();
        const filtered = history.filter(e => e.id !== id);
        this._save(filtered);
    },

    // Filter by type
    filter(type) {
        const history = this.getAll();
        if (type === 'all') return history;
        if (type === 'favorites') return history.filter(e => e.isFavorite);
        return history.filter(e => e.source === type);
    },

    // Get single entry
    get(id) {
        return this.getAll().find(e => e.id === id);
    },

    // Clear all history
    clear() {
        this._save([]);
    }
};
