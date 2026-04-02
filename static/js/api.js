// api.js - API communication layer

const API = {
    // Extract grid from image
    async extractGrid(imageBlob) {
        const formData = new FormData();
        formData.append('file', imageBlob);

        const response = await fetch('/api/extract', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Network error: ${response.status}`);
        }

        return response.json();
    },

    // Solve puzzle
    async solvePuzzle(grid, method = 'backtracking') {
        const response = await fetch('/api/solve', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ grid, method })
        });

        if (!response.ok) {
            throw new Error(`Network error: ${response.status}`);
        }

        return response.json();
    },

    // Health check
    async health() {
        const response = await fetch('/api/health');
        return response.json();
    }
};
