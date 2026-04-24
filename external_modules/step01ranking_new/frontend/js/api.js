// API client for ranking system v2 endpoints

class RankingAPI {
    constructor(baseUrl = "") {
        this.baseUrl = baseUrl;
        this.apiBase = `${baseUrl}/api/v2`;
    }

    async getStatus() {
        // Get ranking system status
        const response = await fetch(`${this.apiBase}/ranking/status`);
        if (!response.ok) throw new Error("Failed to get status");
        return await response.json();
    }

    async getNextPair() {
        // Get next pair for comparison (LRU handled on server side)
        const response = await fetch(`${this.apiBase}/ranking/next-pair`);
        if (response.status === 204) return null;
        if (!response.ok) throw new Error("Failed to get next pair");
        return await response.json();
    }

    async submitComparison(filenameA, filenameB, winner) {
        // Submit comparison result
        const payload = {
            filename_a: filenameA,
            filename_b: filenameB,
            winner: winner,
        };
        const response = await fetch(`${this.apiBase}/ranking/submit-comparison`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!response.ok) throw new Error("Failed to submit comparison");
        return await response.json();
    }

    async getGalleryImages(page = 1, perPage = 20, filters = {}) {
        // Get paginated gallery images
        const params = new URLSearchParams({
            page,
            per_page: perPage,
            tier_min: filters.tierMin || 0,
            tier_max: filters.tierMax || 9,
            confidence_min: filters.confidenceMin || 0.0,
            confidence_max: filters.confidenceMax || 1.0,
            sort: filters.sort || "score_desc",
        });
        const response = await fetch(`${this.apiBase}/gallery/images?${params}`);
        if (!response.ok) throw new Error("Failed to get gallery images");
        return await response.json();
    }

    async getTierImages(tier) {
        // Get all images in a specific tier
        const response = await fetch(`${this.apiBase}/gallery/tier/${tier}`);
        if (!response.ok) throw new Error("Failed to get tier images");
        return await response.json();
    }

    async searchImages(query, scoreMin = 0.0, scoreMax = 1.0) {
        // Search images
        const params = new URLSearchParams({
            query,
            score_min: scoreMin,
            score_max: scoreMax,
        });
        const response = await fetch(`${this.apiBase}/gallery/search?${params}`);
        if (!response.ok) throw new Error("Failed to search images");
        return await response.json();
    }
    async getImageHistory(filename) {
        // Get comparison history for a specific image
        const response = await fetch(`${this.apiBase}/gallery/history/${encodeURIComponent(filename)}`);
        if (!response.ok) throw new Error("Failed to get image history");
        return await response.json();
    }
    async submitSyncAll() {
        // Trigger full database-to-JSON backup
        const response = await fetch(`${this.apiBase}/ranking/sync-all`, {
            method: "POST"
        });
        if (!response.ok) throw new Error("Failed to trigger sync-all");
        return await response.json();
    }
}

// Create global API instance
const api = new RankingAPI();
