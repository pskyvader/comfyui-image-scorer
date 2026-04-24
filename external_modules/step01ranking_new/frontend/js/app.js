// Main application logic for compare page

class RankingApp {
    constructor() {
        this.currentPair = null;
        this.nextPair = null;
        this.isLoading = false;
        this.imageCache = {};
        
        this.cacheElements();
        this.attachEventListeners();
        this.initialize();
    }

    cacheElements() {
        this.statusEl = document.getElementById("comparison-status");
        this.leftImg = document.getElementById("left-image");
        this.rightImg = document.getElementById("right-image");
        this.leftFilename = document.getElementById("left-filename");
        this.rightFilename = document.getElementById("right-filename");
        this.leftScore = document.getElementById("left-score");
        this.rightScore = document.getElementById("right-score");
        this.leftConfidence = document.getElementById("left-confidence");
        this.rightConfidence = document.getElementById("right-confidence");
        this.leftCount = document.getElementById("left-count");
        this.rightCount = document.getElementById("right-count");
        this.leftButton = document.getElementById("left-button");
        this.rightButton = document.getElementById("right-button");
        this.skipButton = document.getElementById("skip-button");
        this.statsTotal = document.getElementById("stat-total");
        this.statsRanked = document.getElementById("stat-ranked");
        this.statsComparisons = document.getElementById("stat-comparisons");
        this.statusBadge = document.getElementById("status-indicator");
        this.errorBox = document.getElementById("error-box");
        this.errorTitle = document.getElementById("error-title");
        this.errorMessage = document.getElementById("error-message");
        this.cardsContainer = document.getElementById("cards-container");
        this.loadingOverlay = document.getElementById("loading-overlay");
        
        // Debug & Backup
        this.backupBtn = document.getElementById("backup-button");
        this.debugBtn = document.getElementById("toggle-debug");
        this.debugPanel = document.getElementById("debug-panel");
        this.debugContent = document.getElementById("debug-content");
    }

    attachEventListeners() {
        this.leftButton.addEventListener("click", () => this.submitVote("left"));
        this.rightButton.addEventListener("click", () => this.submitVote("right"));
        
        const leftCard = document.querySelector(".left-card");
        const rightCard = document.querySelector(".right-card");
        
        if (leftCard) {
            leftCard.addEventListener("click", (e) => {
                if (e.target !== this.leftButton) this.submitVote("left");
            });
        }
        if (rightCard) {
            rightCard.addEventListener("click", (e) => {
                if (e.target !== this.rightButton) this.submitVote("right");
            });
        }
        
        if (this.skipButton) {
            this.skipButton.addEventListener("click", () => this.loadPair());
        }

        if (this.backupBtn) {
            this.backupBtn.addEventListener("click", () => this.handleBackup());
        }

        if (this.debugBtn) {
            this.debugBtn.addEventListener("click", () => {
                this.debugPanel.classList.toggle("hidden");
            });
        }
    }

    async handleBackup() {
        this.backupBtn.textContent = "Syncing...";
        this.backupBtn.disabled = true;
        try {
            const res = await api.submitSyncAll();
            Utils.showToast(`Backup Complete: ${res.synced_count} images synced.`, "success");
        } catch (e) {
            Utils.showToast(`Backup Failed: ${e.message}`, "error");
        } finally {
            this.backupBtn.textContent = "Backup";
            this.backupBtn.disabled = false;
        }
    }

    async initialize() {
        try {
            this.updateStatus("Initializing...");
            await this.loadPair();
            this.preloadNextPair();
            await this.updateStats();
            this.updateStatus("Ready");
        } catch (error) {
            this.updateStatus(`Error: ${error.message}`);
            Utils.showToast(`Error: ${error.message}`, "error");
        }
    }

    async loadPair() {
        this.isLoading = true;
        this.updateStatus("Loading pair...");
        this.showLoading(true);

        try {
            const pair = await api.getNextPair();

            if (!pair) {
                this.updateStatus("No more pairs to compare!");
                this.leftButton.disabled = true;
                this.rightButton.disabled = true;
                this.showLoading(false);
                return;
            }

            this.currentPair = pair;
            await this.renderPair();
            this.updateStatus("Ready");
            this.isLoading = false;
            this.showLoading(false);
        } catch (error) {
            this.showErrorBox("System Not Ready", error.message);
            this.updateStatus(`Error`);
            this.showLoading(false);
            throw error;
        }
    }

    showLoading(show) {
        if (this.loadingOverlay) {
            if (show) {
                this.loadingOverlay.classList.remove("pointer-events-none", "opacity-0");
                this.loadingOverlay.classList.add("opacity-100");
            } else {
                this.loadingOverlay.classList.add("pointer-events-none", "opacity-0");
                this.loadingOverlay.classList.remove("opacity-100");
            }
        }
    }

    showErrorBox(title, message) {
        if (this.cardsContainer) this.cardsContainer.classList.add("hidden");
        if (this.skipButton) this.skipButton.classList.add("hidden");
        if (this.errorBox) {
            this.errorBox.classList.remove("hidden");
            this.errorTitle.textContent = title;
            this.errorMessage.textContent = message;
        }
    }

    async preloadNextPair() {
        try {
            const nextPair = await api.getNextPair();
            if (nextPair) {
                this.nextPair = nextPair;
                const leftSrc = this.getImageUrl(nextPair.left.filename);
                const rightSrc = this.getImageUrl(nextPair.right.filename);
                await Promise.all([
                    Utils.preloadImage(leftSrc),
                    Utils.preloadImage(rightSrc),
                ]);
            }
        } catch (error) {
            console.log("Pre-load failed:", error);
        }
    }

    async renderPair() {
        const left = this.currentPair.left;
        const right = this.currentPair.right;
        const rationale = this.currentPair.rationale;

        if (left.filename === right.filename) {
            this.loadPair();
            return;
        }

        this.showLoading(true);
        this.leftImg.classList.remove("opacity-100");
        this.rightImg.classList.remove("opacity-100");
        this.leftImg.classList.add("opacity-0");
        this.rightImg.classList.add("opacity-0");

        this.leftFilename.textContent = left.filename;
        this.rightFilename.textContent = right.filename;
        this.leftScore.textContent = Utils.formatScore(left.score);
        this.rightScore.textContent = Utils.formatScore(right.score);
        this.leftConfidence.textContent = Utils.formatScore(left.confidence);
        this.rightConfidence.textContent = Utils.formatScore(right.confidence);
        this.leftCount.textContent = left.comparison_count;
        this.rightCount.textContent = right.comparison_count;

        // Render Debug Info
        if (this.debugContent && rationale) {
            this.debugContent.innerHTML = `
                <div>
                    <p class="text-purple-400">Selection Lead (Seed)</p>
                    <p class="text-white truncate">${rationale.seed}</p>
                    <p class="mt-1 text-purple-400">Strategy</p>
                    <p class="text-white">${rationale.strategy}</p>
                </div>
                <div>
                    <p class="text-purple-400">Score Distance</p>
                    <p class="text-white font-bold">${rationale.score_diff}</p>
                    <p class="mt-1 text-purple-400">Allowed Range</p>
                    <p class="text-white font-bold">${rationale.allowed_range}</p>
                </div>
            `;
        }

        this.leftImg.src = this.getImageUrl(left.filename);
        this.rightImg.src = this.getImageUrl(right.filename);

        const waitForImage = (img) => new Promise(resolve => {
            if (img.complete && img.naturalWidth !== 0) return resolve();
            img.onload = resolve;
            img.onerror = resolve;
            setTimeout(resolve, 10000);
        });

        await Promise.all([
            waitForImage(this.leftImg),
            waitForImage(this.rightImg)
        ]);

        this.leftImg.classList.remove("opacity-0");
        this.rightImg.classList.remove("opacity-0");
        this.leftImg.classList.add("opacity-100");
        this.rightImg.classList.add("opacity-100");
        this.showLoading(false);
    }

    getImageUrl(filename) {
        return `/output/ranked/${encodeURIComponent(filename)}`;
    }

    async submitVote(winner) {
        if (this.isLoading) return;

        this.isLoading = true;
        this.leftButton.disabled = true;
        this.rightButton.disabled = true;

        try {
            const filenameA = this.currentPair.left.filename;
            const filenameB = this.currentPair.right.filename;
            const winnerFilename = winner === "left" ? filenameA : filenameB;

            this.updateStatus("Recording vote...");

            api.submitComparison(filenameA, filenameB, winnerFilename).catch((error) => {
                console.error("Background submission error:", error);
                Utils.showToast(`Submission error: ${error.message}`, "error");
            });

            if (this.nextPair) {
                this.currentPair = this.nextPair;
                this.nextPair = null;
                await this.renderPair();
                this.updateStatus("Ready");
                this.preloadNextPair();
            } else {
                await this.loadPair();
            }

            this.updateStats();
            Utils.showToast("Vote recorded!", "success");
        } catch (error) {
            Utils.showToast(`Error: ${error.message}`, "error");
            this.updateStatus(`Error: ${error.message}`);
        } finally {
            this.isLoading = false;
            this.leftButton.disabled = false;
            this.rightButton.disabled = false;
        }
    }

    async updateStats() {
        try {
            const status = await api.getStatus();
            this.statsTotal.textContent = status.total_images;
            this.statsRanked.textContent = status.ranked_images;
            this.statsComparisons.textContent = status.total_comparisons;

            const progress = status.total_images > 0 ? status.ranked_images / status.total_images : 0;
            if (progress > 0.8) {
                this.statusBadge.style.background = "#4CAF50";
            } else if (progress > 0.5) {
                this.statusBadge.style.background = "#FF9800";
            } else {
                this.statusBadge.style.background = "#2196F3";
            }

            if (status.min_images && status.total_images < status.min_images) {
                this.leftButton.style.display = "none";
                this.rightButton.style.display = "none";
                this.updateStatus(`Need at least ${status.min_images} images to start comparisons.`);
            } else {
                this.leftButton.style.display = "inline-block";
                this.rightButton.style.display = "inline-block";
            }
        } catch (error) {
            console.log("Failed to update stats:", error);
        }
    }

    updateStatus(message) {
        this.statusEl.innerHTML = `<p>${message}</p>`;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    window.app = new RankingApp();
});
