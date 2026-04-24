// ═══════════════════════════════════════════════════════════════════
// BATCH MODE (30 images)
// ═══════════════════════════════════════════════════════════════════

const batchMode = {
    // State
    batchImages: [],
    batchScores: {},
    fetching: false,

    // DOM elements (cached on init)
    grid: null,
    remainingDiv: null,
    loader: null,
    controls: null,

    // Initialize the mode
    async init() {
        const container = document.getElementById("batch-container");
        this.grid = container.querySelector("#batch-grid");
        this.remainingDiv = container.querySelector("#batch-remaining");
        this.loader = document.getElementById("loader");
        this.controls = container.querySelector(".batch-controls");

        // Load initial batch
        this.fetchBatch();
    },

    // Status update from main polling
    onStatusUpdate(status) {
        const scanFinished = status.cached >= status.total && status.total > 0;

        if (status.unscored === 0 && !scanFinished) {
            // Scanning in progress, show loader
            this.loader.classList.remove("hidden");
            if (this.controls) this.controls.classList.add("hidden");
            this.grid.classList.add("hidden");
            this.remainingDiv.classList.add("hidden");
            this.loader.querySelector(".loader-text").innerText = `Scanning images… (${status.cached}/${status.total} found)`;
        } else if (status.unscored === 0 && !this.fetching) {
            // No unscored images available
            this.loader.classList.remove("hidden");
            if (this.controls) this.controls.classList.add("hidden");
            this.grid.classList.add("hidden");
            this.remainingDiv.classList.add("hidden");
            if (scanFinished) {
                this.loader.querySelector(".loader-text").innerText = "All images scored!";
                this.loader.querySelector(".spinner").classList.add("hidden");

            } else {
                this.loader.querySelector(".loader-text").innerText = "No unscored images available…";
            }
            // && !status.scanning
        } else if (status.unscored > 0 && !this.fetching) {
            // Images available and not scanning
            this.loader.classList.add("hidden");
            if (this.controls) this.controls.classList.remove("hidden");
            this.grid.classList.remove("hidden");
            this.remainingDiv.classList.remove("hidden");
        }
    },

    // Fetch a batch of unscored images
    async fetchBatch() {
        this.fetching = true;
        this.loader.classList.remove("hidden");
        this.loader.querySelector(".loader-text").innerText = "Fetching 30 images…";

        try {
            const res = await fetch("/random_unscored_batch?n=30");
            if (res.status === 204) {
                this.loader.querySelector(".loader-text").innerText = "No unscored images available…";
                resumePolling();
                this.fetching = false;
                return;
            }

            const data = await res.json();
            this.batchImages = data.images || [];
            this.batchScores = {};

            this.renderBatchGrid();
            this.loader.classList.add("hidden");
            resumePolling();
            this.fetching = false;
        } catch (e) {
            console.error("Failed to fetch batch:", e);
            this.loader.querySelector(".loader-text").innerText = "Error fetching batch…";
            resumePolling();
            this.fetching = false;
        }
    },

    // Render the batch grid
    renderBatchGrid() {
        this.grid.innerHTML = "";

        this.batchImages.forEach((img, idx) => {
            const card = document.createElement("div");
            card.className = "batch-card";

            const imgEl = document.createElement("img");
            imgEl.src = "/image/" + encodeURIComponent(img.replace(/\\/g, "/"));
            imgEl.alt = `Image ${idx + 1}`;

            const info = document.createElement("div");
            info.className = "batch-card-info";
            info.innerText = `${idx + 1}/${this.batchImages.length}`;

            const scores = document.createElement("div");
            scores.className = "batch-card-scores";

            for (let s = 1; s <= 5; s++) {
                const btn = document.createElement("button");
                btn.className = "batch-score-btn";
                btn.innerText = s;
                btn.onclick = () => this.selectBatchScore(img, s, btn);
                if (this.batchScores[img] === s) {
                    btn.classList.add("selected");
                }
                scores.appendChild(btn);
            }

            card.appendChild(imgEl);
            card.appendChild(info);
            card.appendChild(scores);
            this.grid.appendChild(card);
        });

        this.updateBatchRemaining();
    },

    // Select a score for an image
    selectBatchScore(img, score, buttonEl) {
        this.batchScores[img] = score;

        // Update button styling
        const allBtns = buttonEl.parentElement.querySelectorAll(".batch-score-btn");
        allBtns.forEach(btn => btn.classList.remove("selected"));
        buttonEl.classList.add("selected");

        this.updateBatchRemaining();
    },

    // Update remaining count
    updateBatchRemaining() {
        const scored = Object.keys(this.batchScores).length;
        const total = this.batchImages.length;
        this.remainingDiv.innerText = `Scored: ${scored}/${total}`;
    },

    // Submit all scores
    async submitBatchScores() {
        if (Object.keys(this.batchScores).length === 0) {
            alert("Please score at least one image!");
            return;
        }

        this.loader.classList.remove("hidden");
        this.loader.querySelector(".loader-text").innerText = "Submitting scores…";

        const items = Object.entries(this.batchScores).map(([image, score]) => ({
            image,
            score,
        }));

        try {
            const res = await fetch("/submit_scores", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(items),
            });

            const result = await res.json();
            this.loader.querySelector(".loader-text").innerText = `Submitted! (${result.ok.length} OK, ${result.errors.length} errors)`;

            // Fetch next batch
            setTimeout(() => this.fetchBatch(), 1500);
        } catch (e) {
            console.error("Error submitting batch:", e);
            this.loader.querySelector(".loader-text").innerText = "Error submitting scores…";
        }
    },
};

// ═══════════════════════════════════════════════════════════════════
// STATE CLEANUP FOR MODE SWITCHING
// ═══════════════════════════════════════════════════════════════════

async function clearBatchModeState() {
    /**
     * Clear all batch mode state when switching away from this mode
     */
    try {
        // Clear the batchMode state
        batchMode.images = [];
        batchMode.batchScores = {};
        batchMode.selectedCount = 0;
        batchMode.loadingBatch = false;
        
        // Clear DOM references
        batchMode.table = null;
        batchMode.loader = null;
        batchMode.submitBtn = null;
    } catch (e) {
        console.error("Error clearing batch mode state:", e);
    }
}
