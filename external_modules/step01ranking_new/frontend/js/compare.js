// ═══════════════════════════════════════════════════════════════════
// COMPARE MODE
// ═══════════════════════════════════════════════════════════════════

const compareMode = {
    // State
    compareLeftImage: null,
    compareRightImage: null,
    compareLeftData: null,
    compareRightData: null,
    fetching: false,

    // DOM elements (cached on init)
    container: null,
    loader: null,
    manualScoreInput: null,

    // Initialize the mode
    async init() {
        this.container = document.getElementById("compare-container");
        this.loader = document.getElementById("loader");
        this.manualScoreInput = this.container.querySelector("#manual_score");

        // Load first comparison pair
        this.fetchNextComparePair();
    },

    // Status update from main polling
    onStatusUpdate(status) {
        const scanFinished = status.cached >= status.total && status.total > 0;

        // For compare mode, the metric is scored_uncompared, not unscored
        if (status.scored_uncompared === 0 && !scanFinished) {
            // Scanning and nothing to compare
            this.loader.classList.remove("hidden");
            this.loader.querySelector(".loader-text").innerText = `Scanning images… (${status.cached}/${status.total} found)`;
        } else if (status.scored_uncompared === 0 && !this.fetching) {
            // No images to compare
            this.loader.classList.remove("hidden");
            if (scanFinished) {
                this.loader.querySelector(".loader-text").innerText = "All images compared!";
                this.loader.querySelector(".spinner").classList.add("hidden");
            } else {
                this.loader.querySelector(".loader-text").innerText = "No images to compare yet…";
            }
        } else if (status.scored_uncompared > 0 && !this.fetching) {
            // Images available and not scanning
            this.loader.classList.add("hidden");
        }
    },

    // Fetch next pair to compare
    async fetchNextComparePair() {
        this.fetching = true;
        this.loader.classList.remove("hidden");
        this.loader.querySelector(".loader-text").innerText = "Finding pair to compare…";

        // Clear previous state BEFORE fetching to prevent stale data showing on error
        this.compareLeftImage = null;
        this.compareRightImage = null;
        this.compareLeftData = null;
        this.compareRightData = null;

        try {
            const res = await fetch(`/api/v2/ranking/next-pair`);

            // Check for empty response (no pairs available)
            const text = await res.text();
            if (!text || text === "{}" || res.status === 204) {
                this.loader.querySelector(".loader-text").innerText = "No more pairs to compare!";
                resumePolling();
                this.fetching = false;
                return;
            }

            const data = JSON.parse(text);
            if (data.error) {
                this.loader.querySelector(".loader-text").innerText = data.message || data.error;
                resumePolling();
                this.fetching = false;
                return;
            }

            // Debug log
            console.log("[NEXT-PAIR] Got pair:", data.left.filename, "vs", data.right.filename);

            this.compareLeftImage = data.left.filename;
            this.compareRightImage = data.right.filename;
            this.compareLeftData = data.left;
            this.compareRightData = data.right;

            this.renderComparePair();
            this.loader.classList.add("hidden");
            resumePolling();
            this.fetching = false;
        } catch (e) {
            console.error("Failed to fetch compare pair:", e);
            this.loader.querySelector(".loader-text").innerText = "Error fetching pair…";
            resumePolling();
            this.fetching = false;
        }
    },

    // Render the comparison pair
    renderComparePair() {
        const leftImg = this.container.querySelector("#compare-left");
        const rightImg = this.container.querySelector("#compare-right");

        // Add timestamp to force fresh fetch (cache buster)
        const ts = Date.now();
        const leftUrl = "/output/ranked/" + encodeURIComponent(this.compareLeftImage.replace(/\\/g, "/")) + "?t=" + ts;
        const rightUrl = "/output/ranked/" + encodeURIComponent(this.compareRightImage.replace(/\\/g, "/")) + "?t=" + ts;

        // Debug log
        console.log("[RENDER] Setting images:", leftUrl, rightUrl);

        // Force reload by setting src after clearing
        leftImg.src = "";
        leftImg.src = leftUrl;

        rightImg.src = "";
        rightImg.src = rightUrl;

        this.container.querySelector("#left_file_id").innerText = this.compareLeftImage;
        this.container.querySelector("#right_file_id").innerText = this.compareRightImage;

        // Handle both v2 API (score, confidence, comparison_count) and old API (score, etc)
        this.container.querySelector("#compare-left-score").innerText = this.compareLeftData.score ?? "0.5";
        this.container.querySelector("#compare-left-modifier").innerText = this.compareLeftData.confidence ?? "-";
        this.container.querySelector("#compare-left-count").innerText = this.compareLeftData.comparison_count ?? 0;
        this.container.querySelector("#compare-left-volatility").innerText = "-";

        this.container.querySelector("#compare-right-score").innerText = this.compareRightData.score ?? "0.5";
        this.container.querySelector("#compare-right-modifier").innerText = this.compareRightData.confidence ?? "-";
        this.container.querySelector("#compare-right-count").innerText = this.compareRightData.comparison_count ?? 0;
        this.container.querySelector("#compare-right-volatility").innerText = "-";


        // Show comparison buttons
        const buttons = this.container.querySelectorAll(".btn-compare");
        buttons.forEach(btn => btn.classList.remove("hidden"));
    },

    // Submit comparison result
    async submitComparison(winner) {
        const buttons = this.container.querySelectorAll(".btn-compare");
        buttons.forEach(btn => btn.classList.add("hidden"));

        this.loader.classList.remove("hidden");
        this.loader.querySelector(".loader-text").innerText = "Submitting comparison…";

        // Determine winner and loser based on which button was clicked
        const winnerImage = winner === "left" ? this.compareLeftImage : this.compareRightImage;
        const loserImage = winner === "left" ? this.compareRightImage : this.compareLeftImage;
        const winnerData = winner === "left" ? this.compareLeftData : this.compareRightData;
        const loserData = winner === "left" ? this.compareRightData : this.compareLeftData;

        try {
            const body = JSON.stringify({
                filename_a: this.compareLeftImage,
                filename_b: this.compareRightImage,
                winner: winnerImage,
            });

            // SUBMIT FIRST, then fetch next pair
            console.log("[SUBMIT] Sending:", this.compareLeftImage, "vs", this.compareRightImage, "winner:", winnerImage);
            await this.fetchNextComparePair();

            const res = await fetch("/api/v2/ranking/submit-comparison", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: body,
            });

            const result = await res.json();
            if (result.error) {
                console.error("[SUBMIT] Error:", result.error);
                this.loader.querySelector(".loader-text").innerText = `Error: ${result.error}`;
                buttons.forEach(btn => btn.classList.remove("hidden"));
                // Clear the pair since submission failed
                this.compareLeftImage = null;
                this.compareRightImage = null;
                return;
            }

            console.log("[SUBMIT] Success, fetching next...");

            // Update local data with returned scores/confidence
            if (result.images) {
                this.compareLeftData = result.images[this.compareLeftImage] || this.compareLeftData;
                this.compareRightData = result.images[this.compareRightImage] || this.compareRightData;
            }

            // Only fetch next pair after successful submission
            this.loader.querySelector(".loader-text").innerText = "Loading next pair…";
            // await this.fetchNextComparePair();

            // Wait for images to actually load before hiding loader
            // await new Promise(resolve => {
            //     const leftImg = this.container.querySelector("#compare-left");
            //     const rightImg = this.container.querySelector("#compare-right");
            //     let loaded = 0;
            //     const onLoad = () => {
            //         loaded++;
            //         if (loaded >= 2) resolve();
            //     };
            //     leftImg.onload = onLoad;
            //     rightImg.onload = onLoad;
            //     // Fallback timeout in case onload doesn't fire
            //     setTimeout(resolve, 500);
            // });

        } catch (e) {
            console.error("Error submitting comparison:", e);
            this.loader.querySelector(".loader-text").innerText = "Error submitting comparison…";
            buttons.forEach(btn => btn.classList.remove("hidden"));
        }
    },

    // Skip current comparison pair
    skipComparison() {
        this.fetchNextComparePair();
    },
};

// ═══════════════════════════════════════════════════════════════════
// STATE CLEANUP FOR MODE SWITCHING
// ═══════════════════════════════════════════════════════════════════

async function clearCompareModeState() {
    /**
     * Clear all compare mode state when switching away from this mode
     */
    try {
        // Clear the compareMode state
        compareMode.comparisons = [];
        compareMode.imageA = null;
        compareMode.imageB = null;
        compareMode.loadingComparison = false;

        // Clear DOM references
        compareMode.imgA = null;
        compareMode.imgB = null;
        compareMode.controls = null;
        compareMode.loader = null;
    } catch (e) {
        console.error("Error clearing compare mode state:", e);
    }
}
