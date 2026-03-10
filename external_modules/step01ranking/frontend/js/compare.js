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

        try {
            const score = this.manualScoreInput ? this.manualScoreInput.value : "0";
            const params = new URLSearchParams({ score });

            const res = await fetch(`/compare/next/?${params.toString()}`);

            // Check for empty response (no pairs available)
            const text = await res.text();
            if (!text || text === "{}") {
                this.loader.querySelector(".loader-text").innerText = "No more pairs to compare!";
                resumePolling();
                this.fetching = false;
                return;
            }

            const data = JSON.parse(text);
            this.compareLeftImage = data.left.image;
            this.compareRightImage = data.right.image;
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

        leftImg.src = "/image/" + encodeURIComponent(this.compareLeftImage.replace(/\\/g, "/"));
        rightImg.src = "/image/" + encodeURIComponent(this.compareRightImage.replace(/\\/g, "/"));

        this.container.querySelector("#left_file_id").innerText = this.compareLeftImage;
        this.container.querySelector("#right_file_id").innerText = this.compareRightImage;

        this.container.querySelector("#compare-left-score").innerText = this.compareLeftData.score;
        this.container.querySelector("#compare-left-modifier").innerText = this.compareLeftData.score_modifier;

        this.container.querySelector("#compare-left-count").innerText = this.compareLeftData.comparison_count;

        this.container.querySelector("#compare-right-score").innerText = this.compareRightData.score;
        this.container.querySelector("#compare-right-modifier").innerText = this.compareRightData.score_modifier;

        this.container.querySelector("#compare-right-count").innerText = this.compareRightData.comparison_count;


        this.container.querySelector("#compare-right-score").innerText = this.compareRightData.score;
        this.container.querySelector("#compare-right-count").innerText = this.compareRightData.comparison_count;
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
            const res = await fetch("/compare/submit", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    winner_image: winnerImage,
                    loser_image: loserImage,
                    winner_data: winnerData,
                    loser_data: loserData,
                }),
            });

            this.loader.querySelector(".loader-text").innerText = "Updating…";
            this.fetchNextComparePair();

            const result = await res.json();
            if (!result.ok) {
                this.loader.querySelector(".loader-text").innerText = `Error: ${result.error}`;
                buttons.forEach(btn => btn.classList.remove("hidden"));
                return;
            }

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
