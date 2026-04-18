// ═══════════════════════════════════════════════════════════════════
// SINGLE MODE
// ═══════════════════════════════════════════════════════════════════

const singleMode = {
    // State
    currentImage: null,
    loadingImage: false,

    // DOM elements (cached on init)
    previewDiv: null,
    imgTag: null,
    controls: null,
    loader: null,

    // Initialize the mode
    async init() {
        const container = document.getElementById("single-container");
        this.previewDiv = container.querySelector("#single-preview");
        this.imgTag = container.querySelector("#preview");
        this.controls = container.querySelector("#single-controls");
        this.loader = document.getElementById("loader");

        // Load first image
        this.loadNext();
    },

    // Status update from main polling
    onStatusUpdate(status) {
        const scanFinished = status.cached >= status.total && status.total > 0;

        if (status.unscored === 0 && !scanFinished) {
            // Scanning in progress, show loader
            this.loader.classList.remove("hidden");
            this.previewDiv.classList.add("hidden");
            this.loader.querySelector(".loader-text").innerText = `Scanning images… (${status.cached}/${status.total} found)`;
        } else if (status.unscored > 0 && !this.loadingImage && !scanFinished) {
            // Images available, hide loader if no more scanning
            this.loader.classList.add("hidden");
        } else if (status.unscored === 0 && !this.loadingImage && !scanFinished) {
            // No unscored images yet, show waiting message
            this.loader.classList.remove("hidden");
            this.previewDiv.classList.add("hidden");
            this.imgTag.classList.add("hidden");
            this.controls.classList.add("hidden");
            this.loader.querySelector(".loader-text").innerText = "Waiting for new images…";
        } else if (scanFinished && status.unscored === 0 && !this.loadingImage) {
            // Scan finished, no more images
            this.loader.classList.remove("hidden");
            this.previewDiv.classList.add("hidden");
            this.imgTag.classList.add("hidden");
            this.controls.classList.add("hidden");
            this.loader.querySelector(".loader-text").innerText = "All images scored!";
            this.loader.querySelector(".spinner").classList.add("hidden");
        } else if (scanFinished && status.unscored > 0 && !this.loadingImage && !this.imgTag.src) {
            // Scan finished, images available, not currently showing one
            this.loader.classList.add("hidden");
        }
    },

    // Load next unscored image
    async loadNext() {
        if (this.loadingImage) return;
        this.loadingImage = true;
        stopPolling();  // Stop polling while loading

        try {
            const res = await fetch("/random_unscored");
            if (res.status === 204) {
                // No unscored images currently
                this.loader.querySelector(".loader-text").innerText = "Waiting for new images…";
                this.loader.classList.remove("hidden");
                this.previewDiv.classList.add("hidden");
                this.imgTag.classList.add("hidden");
                this.controls.classList.add("hidden");
                this.loadingImage = false;
                resumePolling();
                return;
            }

            const data = await res.json();
            this.currentImage = data.image;

            // Reset img tag
            this.imgTag.onload = null;
            this.imgTag.onerror = null;
            this.imgTag.src = "";
            this.imgTag.classList.add("hidden");

            this.loader.querySelector(".loader-text").innerText = "Loading image…";
            this.loader.classList.remove("hidden");
            this.previewDiv.classList.remove("hidden");
            this.controls.classList.add("hidden");

            // Set up load handlers
            this.imgTag.onload = () => {
                this.loader.classList.add("hidden");
                this.imgTag.classList.remove("hidden");
                this.controls.classList.remove("hidden");
                this.loadingImage = false;
                resumePolling();
            };

            this.imgTag.onerror = () => {
                console.error("Failed to load image:", this.currentImage);
                this.loader.querySelector(".loader-text").innerText = "Failed to load image. Skipping…";
                this.loadingImage = false;
                resumePolling();
                setTimeout(() => this.loadNext(), 500);
            };

            // Start loading
            this.imgTag.src = `/image/${encodeURIComponent(this.currentImage)}`;
            // this.imgTag.src =  `/thumbnail/${encodeURIComponent(this.currentImage)}`;
        } catch (e) {
            console.error("Error loading next image:", e);
            this.loader.querySelector(".loader-text").innerText = "Error loading image";
            this.loadingImage = false;
            resumePolling();
        }
    },

    // Submit a score for the current image
    async submitScore(score) {
        if (!this.currentImage) return;

        this.controls.classList.add("hidden");
        this.imgTag.classList.add("hidden");
        this.loader.classList.remove("hidden");

        try {
            const data = { image: this.currentImage, score }
            await fetch("/submit_score", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });
        } catch (e) {
            console.error("Error submitting score:", e);
            // this.loader.querySelector(".loader-text").innerText = "Error submitting score…";
            // this.loadingImage = false;
            // resumePolling();
        }


        await this.loadNext();
    },

    // Skip current image
    skipImage() {
        this.loadNext();
    },
};

// ═══════════════════════════════════════════════════════════════════
// STATE CLEANUP FOR MODE SWITCHING
// ═══════════════════════════════════════════════════════════════════

async function clearSingleModeState() {
    /**
     * Clear all single mode state when switching away from this mode
     */
    try {
        // Clear the singleMode state object
        singleMode.currentImage = null;
        singleMode.loadingImage = false;

        // Clear DOM references
        singleMode.previewDiv = null;
        singleMode.imgTag = null;
        singleMode.controls = null;
        singleMode.loader = null;
    } catch (e) {
        console.error("Error clearing single mode state:", e);
    }
}

/**
 * Reload image list for single mode when new images detected
 * Called by client.js periodically when in single/batch modes
 */
async function reloadImageList() {
    try {
        // Force reload unscored images
        if (typeof singleMode.loadNext === 'function') {
            await singleMode.loadNext();
        }
    } catch (e) {
        console.error("Error reloading image list:", e);
    }
}
