// ═══════════════════════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════════════════════

let poller = null;
let currentMode = "single";
let previousMode = null;
let newImagesCheckInterval = null;
// let scanning = false;

// Status data
let statusData = {
    unscored: 0,
    scored_uncompared: 0,
    compared: 0,
    cached: 0,
    total: 0
};

// DOM elements
const loader = document.getElementById("loader");
const statusText = document.getElementById("status-text");

// ═══════════════════════════════════════════════════════════════════
// STATUS POLLING
// ═══════════════════════════════════════════════════════════════════

async function pollStatus() {
    try {
        const res = await fetch("/status");
        const data = await res.json();

        statusData = data;
        updateStatusUI();
    } catch (e) {
        if (e) {
            console.error("Status polling error:", e);
        }

    }
}

function updateStatusUI() {
    // Format: "123 unscored | 456 uncompared | 864 partially compared | 789 compared | 234 cached | 567 total"
    const display = `${statusData.unscored}/${statusData.cached} unscored | ${statusData.uncompared} not compared  / ${statusData.partially_compared} partially compared / ${statusData.compared} fully compared | ${statusData.cached}/${statusData.total} cached`;
    statusText.innerText = display;

    // Update scanning state
    // scanning = statusData.scanning;

    // Check if scan is finished (cached >= total)
    //!statusData.scanning &&
    const scanFinished = statusData.cached >= statusData.total && statusData.total > 0;

    // Stop polling when scan is finished, unless a mode needs it
    if (scanFinished && !isModeBusy()) {
        stopPolling();
        //statusData.scanning || 
    } else if (!poller && (!scanFinished)) {
        // Restart polling if needed
        startPolling();
    }

    // If we're in a scoring mode and all images are scored, trigger a rescan for new images
    if ((currentMode === "single" || currentMode === "batch") && statusData.unscored === 0 && scanFinished) {
        // Trigger background scan to pick up any new images added since last scan
        fetch("/status").catch(e => console.error("Auto-rescan error:", e));
    }

    // Notify mode handlers of status update
    if (currentMode === "single" && typeof singleMode !== "undefined" && singleMode.onStatusUpdate) {
        singleMode.onStatusUpdate(statusData);
    } else if (currentMode === "batch" && typeof batchMode !== "undefined" && batchMode.onStatusUpdate) {
        batchMode.onStatusUpdate(statusData);
    } else if (currentMode === "compare" && typeof compareMode !== "undefined" && compareMode.onStatusUpdate) {
        compareMode.onStatusUpdate(statusData);
    }
}

function isModeBusy() {
    // Check if any mode is currently busy/waiting for user action
    if (currentMode === "single" && typeof singleMode !== "undefined" && singleMode.loadingImage) {
        return true;
    } else if (currentMode === "batch" && typeof batchMode !== "undefined" && batchMode.fetching) {
        return true;
    } else if (currentMode === "compare" && typeof compareMode !== "undefined" && compareMode.fetching) {
        return true;
    }
    return false;
}

function resumePolling() {
    // Resume polling after an action (submission, etc.)
    if (!poller) {
        startPolling();
    }
}

function startPolling() {
    if (!poller) {
        poller = setInterval(pollStatus, 3000);
        // Initial poll
        pollStatus();
    }
}

function stopPolling() {
    clearInterval(poller);
    poller = null;
}

// ═══════════════════════════════════════════════════════════════════
// MODE SWITCHING
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// MODE SWITCHING & STATE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════

async function switchMode(mode) {
    try {
        // Store previous mode for cleanup
        previousMode = currentMode;
        currentMode = mode;

        // 1. Clear previous mode state
        await clearModeState(previousMode);

        // 2. Clear all container contents
        document.getElementById("single-container").innerHTML = "";
        document.getElementById("batch-container").innerHTML = "";
        document.getElementById("compare-container").innerHTML = "";
        document.getElementById("gallery-container").innerHTML = "";

        // 3. Hide all containers
        document.getElementById("single-container").classList.remove("visible");
        document.getElementById("batch-container").classList.remove("visible");
        document.getElementById("compare-container").classList.remove("visible");
        document.getElementById("gallery-container").classList.remove("visible");

        // 4. Update UI
        updateModeButtons();

        // 5. Load new mode with cache-busting
        const timestamp = Date.now();
        if (mode === "single") {
            const container = document.getElementById("single-container");
            const html = await fetch(`/single.html?t=${timestamp}`).then(r => r.text());
            container.innerHTML = html;
            // Wait for scripts to potentially load
            await new Promise(resolve => setTimeout(resolve, 100));
            if (typeof singleMode !== "undefined" && singleMode.init) {
                await singleMode.init();
            } else {
                console.error("singleMode.init not available after loading HTML");
            }
            container.classList.add("visible");
        } else if (mode === "batch") {
            const container = document.getElementById("batch-container");
            const html = await fetch(`/batch.html?t=${timestamp}`).then(r => r.text());
            container.innerHTML = html;
            // Wait for scripts to potentially load
            await new Promise(resolve => setTimeout(resolve, 100));
            if (typeof batchMode !== "undefined" && batchMode.init) {
                await batchMode.init();
            } else {
                console.error("batchMode.init not available after loading HTML");
            }
            container.classList.add("visible");
        } else if (mode === "compare") {
            const container = document.getElementById("compare-container");
            const html = await fetch(`/compare.html?t=${timestamp}`).then(r => r.text());
            container.innerHTML = html;
            // Wait for scripts to potentially load
            await new Promise(resolve => setTimeout(resolve, 100));
            if (typeof compareMode !== "undefined" && compareMode.init) {
                await compareMode.init();
            } else {
                console.error("compareMode.init not available after loading HTML");
            }
            container.classList.add("visible");
        } else if (mode === "gallery") {
            const container = document.getElementById("gallery-container");
            const html = await fetch(`/gallery.html?t=${timestamp}`).then(r => r.text());
            container.innerHTML = html;
            // Wait for scripts to potentially load
            await new Promise(resolve => setTimeout(resolve, 100));
            if (typeof initializeGallery !== "undefined") {
                await initializeGallery();
            } else {
                console.error("initializeGallery not available after loading HTML");
            }
            container.classList.add("visible");
        }
        
        // Resume polling after mode switch
        startPolling();
    } catch (e) {
        console.error("Error switching mode to", mode, ":", e);
        currentMode = previousMode; // Revert to previous mode on error
        updateModeButtons();
    }
}

async function clearModeState(mode) {
    /**
     * Clear global state and event listeners from the previous mode
     */
    switch(mode) {
        case 'single':
            if (typeof clearSingleModeState === 'function') {
                await clearSingleModeState();
            }
            break;
        case 'batch':
            if (typeof clearBatchModeState === 'function') {
                await clearBatchModeState();
            }
            break;
        case 'compare':
            if (typeof clearCompareModeState === 'function') {
                await clearCompareModeState();
            }
            break;
        case 'gallery':
            if (typeof clearGalleryModeState === 'function') {
                await clearGalleryModeState();
            }
            break;
    }
}

function updateModeButtons() {
    document.getElementById("mode-single").classList.toggle("active", currentMode === "single");
    document.getElementById("mode-batch").classList.toggle("active", currentMode === "batch");
    document.getElementById("mode-compare").classList.toggle("active", currentMode === "compare");
    document.getElementById("mode-gallery").classList.toggle("active", currentMode === "gallery");
}

// ═══════════════════════════════════════════════════════════════════
// NEW IMAGE DETECTION & AUTO-REFRESH
// ═══════════════════════════════════════════════════════════════════

async function checkForNewImages() {
    /**
     * Periodically check if new images have been added to the input folder
     * If new images detected, refresh the current mode's image list
     */
    try {
        const response = await fetch("/status");
        const status = await response.json();

        // If we have new unscored images and we're in a scoring mode, refresh
        if (status.unscored > 0 && (currentMode === 'single' || currentMode === 'batch')) {
            // Call mode-specific reload (defined in single.js and batch.js)
            if (typeof reloadImageList === 'function') {
                await reloadImageList();
            }
        }
    } catch (error) {
        console.error("Error checking for new images:", error);
    }
}

function startNewImageMonitoring() {
    /**
     * Start periodic check for new images (every 5 seconds)
     */
    if (!newImagesCheckInterval) {
        console.log("Starting new image monitoring...");
        newImagesCheckInterval = setInterval(checkForNewImages, 5000);
    }
}

function stopNewImageMonitoring() {
    /**
     * Stop periodic check for new images
     */
    if (newImagesCheckInterval) {
        console.log("Stopping new image monitoring...");
        clearInterval(newImagesCheckInterval);
        newImagesCheckInterval = null;
    }
}

// ═══════════════════════════════════════════════════════════════════
// INITIALIZE
// ═══════════════════════════════════════════════════════════════════

startPolling();
startNewImageMonitoring();
switchMode("single");


