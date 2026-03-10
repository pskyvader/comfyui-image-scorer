// ═══════════════════════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════════════════════

let poller = null;
let currentMode = "single";
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

async function switchMode(mode) {
    currentMode = mode;
    updateModeButtons();

    // Hide all containers
    document.getElementById("single-container").classList.remove("visible");
    document.getElementById("batch-container").classList.remove("visible");
    document.getElementById("compare-container").classList.remove("visible");

    // Show selected container and initialize mode
    if (mode === "single") {
        const container = document.getElementById("single-container");
        if (container.innerHTML === "") {
            const html = await fetch("/single.html").then(r => r.text());
            container.innerHTML = html;
            if (typeof singleMode !== "undefined" && singleMode.init) await singleMode.init();
        }
        container.classList.add("visible");
    } else if (mode === "batch") {
        const container = document.getElementById("batch-container");
        if (container.innerHTML === "") {
            const html = await fetch("/batch.html").then(r => r.text());
            container.innerHTML = html;
            if (typeof batchMode !== "undefined" && batchMode.init) await batchMode.init();
        }
        container.classList.add("visible");
    } else if (mode === "compare") {
        const container = document.getElementById("compare-container");
        if (container.innerHTML === "") {
            const html = await fetch("/compare.html").then(r => r.text());
            container.innerHTML = html;
            if (typeof compareMode !== "undefined" && compareMode.init) await compareMode.init();
        }
        container.classList.add("visible");
    }
}

function updateModeButtons() {
    document.getElementById("mode-single").classList.toggle("active", currentMode === "single");
    document.getElementById("mode-batch").classList.toggle("active", currentMode === "batch");
    document.getElementById("mode-compare").classList.toggle("active", currentMode === "compare");
}

// ═══════════════════════════════════════════════════════════════════
// INITIALIZE
// ═══════════════════════════════════════════════════════════════════

startPolling();
switchMode("single");

