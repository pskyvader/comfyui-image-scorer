// ═══════════════════════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════════════════════

let poller = null;
let currentMode = "single"; // single, batch, compare
let currentImage = null;
let loadingImage = false;
let polling = false;

// Batch mode state
let batchImages = [];
let batchScores = {};

// Compare mode state
let compareLeftImage = null;
let compareRightImage = null;
let compareLeftData = null;
let compareRightData = null;

// DOM elements
const loader = document.getElementById("loader");
const previewDiv = document.getElementById("single-preview");
const imgTag = document.getElementById("preview");
const controls = document.getElementById("single-controls");
const counterDiv = document.getElementById("counter");
const statusDiv = document.getElementById("status");

// ═══════════════════════════════════════════════════════════════════
// POLLING & STATUS
// ═══════════════════════════════════════════════════════════════════

async function pollStatus() {
    if (loadingImage || polling) return;
    polling = true;

    try {
        const res = await fetch("/status");
        const s = await res.json();
        updateStatusUI(s);

        if (s.state === "ready") {
            const src = imgTag.src;
            if (!loadingImage && src.trim().length === 0 && currentMode === "single") {
                loadNext();
            } else {
                if (s.cached_total >= s.total) {
                    stopPolling();
                }
            }
        }

        if (s.state === "done") {
            stopPolling();
            loader.querySelector(".loader-text").innerText = "All images scored!";
            loader.classList.remove("hidden");
            previewDiv.classList.add("hidden");
            imgTag.classList.add("hidden");
            controls.classList.add("hidden");
        }
    } catch (e) {
        console.error(e);
    }
    finally {
        polling = false;
    }
}

function startPolling() {
    if (!poller) {
        poller = setInterval(pollStatus, 1000);
    }
}

function stopPolling() {
    clearInterval(poller);
    poller = null;
}

function updateStatusUI(s) {
    if (loadingImage) {
        if (s.state === "scanning") {
            loader.querySelector(".loader-text").innerText = `Scanning images… (${s.cached_total}/${s.total} found)`;
        }
        return;
    }

    if (s.state === "scanning") {
        loader.classList.remove("hidden");
        previewDiv.classList.add("hidden");
        imgTag.classList.add("hidden");
        controls.classList.add("hidden");
        document.getElementById("batch-area").classList.add("hidden");
        document.getElementById("compare-area").classList.add("hidden");
        loader.querySelector(".loader-text").innerText = `Scanning images… (${s.cached_total}/${s.total} found)`;
        counterDiv.innerText = "";
    } else if (s.state === "ready") {
        loader.classList.add("hidden");
        if (currentMode === "compare" && s.comparison) {
            // Comparison mode stats
            counterDiv.innerText = `${s.comparison.not_compared} not compared images (${s.comparison.scored}/${s.total} scored, ${s.comparison.fully_compared} fully compared)`;
        } else {
            // Regular scoring mode stats
            counterDiv.innerText = `${s.valid} unscored images (${s.cached_unscored}/${s.total} cached)`;
        }
    } else if (s.state === "done") {
        loader.classList.add("hidden");
        previewDiv.classList.add("hidden");
        imgTag.classList.add("hidden");
        controls.classList.add("hidden");
        document.getElementById("batch-area").classList.add("hidden");
        document.getElementById("compare-area").classList.add("hidden");
        counterDiv.innerText = "All images scored!";
    }
}

// ═══════════════════════════════════════════════════════════════════
// MODE SWITCHING
// ═══════════════════════════════════════════════════════════════════

function switchMode(mode) {
    currentMode = mode;
    updateModeButtons();

    // Hide all views
    const batchArea = document.getElementById("batch-area");
    const compareArea = document.getElementById("compare-area");
    loader.classList.add("hidden");
    previewDiv.classList.add("hidden");
    imgTag.classList.add("hidden");
    controls.classList.add("hidden");
    batchArea.classList.remove("visible");
    batchArea.classList.add("hidden");
    compareArea.classList.remove("visible");
    compareArea.classList.add("hidden");

    if (mode === "single") {
        previewDiv.classList.remove("hidden");
        controls.classList.remove("hidden");
        currentImage = null;
        loadNext();
    } else if (mode === "batch") {
        batchArea.classList.remove("hidden");
        batchArea.classList.add("visible");
        batchImages = [];
        batchScores = {};
        fetchBatch();
    } else if (mode === "compare") {
        compareArea.classList.remove("hidden");
        compareArea.classList.add("visible");
        fetchNextComparePair();
    }
}

function updateModeButtons() {
    document.getElementById("mode-single").classList.toggle("active", currentMode === "single");
    document.getElementById("mode-batch").classList.toggle("active", currentMode === "batch");
    document.getElementById("mode-compare").classList.toggle("active", currentMode === "compare");
}

// ═══════════════════════════════════════════════════════════════════
// SINGLE MODE
// ═══════════════════════════════════════════════════════════════════

async function loadNext() {
    if (loadingImage) return;
    loadingImage = true;
    stopPolling();

    try {
        const res = await fetch("/random_unscored");
        if (res.status === 204) {
            loader.querySelector(".loader-text").innerText = "Waiting for new images…";
            loader.classList.remove("hidden");
            previewDiv.classList.add("hidden");
            imgTag.classList.add("hidden");
            controls.classList.add("hidden");
            loadingImage = false;
            startPolling();
            return;
        }

        const data = await res.json();
        currentImage = data.image;

        imgTag.onload = null;
        imgTag.onerror = null;
        imgTag.src = "";
        imgTag.classList.add("hidden");

        loader.querySelector(".loader-text").innerText = "Loading image…";
        loader.classList.remove("hidden");
        previewDiv.classList.remove("hidden");
        controls.classList.add("hidden");

        imgTag.onload = () => {
            loader.classList.add("hidden");
            imgTag.classList.remove("hidden");
            controls.classList.remove("hidden");
            statusDiv.innerText = currentImage;
            loadingImage = false;
            startPolling();
        };

        imgTag.onerror = () => {
            console.error("Failed to load image:", currentImage);
            loader.querySelector(".loader-text").innerText = "Failed to load image…";
            loadingImage = false;
            startPolling();
        };

        imgTag.src = "/image/" + encodeURIComponent(currentImage.replace(/\\/g, "/"));
    } catch (e) {
        console.error(e);
        loader.querySelector(".loader-text").innerText = "Error loading image…";
        loadingImage = false;
        startPolling();
    }
}

function submitScore(score) {
    if (!currentImage) return;

    controls.classList.add("hidden");
    imgTag.classList.add("hidden");
    loader.classList.remove("hidden");

    fetch("/submit_score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: currentImage, score }),
    }).then(() => loadNext());
}

function skipImage() {
    loadNext();
}

function refreshCount() {
    location.reload();
}

// ═══════════════════════════════════════════════════════════════════
// BATCH MODE (30 images)
// ═══════════════════════════════════════════════════════════════════

async function fetchBatch() {
    loader.classList.remove("hidden");
    loader.querySelector(".loader-text").innerText = "Fetching 30 images…";

    try {
        const res = await fetch("/random_unscored_batch?n=30");
        if (res.status === 204) {
            loader.querySelector(".loader-text").innerText = "No unscored images available…";
            startPolling();
            return;
        }

        const data = await res.json();
        batchImages = data.images || [];
        batchScores = {};

        renderBatchGrid();
        loader.classList.add("hidden");
        startPolling();
    } catch (e) {
        console.error("Failed to fetch batch:", e);
        loader.querySelector(".loader-text").innerText = "Error fetching batch…";
        startPolling();
    }
}

function renderBatchGrid() {
    const grid = document.getElementById("batch-grid");
    grid.innerHTML = "";

    batchImages.forEach((img, idx) => {
        const card = document.createElement("div");
        card.className = "batch-card";

        const imgEl = document.createElement("img");
        imgEl.src = "/image/" + encodeURIComponent(img.replace(/\\/g, "/"));
        imgEl.alt = `Image ${idx + 1}`;

        const info = document.createElement("div");
        info.className = "batch-card-info";
        info.innerText = `${idx + 1}/${batchImages.length}`;

        const scores = document.createElement("div");
        scores.className = "batch-card-scores";

        for (let s = 1; s <= 5; s++) {
            const btn = document.createElement("button");
            btn.className = "batch-score-btn";
            btn.innerText = s;
            btn.onclick = () => selectBatchScore(img, s, btn);
            if (batchScores[img] === s) {
                btn.classList.add("selected");
            }
            scores.appendChild(btn);
        }

        card.appendChild(imgEl);
        card.appendChild(info);
        card.appendChild(scores);
        grid.appendChild(card);
    });

    updateBatchRemaining();
}

function selectBatchScore(img, score, buttonEl) {
    batchScores[img] = score;

    // Update button styling
    const allBtns = buttonEl.parentElement.querySelectorAll(".batch-score-btn");
    allBtns.forEach(btn => btn.classList.remove("selected"));
    buttonEl.classList.add("selected");

    updateBatchRemaining();
}

function updateBatchRemaining() {
    const scored = Object.keys(batchScores).length;
    const total = batchImages.length;
    document.getElementById("batch-remaining").innerText = `Scored: ${scored}/${total}`;
}

async function submitBatchScores() {
    if (Object.keys(batchScores).length === 0) {
        alert("Please score at least one image!");
        return;
    }

    loader.classList.remove("hidden");
    loader.querySelector(".loader-text").innerText = "Submitting scores…";

    const items = Object.entries(batchScores).map(([image, score]) => ({
        image,
        score,
    }));

    try {
        const res = await fetch("/submit_scores", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(items),
        }).then(() => fetchBatch());

        const result = await res.json();
        loader.querySelector(".loader-text").innerText = `Submitted! (${result.ok.length} OK, ${result.errors.length} errors)`;

        // setTimeout(() => {
        //     fetchBatch();
        // }, 1500);
    } catch (e) {
        console.error("Error submitting batch:", e);
        loader.querySelector(".loader-text").innerText = "Error submitting scores…";
    }
}

// ═══════════════════════════════════════════════════════════════════
// COMPARE MODE
// ═══════════════════════════════════════════════════════════════════

async function fetchNextComparePair() {
    loader.classList.remove("hidden");
    loader.querySelector(".loader-text").innerText = "Finding pair to compare…";




    const score = document.getElementById("manual_score").value;


    try {
        const params = new URLSearchParams({
            score: score
        });

        const res = await fetch(`/compare/next/?${params.toString()}`);
        if (res.status === 204) {
            loader.querySelector(".loader-text").innerText = "No more pairs to compare!";
            return;
        }

        const data = await res.json();
        compareLeftImage = data.left.image;
        compareRightImage = data.right.image;
        compareLeftData = data.left;
        compareRightData = data.right;

        renderComparePair();
        loader.classList.add("hidden");
    } catch (e) {
        console.error("Failed to fetch compare pair:", e);
        loader.querySelector(".loader-text").innerText = "Error fetching pair…";
    }
}

function renderComparePair() {
    const leftImg = document.getElementById("compare-left");
    const rightImg = document.getElementById("compare-right");

    leftImg.src = "/image/" + encodeURIComponent(compareLeftImage.replace(/\\/g, "/"));
    rightImg.src = "/image/" + encodeURIComponent(compareRightImage.replace(/\\/g, "/"));

    document.getElementById("left_file_id").innerText = compareLeftImage;
    document.getElementById("right_file_id").innerText = compareRightImage;

    document.getElementById("compare-left-score").innerText = compareLeftData.score || "-";
    document.getElementById("compare-left-count").innerText = compareLeftData.comparison_count || 0;

    document.getElementById("compare-right-score").innerText = compareRightData.score || "-";
    document.getElementById("compare-right-count").innerText = compareRightData.comparison_count || 0;
}

async function submitComparison(winner) {
    loader.classList.remove("hidden");
    loader.querySelector(".loader-text").innerText = "Submitting comparison…";

    try {
        const res = await fetch("/compare/submit", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                winner: winner,
                left_image: compareLeftImage,
                right_image: compareRightImage,
                left_data: compareLeftData,
                right_data: compareRightData,
            }),
        });

        const result = await res.json();
        if (!result.ok) {
            loader.querySelector(".loader-text").innerText = `Error: ${result.error}`;
            return;
        }

        loader.querySelector(".loader-text").innerText = "Updating…";
        setTimeout(() => {
            fetchNextComparePair();
        }, 500);
    } catch (e) {
        console.error("Error submitting comparison:", e);
        loader.querySelector(".loader-text").innerText = "Error submitting comparison…";
    }
}

function skipComparison() {
    fetchNextComparePair();
}

// ═══════════════════════════════════════════════════════════════════
// INITIALIZE
// ═══════════════════════════════════════════════════════════════════

startPolling();
updateModeButtons();
