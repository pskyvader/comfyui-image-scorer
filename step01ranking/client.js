let poller = null;
let currentImage = null;
let loadingImage = false;
let polling = false;

// DOM elements
const loader = document.getElementById("loader");
const previewDiv = document.getElementById("single-preview");
const imgTag = document.getElementById("preview");
const controls = document.getElementById("single-controls");

const counterDiv = document.getElementById("counter");
const statusDiv = document.getElementById("status");

// ───────────────────────────
// Polling / status
// ───────────────────────────

async function pollStatus() {
    if (loadingImage || polling) return; // skip while loading
    polling = true;
    

    try {
        const res = await fetch("/status");
        const s = await res.json();
        updateStatusUI(s);

        if (s.state === "ready") {
            const src = imgTag.src;
            if (!loadingImage && src.trim().length === 0) {
                loadNext();
            } else{
                if(s.cached===s.total){
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
        // pollStatus();
    }
}

function stopPolling() {
    clearInterval(poller);
    poller = null;
}

// ───────────────────────────
// Load / show image
// ───────────────────────────
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
        currentImage = data.image;//.replace(/\\/g, "/"); // normalize backslashes

        // Reset previous handlers & src, hide image
        imgTag.onload = null;
        imgTag.onerror = null;
        imgTag.src = "";
        imgTag.classList.add("hidden");

        // Show loader while image is loading
        loader.querySelector(".loader-text").innerText = "Loading image…";
        loader.classList.remove("hidden");
        previewDiv.classList.remove("hidden"); // container visible
        controls.classList.add("hidden");

        // Wait for image to fully load
        imgTag.onload = () => {
            loader.classList.add("hidden");
            imgTag.classList.remove("hidden"); // <--- crucial fix
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

        // Set src
        imgTag.src = "/image/" + encodeURIComponent(currentImage.replace(/\\/g, "/"));
    } catch (e) {
        console.error(e);
        loader.querySelector(".loader-text").innerText = "Error loading image…";
        loadingImage = false;
        startPolling();
    }
}

// ───────────────────────────
// Submit score
// ───────────────────────────
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

// ───────────────────────────
// Update loader/footer
// ───────────────────────────
function updateStatusUI(s) {
    if (loadingImage) {
        // still show scan progress while loading
        if (s.state === "scanning") {
            loader.querySelector(".loader-text").innerText = `Scanning images… (${s.cached}/${s.total} found)`;
        }
        return;
    }
    console.log("state", s.state);

    if (s.state === "scanning") {
        loader.classList.remove("hidden");
        previewDiv.classList.add("hidden");
        imgTag.classList.add("hidden");
        controls.classList.add("hidden");
        loader.querySelector(".loader-text").innerText = `Scanning images… (${s.cached}/${s.total} found)`;
        counterDiv.innerText = "";
    } else if (s.state === "ready") {
        loader.classList.add("hidden");
        counterDiv.innerText = `${s.valid} unscored images ready (${s.cached}/${s.total} cached)`;

    } else if (s.state === "done") {
        loader.classList.add("hidden");
        previewDiv.classList.add("hidden");
        imgTag.classList.add("hidden");
        controls.classList.add("hidden");
        counterDiv.innerText = "All images scored!";
    }
}

// ───────────────────────────
// Initialize
// ───────────────────────────
startPolling();
