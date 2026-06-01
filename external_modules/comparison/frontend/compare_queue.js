/**
 * Timeout-aware fetch utility + Queue/Prefetch managers for comparison mode.
 *
 * This file contains backend/network logic only — no DOM manipulation.
 * Loaded before compare.js which wires these managers to the UI.
 */

const compareQueueLogger = FrontendLogger.create("external_modules.comparison.frontend.compare_queue");

const DEFAULT_TIMEOUT_MS = 30000;

// ── Timeout-aware fetch ──────────────────────────────────────────────

/**
 * Wraps fetch with an AbortController-based timeout.
 * Rejects with an AbortError when the timeout fires.
 */
function fetchWithTimeout(url, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
    const controller = new AbortController();
    const existingSignal = options.signal;
    if (existingSignal) {
        existingSignal.addEventListener("abort", () => controller.abort());
    }
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    return fetch(url, { ...options, signal: controller.signal })
        .finally(() => clearTimeout(timer));
}

// ── CompareApi ───────────────────────────────────────────────────────

/**
 * Encapsulates all backend network calls for comparison mode.
 */
const CompareApi = {
    async getRankingConfig() {
        return api._get("/ranking/config");
    },
    async getStatus() {
        return api._get("/ranking/status");
    },
    async getNextPair(timeoutMs = DEFAULT_TIMEOUT_MS) {
        const resp = await fetchWithTimeout(
            `${api.apiBase}/ranking/next-pair`,
            {},
            timeoutMs
        );
        if (resp.status === 204) {
            return null;
        }
        if (!resp.ok) {
            throw new Error("Failed to get next pair");
        }
        return await resp.json();
    },
    async submitComparison(filenameA, filenameB, winner, timeoutMs = DEFAULT_TIMEOUT_MS) {
        const resp = await fetchWithTimeout(
            `${api.apiBase}/ranking/submit-comparison`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    filename_a: filenameA,
                    filename_b: filenameB,
                    winner: winner,
                }),
            },
            timeoutMs
        );
        if (!resp.ok) {
            throw new Error(`Server error ${resp.status}`);
        }
        return await resp.json();
    },
    async resetCache() {
        return api._post("/ranking/reset");
    },
    async getImage(filename) {
        const resp = await fetchWithTimeout(
            `${api.apiBase}/gallery/image/${encodeURIComponent(filename)}`
        );
        if (!resp.ok) {
            throw new Error("Failed to get image info");
        }
        return await resp.json();
    }
};

// ── QueueManager ─────────────────────────────────────────────────────

/**
 * Manages the two-stage submission pipeline:
 *   submissionQueue  →  (undo delay)  →  outgoingQueue  →  server
 *
 * Callbacks (set by consumer):
 *   onQueueChange()         – queue sizes changed
 *   onSubmissionComplete()  – a vote was accepted by the server
 *   onError(message)        – something failed
 *   onUndoUIUpdate()        – undo button state changed
 *   onDrained()             – both queues empty, safe to prefetch
 */
class QueueManager {
    constructor() {
        this.submissionQueue = [];
        this.outgoingQueue = [];
        this._processLoopRunning = false;
        this._UNDO_DELAY_MS = 10000;
        this._submissionCount = 0;
        this._parallelRequests = false;
        this._timeoutMs = DEFAULT_TIMEOUT_MS;

        // Callbacks — wired by consumer
        this.onQueueChange = () => {};
        this.onSubmissionComplete = () => {};
        this.onError = () => {};
        this.onUndoUIUpdate = () => {};
        this.onDrained = () => {};
    }

    get pendingCount() {
        return this.submissionQueue.length + this.outgoingQueue.length;
    }

    enqueue(entry) {
        this.submissionQueue.push({
            ...entry,
            queuedAt: Date.now(),
        });
        this.onQueueChange();
        this.onUndoUIUpdate();
        this._startProcessLoop();
    }

    undoLast() {
        if (this.submissionQueue.length === 0) {
            return false;
        }
        this.submissionQueue.pop();
        // Reset timer on the new last item so user gets another full window to undo it
        if (this.submissionQueue.length > 0) {
            this.submissionQueue[this.submissionQueue.length - 1].queuedAt = Date.now();
        }
        this.onQueueChange();
        this.onUndoUIUpdate();
        return true;
    }

    /** Returns the last submission queue entry (for undo UI preview). */
    peekLast() {
        return this.submissionQueue.length > 0
            ? this.submissionQueue[this.submissionQueue.length - 1]
            : null;
    }

    reset() {
        this.submissionQueue = [];
        this.outgoingQueue = [];
        this._processLoopRunning = false;
        this.onQueueChange();
        this.onUndoUIUpdate();
    }

    // ── internal ──

    _startProcessLoop() {
        if (this._processLoopRunning) {
            return;
        }
        this._processLoopRunning = true;
        this._processLoop();
    }

    async _processLoop() {
        while (this._processLoopRunning) {
            if (this.submissionQueue.length === 0) {
                this._processLoopRunning = false;
                this.onDrained();
                return;
            }

            const front = this.submissionQueue[0];
            const elapsed = Date.now() - front.queuedAt;
            const remaining = this._UNDO_DELAY_MS - elapsed;

            if (remaining <= 0) {
                this.submissionQueue.shift();
                this.outgoingQueue.push(front);
                this.onQueueChange();
                this.onUndoUIUpdate();

                const submitTask = this._executeSubmission(front);

                if (!this._parallelRequests) {
                    await submitTask;
                }
                // In parallel mode, fire-and-forget — loop continues immediately
            } else {
                await new Promise(r => setTimeout(r, remaining));
            }
        }
    }

    async _executeSubmission(entry) {
        try {
            await CompareApi.submitComparison(entry.filenameA, entry.filenameB, entry.winnerFilename, this._timeoutMs);
            this._submissionCount++;
            this.onSubmissionComplete();
        } catch (e) {
            const isTimeout = e.name === "AbortError";
            const reason = isTimeout ? "timed out" : e.message || String(e);
            compareQueueLogger.warn("Submission failed:", null, reason, entry);
            this.onError(`Submission dropped (${reason}): ${entry.filenameA} vs ${entry.filenameB}`);
        } finally {
            this.outgoingQueue = this.outgoingQueue.filter(x => x !== entry);
            this.onQueueChange();
            this.onUndoUIUpdate();
        }
    }
}

// ── PrefetchManager ──────────────────────────────────────────────────

/**
 * Maintains a cache of pre-fetched + pre-loaded image pairs so the UI
 * can display the next comparison instantly.
 *
 * Callbacks (set by consumer):
 *   onCacheChange()  – cache size changed
 */
class PrefetchManager {
    constructor() {
        this._cache = [];
        this._prefetching = false;
        this._targetSize = 0;
        this._parallelRequests = false;
        this._timeoutMs = DEFAULT_TIMEOUT_MS;

        // Callbacks
        this.onCacheChange = () => {};
    }

    get length() {
        return this._cache.length;
    }

    shift() {
        const item = this._cache.shift();
        this.onCacheChange();
        return item;
    }

    setTargetSize(size) {
        this._targetSize = size;
    }

    reset() {
        this._cache = [];
        this._prefetching = false;
        this.onCacheChange();
    }

    /** Returns true if the outgoing queue should block prefetch. Set by consumer. */
    isBlocked = null; // () => boolean

    start() {
        if (this._prefetching || this._cache.length >= this._targetSize) {
            compareQueueLogger.debug("Prefetch skipped", null, {
                prefetching: this._prefetching,
                cached: this._cache.length,
                target: this._targetSize,
            });
            return;
        }
        this._prefetching = true;
        this._loop();
    }

    stop() {
        this._prefetching = false;
    }

    async _loop() {
        compareQueueLogger.info("Prefetch loop started");

        while (this._prefetching && this._cache.length < this._targetSize) {
            // Wait if outgoing submissions are still in-flight
            if (this.isBlocked && this.isBlocked()) {
                await new Promise(r => setTimeout(r, 100));
                continue;
            }

            let pair;
            try {
                pair = await CompareApi.getNextPair(this._timeoutMs);
            } catch (e) {
                const isTimeout = e.name === "AbortError";
                const reason = isTimeout ? "timed out" : e.message || String(e);
                compareQueueLogger.warn("Prefetch fetch failed, dropping and retrying:", null, reason);
                // Drop this attempt and try the next one
                continue;
            }

            if (!pair) {
                break; // no more pairs available
            }

            // Preload images with timeout
            const leftUrl = `/images/${encodeURIComponent(pair.left.filename)}`;
            const rightUrl = `/images/${encodeURIComponent(pair.right.filename)}`;
            const leftImg = new Image();
            const rightImg = new Image();

            try {
                if (this._parallelRequests) {
                    // Parallel image loading
                    await Promise.all([
                        this._loadImageWithTimeout(leftImg, leftUrl),
                        this._loadImageWithTimeout(rightImg, rightUrl),
                    ]);
                } else {
                    // Sequential image loading
                    await this._loadImageWithTimeout(leftImg, leftUrl);
                    await this._loadImageWithTimeout(rightImg, rightUrl);
                }
            } catch (e) {
                compareQueueLogger.warn("Image preload failed, dropping pair:", null, pair.left.filename, pair.right.filename);
                // Drop this pair and try the next one
                continue;
            }

            this._cache.push({ pair, leftImg, rightImg });
            this.onCacheChange();
        }

        this._prefetching = false;
    }

    _loadImageWithTimeout(img, url) {
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                img.onload = null;
                img.onerror = null;
                img.src = "";
                reject(new DOMException("Image load timed out", "AbortError"));
            }, this._timeoutMs);

            img.onload = () => {
                clearTimeout(timer);
                resolve();
            };
            img.onerror = () => {
                clearTimeout(timer);
                reject(new Error("Image failed to load"));
            };
            img.src = url;
        });
    }
}
