/**
 * Comparison Mode — UI Orchestrator
 *
 * Depends on compare_queue.js (must be loaded first):
 *   - fetchWithTimeout, DEFAULT_TIMEOUT_MS
 *   - QueueManager
 *   - PrefetchManager
 */

const compareLogger = FrontendLogger.create("external_modules.comparison.frontend.compare");

class CompareMode {
    constructor() {
        this.currentPair = null;
        this._hasSubmitted = false;
        this._config = {};
        this.elements = {};

        // Sub-managers (from compare_queue.js)
        this._queue = new QueueManager();
        this._prefetch = new PrefetchManager();

        // Wire queue callbacks
        this._queue.onQueueChange = () => this._updateQueueIndicator();
        this._queue.onUndoUIUpdate = () => this._updateUndoUI();
        this._queue.onSubmissionComplete = () => {
            this._hasSubmitted = true;
            if (this._queue._submissionCount % (this._config.reserve_count || 1) === 0) {
                this.updateStats();
            }
        };
        this._queue.onError = (msg) => {
            Utils.showToast(msg, "error");
            this._hasSubmitted = true;
        };
        this._queue.onDrained = () => this._prefetch.start();

        // Wire prefetch callbacks
        this._prefetch.onCacheChange = () => this._updateQueueIndicator();
        this._prefetch.isBlocked = () => this._queue.outgoingQueue.length > 0;
    }

    // ── API methods moved to compare_queue.js ──────────────────────────────

    get _targetCacheSize() {
        return this._hasSubmitted ? this._config.reserve_count : 0;
    }

    // ── DOM caching ──────────────────────────────────────────────────

    cacheElements() {
        this.elements = {
            status: document.getElementById("comparison-status"),
            leftImg: document.getElementById("left-image"),
            rightImg: document.getElementById("right-image"),
            leftFilename: document.getElementById("left-filename"),
            rightFilename: document.getElementById("right-filename"),
            leftScore: document.getElementById("left-score"),
            rightScore: document.getElementById("right-score"),
            leftChain: document.getElementById("left-chain"),
            rightChain: document.getElementById("right-chain"),
            leftComp: document.getElementById("left-compsize"),
            rightComp: document.getElementById("right-compsize"),
            leftComparisons: document.getElementById("left-comparisons"),
            rightComparisons: document.getElementById("right-comparisons"),
            leftButton: document.getElementById("left-button"),
            rightButton: document.getElementById("right-button"),
            skipButton: document.getElementById("skip-button"),
            statsTotal: document.getElementById("stat-total"),
            statsRanked: document.getElementById("stat-ranked"),
            statsComparisons: document.getElementById("stat-comparisons"),
            statsComponents: document.getElementById("stat-components"),
            statsChains: document.getElementById("stat-chains"),

            statsLevel: document.getElementById("stat-level"),
            statsActive: document.getElementById("stat-active"),
            statsNextLevel: document.getElementById("stat-next-level"),
            statsNextLevelNum: document.getElementById("stat-next-level-num"),
            statsBaseLevel: document.getElementById("stat-base-level"),
            loadingOverlay: document.getElementById("loading-overlay"),
            loadingOverlayText: document.getElementById("loading-overlay-text"),
            debugBtn: document.getElementById("toggle-debug"),
            debugPanel: document.getElementById("debug-panel"),
            debugContent: document.getElementById("debug-content"),
            leftCard: document.querySelector(".left-card"),
            rightCard: document.querySelector(".right-card"),
            collapsibleIndicator: document.getElementById("collapsible-indicator"),
            resetCacheButton: document.getElementById("reset-cache-button"),
            queueIndicator: this._createQueueIndicator(),
            undoButton: this._createUndoButton(),
        };
    }

    // ── UI element creation ──────────────────────────────────────────

    _createQueueIndicator() {
        const el = document.createElement("div");
        el.id = "queue-indicator";
        el.style.cssText = `
            position: fixed;
            bottom: 12px;
            right: 12px;
            background: rgba(0,0,0,0.7);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 6px 8px;
            font-size: 10px;
            font-family: monospace;
            color: #a0a0a0;
            z-index: 9999;
            pointer-events: none;
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 2px;
            transition: opacity 0.2s;
            opacity: 0;
        `;
        el.innerHTML = `
            <span id="qi-queue" title="Votes pending submission">⬆ 0</span>
            <span id="qi-pending" title="Votes submitting">⬆ 0</span>
            <span id="qi-ready" title="Comparisons fully loaded (images ready)">⬇ 0</span>
        `;
        document.body.appendChild(el);
        return el;
    }

    _createUndoButton() {
        const el = document.createElement("div");
        el.id = "undo-button";
        el.style.cssText = `
            position: fixed;
            right: 8px;
            top: 50%;
            transform: translateY(-50%);
            z-index: 9999;
            display: none;
            cursor: pointer;
            background: rgba(0,0,0,0.55);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(240,160,80,0.3);
            border-radius: 12px;
            padding: 6px;
            font-size: 11px;
            font-family: system-ui, sans-serif;
            color: #f0a050;
            font-weight: 700;
            transition: opacity 0.2s, background 0.2s;
            overflow: hidden;
            width: 82px;
            text-align: center;
        `;
        el.innerHTML = `
            <div id="undo-thumbs" style="display:flex;flex-direction:column;align-items:center;gap:4px;margin-bottom:4px;"></div>
            <div style="display:flex;flex-direction:column;align-items:center;gap:2px;position:relative;z-index:1;">
                <span style="font-size:10px;">↩ UNDO</span>
                <span id="undo-timer" style="font-size:9px;color:#888;"></span>
            </div>
            <div id="undo-progress" style="position:absolute;left:0;bottom:0;height:3px;width:0%;background:#f0a050;border-radius:0 0 12px 12px;transition:none;"></div>
        `;
        el.onclick = () => this._undoLast();
        document.body.appendChild(el);
        this._undoProgressEl = el.querySelector("#undo-progress");
        this._undoTimerEl = el.querySelector("#undo-timer");
        this._undoThumbsEl = el.querySelector("#undo-thumbs");
        return el;
    }

    // ── UI updates ───────────────────────────────────────────────────

    _updateQueueIndicator() {
        const el = this.elements.queueIndicator;
        if (!el) {
            return;
        }
        const pending = this._queue.outgoingQueue.length;
        const queue = this._queue.submissionQueue.length;
        const ready = this._prefetch.length;

        el.querySelector("#qi-queue").textContent = `🔄 ${queue}`;
        el.querySelector("#qi-pending").textContent = `⬆ ${pending}`;
        el.querySelector("#qi-ready").textContent = `⬇ ${ready}`;

        el.style.opacity = (pending + ready) > 0 ? "1" : "0.5";
    }

    _updateUndoUI() {
        const btn = this.elements.undoButton;
        if (!btn) {
            return;
        }

        if (this._queue.submissionQueue.length === 0) {
            btn.style.display = "none";
            return;
        }

        btn.style.display = "block";
        compareLogger.info("Updating undo UI for submission queue:", null, this._queue.submissionQueue);

        const last = this._queue.peekLast();
        const isLeftWinner = last.winnerFilename === last.filenameA;
        const imgStyle = "width:64px;height:64px;object-fit:cover;border-radius:6px;";
        const winnerBorder = "border:2px solid #10b981;box-shadow:0 0 8px rgba(16,185,129,0.5);";
        const loserBorder = "border:1px solid rgba(255,255,255,0.1);opacity:0.5;";
        const leftStyle = imgStyle + (isLeftWinner ? winnerBorder : loserBorder);
        const rightStyle = imgStyle + (!isLeftWinner ? winnerBorder : loserBorder);
        this._undoThumbsEl.innerHTML = `
            <img src="${last.leftSrc}" style="${leftStyle}" title="${last.filenameA}" />
            <img src="${last.rightSrc}" style="${rightStyle}" title="${last.filenameB}" />
        `;

        // Animate progress bar and timer for the last item
        const start = last.queuedAt;
        const undoDelay = this._queue._UNDO_DELAY_MS;
        const tick = () => {
            if (this._queue.submissionQueue.length === 0 || this._queue.peekLast() !== last) {
                return;
            }
            const elapsed = Date.now() - start;
            const pct = Math.min(elapsed / undoDelay, 1);
            this._undoProgressEl.style.width = `${pct * 100}%`;
            const secs = Math.max(0, Math.ceil((undoDelay - elapsed) / 1000));
            this._undoTimerEl.textContent = `${secs}s`;
            if (pct < 1) {
                requestAnimationFrame(tick);
            }
        };
        requestAnimationFrame(tick);
    }

    _showUndoButton() {
        this._updateUndoUI();
    }

    _undoLast() {
        if (this._queue.undoLast()) {
            Utils.showToast("Vote undone", "info");
        }
    }

    showLoading(show, message) {
        const overlay = this.elements.loadingOverlay;
        if (!overlay) {
            return;
        }
        if (message && this.elements.loadingOverlayText) {
            this.elements.loadingOverlayText.textContent = message;
        }

        if (show) {
            overlay.style.display = "flex";
            overlay.style.opacity = "1";
            overlay.style.pointerEvents = "auto";
        } else {
            overlay.style.opacity = "0";
            overlay.style.pointerEvents = "none";
            setTimeout(() => {
                if (overlay.style.opacity === "0") {
                    overlay.style.display = "none";
                }
            }, 300);
        }
    }

    updateStatus(msg) {
        if (this.elements.status) {
            this.elements.status.textContent = msg;
        }
    }

    _populateLegend() {
        const fmt = n => n.toLocaleString();
        for (const [id, val] of Object.entries({
            "legend-seed-size": fmt(this._config.seed_size),
            "legend-seed-target": fmt(this._config.seed_target_comparisons),
            "legend-insertion-target": fmt(this._config.insertion_target_comparisons),
        })) {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = val;
            }
        }
    }

    // ── Event listeners ──────────────────────────────────────────────

    attachEventListeners() {
        const { leftButton, rightButton, skipButton, debugBtn, leftCard, rightCard, resetCacheButton } = this.elements;

        leftButton.onclick = () => this.submitVote("left");
        rightButton.onclick = () => this.submitVote("right");

        if (leftCard) {
            leftCard.onclick = (e) => {
                if (e.target !== leftButton) {
                    this.submitVote("left");
                }
            };
        }
        if (rightCard) {
            rightCard.onclick = (e) => {
                if (e.target !== rightButton) {
                    this.submitVote("right");
                }
            };
        }

        if (skipButton) {
            skipButton.onclick = () => {
                this._hasSubmitted = true;
                const pair = this.currentPair;
                if (pair && pair.left && pair.right) {
                    // Put one of the two images into the exclude cache so this
                    // pair cannot be formed again.
                    CompareApi.skipImage(pair.left.filename);
                }
                this.loadPair();
            };
        }
        if (debugBtn) {
            debugBtn.onclick = () => this.elements.debugPanel.classList.toggle("hidden");
        }
        if (resetCacheButton) {
            resetCacheButton.onclick = () => this.resetCache();
        }
    }

    // ── Initialization ───────────────────────────────────────────────

    async init(params) {
        compareLogger.info("Initializing CompareMode...");
        this.cacheElements();
        this.attachEventListeners();

        // Fetch config and phases in parallel
        const [serverConfig, phases] = await Promise.all([
            CompareApi.getRankingConfig(),
            CompareApi.getPhases().catch(e => {
                compareLogger.warn("Failed to load phases:", null, e);
                return null;
            }),
        ]);

        this._config.reserve_count = serverConfig.reserve_count;
        this._config.parallel_requests = serverConfig.parallel_requests;
        this._config.timeout_ms = serverConfig.timeout_ms || 30000;
        this._config.seed_size = serverConfig.seed_size;
        this._config.seed_target_comparisons = serverConfig.seed_target_comparisons;
        this._config.insertion_target_comparisons = serverConfig.insertion_target_comparisons;
        compareLogger.info("Ranking config loaded:", null, this._config);

        // Push phase metadata into CompareView with config values for descriptions
        if (phases) {
            CompareView.setPhases(phases);
            CompareView.renderDescriptions({
                seed_size: serverConfig.seed_size,
                seed_target: serverConfig.seed_target_comparisons,
                insertion_target: serverConfig.insertion_target_comparisons,
            });
        }

        // Configure sub-managers from server config
        this._queue._parallelRequests = !!this._config.parallel_requests;
        this._queue._timeoutMs = this._config.timeout_ms;

        this._prefetch._parallelRequests = !!this._config.parallel_requests;
        this._prefetch._timeoutMs = this._config.timeout_ms;

        const leftFile = params?.get("left");
        const rightFile = params?.get("right");

        if (leftFile && rightFile) {
            this.updateStatus("Loading requested pair...");
            const [leftData, rightData] = await Promise.all([
                CompareApi.getImage(leftFile),
                CompareApi.getImage(rightFile),
            ]);
            const norm = (d, fn) => ({
                filename: d.filename || fn,
                score: d.score ?? 0.5,
                rating_mu: d.rating_mu ?? 25.0,
                rating_sigma: d.rating_sigma ?? 25.0 / 3.0,
                comparison_count: d.comparison_count ?? 0,
                chain_length: d.chain_length ?? 0,
                component_size: d.component_size ?? 0,
                component_id: d.component_id ?? null,
                is_top: !!d.is_top,
                is_bottom: !!d.is_bottom,
                is_seed: false,
                _extremes: d._extremes ?? { top: 0, bottom: 0 },
            });
            const left = norm(leftData, leftFile);
            const right = norm(rightData, rightFile);
            const pair = {
                phase: null,
                total_images: 0,
                total_comparisons: 0,
                level: {},
                base_level: 0,
                collapsable: false,
                same_component: false,
                seed_percentage: this._config.seed_size ? 0 : 0,
                seed_target_comparisons: this._config.seed_target_comparisons || 0,
                insertion_target_comparisons: this._config.insertion_target_comparisons || 0,
                reserve_count: this._config.reserve_count || 1,
                probability_a_beats_b: 0.5,
            };
            this.currentPair = { left, right, pair };
            await this.renderPair();
        } else {
            await this.loadPair();
        }
    }

    // ── Pair loading ─────────────────────────────────────────────────

    async loadPair() {
        this.showLoading(true);
        this.updateStatus(this._prefetch.length > 0 ? "Loading next pair..." : "Finding next pair...");

        let cached;
        let pair;
        if (this._prefetch.length > 0) {
            cached = this._prefetch.shift();
            pair = cached.pair;
        } else {
            try {
                pair = await CompareApi.getNextPair(this._prefetch._timeoutMs);
            } catch (e) {
                const isTimeout = e.name === "AbortError";
                const reason = isTimeout ? "timed out" : e.message || String(e);
                compareLogger.warn("loadPair fetch failed:", null, reason);
                Utils.showToast(`Failed to load pair (${reason}), retrying...`, "error");
                // Drop this attempt and try the next one
                this.showLoading(false);
                setTimeout(() => this.loadPair(), 1000);
                return;
            }

            if (pair) {
                const leftImg = new Image();
                leftImg.src = `/images/${encodeURIComponent(pair.left.filename)}`;
                const rightImg = new Image();
                rightImg.src = `/images/${encodeURIComponent(pair.right.filename)}`;
                cached = { pair, leftImg, rightImg };
            }
        }

        if (!pair) {
            this.updateStatus("No more pairs to compare!");
            this.elements.leftButton.disabled = true;
            this.elements.rightButton.disabled = true;
            this.showLoading(false);
            return;
        }
        compareLogger.info("Loaded pair:", null, pair, { cached: cached !== null });
        compareLogger.info("Loaded pair top-level keys:", null, pair ? Object.keys(pair) : null);
        if (pair && !pair.pair) {
            compareLogger.error("Backend response missing `pair` envelope; has keys:", null, Object.keys(pair));
        }
        this.currentPair = { ...pair, _preloadedLeft: cached?.leftImg, _preloadedRight: cached?.rightImg };
        await this.renderPair();
        this.showLoading(false);

        this._prefetch.setTargetSize(this._targetCacheSize);
        this._prefetch.start();
    }

    isSamePair(pairA, pairB) {
        if (!pairA || !pairB || !pairA.left || !pairA.right || !pairB.left || !pairB.right) {
            return false;
        }

        const a = [pairA.left.filename, pairA.right.filename].sort();
        const b = [pairB.left.filename, pairB.right.filename].sort();
        return a[0] === b[0] && a[1] === b[1];
    }

    // ── Rendering ────────────────────────────────────────────────────

    async renderPair() {
        const { left, right, pair } = this.currentPair;
        const els = this.elements;

        els.leftImg.classList.add("opacity-0");
        els.rightImg.classList.add("opacity-0");

        // Cancel any pending image loads to prevent memory leaks
        if (this._preloadLeft) {
            this._preloadLeft.onload = null;
            this._preloadLeft.onerror = null;
            this._preloadLeft = null;
        }
        if (this._preloadRight) {
            this._preloadRight.onload = null;
            this._preloadRight.onerror = null;
            this._preloadRight = null;
        }

        els.leftFilename.textContent = left.filename;
        els.rightFilename.textContent = right.filename;

        // Delegate all phase-driven detail, footer, phase cards, and debug to CompareView.
        CompareView.render(pair, left, right, els);

        els.leftImg.src = `/images/${encodeURIComponent(left.filename)}`;
        els.rightImg.src = `/images/${encodeURIComponent(right.filename)}`;

        // Use preloaded images if available (instant, no await needed)
        if (this.currentPair._preloadedLeft) {
            els.leftImg.src = this.currentPair._preloadedLeft.src;
            els.leftImg.classList.remove("opacity-0");
            els.leftImg.classList.add("opacity-100");
        } else {
            els.leftImg.src = `/images/${encodeURIComponent(left.filename)}`;
            await new Promise((resolve) => {
                els.leftImg.onload = resolve; els.leftImg.onerror = resolve;
            });
            els.leftImg.classList.remove("opacity-0");
            els.leftImg.classList.add("opacity-100");
        }

        if (this.currentPair._preloadedRight) {
            els.rightImg.src = this.currentPair._preloadedRight.src;
            els.rightImg.classList.remove("opacity-0");
            els.rightImg.classList.add("opacity-100");
        } else {
            els.rightImg.src = `/images/${encodeURIComponent(right.filename)}`;
            await new Promise((resolve) => {
                els.rightImg.onload = resolve; els.rightImg.onerror = resolve;
            });
            els.rightImg.classList.remove("opacity-0");
            els.rightImg.classList.add("opacity-100");
        }
    }

    // ── Vote submission ──────────────────────────────────────────────

    async submitVote(winner) {
        compareLogger.info("Submitting vote for:", null, winner, this.currentPair);
        const winnerFilename = winner === "left" ? this.currentPair.left.filename : this.currentPair.right.filename;
        const filenameA = this.currentPair.left.filename;
        const filenameB = this.currentPair.right.filename;

        const leftSrc = this.currentPair._preloadedLeft?.src;
        const rightSrc = this.currentPair._preloadedRight?.src;

        this._queue.enqueue({ filenameA, filenameB, winnerFilename, leftSrc, rightSrc });
        this._showUndoButton();

        // Mark as submitted immediately so cache filling starts right away
        this._hasSubmitted = true;

        // Immediately advance to next pair without waiting
        this.loadPair();
    }

    // ── Reset / Stats ────────────────────────────────────────────────

    async resetCache() {
        this.showLoading(true, "Resetting cache...");
        try {
            await CompareApi.resetCache();
            this._prefetch.reset();
            this._queue.reset();
            Utils.showToast("Cache cleared!", "success");
        } finally {
            this.showLoading(false);
        }
    }

    async updateStats() {
        const status = await CompareApi.getStatus();
        const els = this.elements;
        if (els.statsTotal) {
            els.statsTotal.textContent = status.total_images;
        }
        if (els.statsRanked) {
            els.statsRanked.textContent = status.ranked_images;
        }
        if (els.statsComparisons) {
            els.statsComparisons.textContent = status.total_comparisons;
        }
        if (els.statsComponents) {
            els.statsComponents.textContent = status.total_components;
        }
        if (els.statsChains) {
            els.statsChains.textContent = status.total_chains;
        }
        if (els.statsLevel) {
            els.statsLevel.textContent = status.base_level;
        }
        if (els.statsActive) {
            els.statsActive.textContent = status.active_nodes;
        }
        if (els.statsNextLevel) {
            els.statsNextLevel.textContent = status.next_level_count;
        }
        if (els.statsNextLevelNum) {
            els.statsNextLevelNum.textContent = status.current_target;
        }
        if (els.statsBaseLevel) {
            els.statsBaseLevel.textContent = status.base_level;
        }
    }
}

window.Sections = window.Sections || {};
window.Sections.compare = CompareMode;
