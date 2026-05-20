/**
 * Comparison Mode Logic
 */

class CompareMode {
    constructor() {
        this.currentPair = null;
        this._submitting = false;
        this._hasSubmitted = false;
        this._prefetching = false;
        this._pairCache = [];
        this._submissionQueue = [];
        this._outgoingQueue = [];
        this._processingQueue = false;
        this._processLoopRunning = false;
        this._UNDO_DELAY_MS = 10000;

        // These will be populated in init()
        this.elements = {};
    }

    get _targetCacheSize() {
        return this._hasSubmitted ? 10 : 0;
    }

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
            statsSkipped: document.getElementById("stat-skipped"),
            statsLevel: document.getElementById("stat-level"),
            statsActive: document.getElementById("stat-active"),
            statsNextLevel: document.getElementById("stat-next-level"),
            statsNextLevelNum: document.getElementById("stat-next-level-num"),
            statsBaseLevel: document.getElementById("stat-base-level"),
            loadingOverlay: document.getElementById("loading-overlay"),
            debugBtn: document.getElementById("toggle-debug"),
            debugPanel: document.getElementById("debug-panel"),
            debugContent: document.getElementById("debug-content"),
            graphIndicator: document.getElementById("graph-indicator"),
            graphComponentId: document.getElementById("graph-component-id"),
            graphCount: document.getElementById("graph-count"),
            leftCard: document.querySelector(".left-card"),
            rightCard: document.querySelector(".right-card"),
            collapsibleIndicator: document.getElementById("collapsible-indicator"),
            resetCacheButton: document.getElementById("reset-cache-button"),
            queueIndicator: this._createQueueIndicator(),
            undoButton: this._createUndoButton(),
        };
    }

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
            padding: 6px 10px;
            font-size: 10px;
            font-family: monospace;
            color: #a0a0a0;
            z-index: 9999;
            pointer-events: none;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: opacity 0.2s;
            opacity: 0;
        `;
        el.innerHTML = `
            <span id="qi-pending" title="Votes pending submission">⬆ 0</span>
            <span style="opacity:0.3">|</span>
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
            bottom: 56px;
            right: 12px;
            z-index: 9999;
            display: none;
            cursor: pointer;
            background: rgba(0,0,0,0.55);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(240,160,80,0.3);
            border-radius: 12px;
            padding: 8px;
            font-size: 13px;
            font-family: system-ui, sans-serif;
            color: #f0a050;
            font-weight: 700;
            transition: opacity 0.2s, background 0.2s;
            overflow: hidden;
            min-width: 180px;
            max-width: 220px;
        `;
        el.innerHTML = `
            <div id="undo-thumbs" style="display:flex;justify-content:center;gap:8px;margin-bottom:6px;"></div>
            <div style="display:flex;justify-content:center;align-items:center;gap:6px;position:relative;z-index:1;">
                <span>↩ UNDO</span>
                <span id="undo-timer" style="font-size:10px;color:#888;"></span>
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

    _updateQueueIndicator() {
        const el = this.elements.queueIndicator;
        if (!el) {
            return;
        }
        const pending = this._submissionQueue.length + this._outgoingQueue.length;
        const ready = this._pairCache.length;

        el.querySelector("#qi-pending").textContent = `⬆ ${pending}`;
        el.querySelector("#qi-ready").textContent = `⬇ ${ready}`;

        el.style.opacity = (pending + ready) > 0 ? "1" : "0.5";
    }

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
            skipButton.onclick = () => this.loadPair();
        }
        if (debugBtn) {
            debugBtn.onclick = () => this.elements.debugPanel.classList.toggle("hidden");
        }
        if (resetCacheButton) {
            resetCacheButton.onclick = () => this.resetCache();
        }
    }

    async init(params) {
        console.log("Initializing CompareMode...");
        this.cacheElements();
        this.attachEventListeners();

        const leftFile = params?.get("left");
        const rightFile = params?.get("right");

        if (leftFile && rightFile) {
            this.updateStatus("Loading requested pair...");
            try {
                const [leftData, rightData] = await Promise.all([
                    api.getImage(leftFile),
                    api.getImage(rightFile),
                ]);
                this.currentPair = {
                    left: leftData,
                    right: rightData,
                    rationale: {
                        strategy: "Manual Selection (Chain Map)",
                        score_diff: Math.abs(leftData.score - rightData.score)
                            .toFixed(4),
                        seed: leftFile,
                    },
                };
                await this.renderPair();
            } catch (e) {
                console.error("Manual load failed:", e);
                await this.loadPair();
            }
        } else {
            await this.loadPair();
        }
    }

    async loadPair() {
        this.showLoading(true);
        this.updateStatus(this._pairCache.length > 0 ? "Loading next pair..." : "Finding next pair...");

        try {
            let cached;
            let pair;
            if (this._pairCache.length > 0) {
                cached = this._pairCache.shift();
                pair = cached.pair;
            } else {
                pair = await api.getNextPair();
            }

            if (!pair) {
                this.updateStatus("No more pairs to compare!");
                this.elements.leftButton.disabled = true;
                this.elements.rightButton.disabled = true;
                this.showLoading(false);
                return;
            }
            this.currentPair = { ...pair, _preloadedLeft: cached?.leftImg, _preloadedRight: cached?.rightImg };
            await this.renderPair();
            this.showLoading(false);

            this._startPrefetch();
        } catch (e) {
            this.updateStatus(`Error: ${e.message}`);
            Utils.showToast(e.message, "error");
            this.showLoading(false);
        }
    }

    isSamePair(pairA, pairB) {
        if (!pairA || !pairB || !pairA.left || !pairA.right || !pairB.left || !pairB.right) {
            return false;
        }

        const a = [pairA.left.filename, pairA.right.filename].sort();
        const b = [pairB.left.filename, pairB.right.filename].sort();
        return a[0] === b[0] && a[1] === b[1];
    }

    async renderPair() {
        const { left, right, pair_meta, collapsable } = this.currentPair;
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

        // Update footer stats
        if (this.currentPair.global_stats) {
            const gs = this.currentPair.global_stats;
            const totalEl = document.getElementById("stat-total");
            const rankedEl = document.getElementById("stat-ranked");
            const levelEl = document.getElementById("stat-level");
            const compsEl = document.getElementById("stat-comparisons");
            const compsStatEl = document.getElementById("stat-components");
            const chainsEl = document.getElementById("stat-chains");
            const skippedEl = document.getElementById("stat-skipped");
            const activeEl = document.getElementById("stat-active");
            const nextLevelEl = document.getElementById("stat-next-level");
            const nextLevelNumEl = document.getElementById("stat-next-level-num");
            const baseLevelEl = document.getElementById("stat-base-level");

            if (totalEl) {
                totalEl.textContent = gs.total_images.toLocaleString();
            }
            if (rankedEl) {
                rankedEl.textContent = (gs.level_count ?? 0).toLocaleString();
            }
            if (levelEl) {
                levelEl.textContent = gs.target_level;
            }
            if (compsEl) {
                compsEl.textContent = gs.total_comparisons.toLocaleString();
            }
            if (compsStatEl) {
                compsStatEl.textContent = (gs.total_components ?? 0).toLocaleString();
            }
            if (chainsEl) {
                chainsEl.textContent = (gs.total_chains ?? 0).toLocaleString();
            }
            if (skippedEl) {
                skippedEl.textContent = (gs.skipped_comparisons ?? 0).toLocaleString();
            }
            if (activeEl) {
                activeEl.textContent = (gs.active_nodes ?? 0).toLocaleString();
            }
            if (nextLevelEl) {
                nextLevelEl.textContent = (gs.next_level_count ?? 0).toLocaleString();
            }
            if (nextLevelNumEl) {
                nextLevelNumEl.textContent = gs.target_level + 1;
            }
            if (baseLevelEl) {
                baseLevelEl.textContent = gs.base_level ?? (gs.target_level - 1);
            }
        }

        // Single line stats separated by |
        const statsText = `Score: ${Utils.formatScore(left.score)} | Chain: ${left.chain_length ?? "-"} | Component: ${left.component_size ?? "-"} | Comparisons: ${left.comparison_count ?? 0}`;
        const rightStatsText = `Score: ${Utils.formatScore(right.score)} | Chain: ${right.chain_length ?? "-"} | Component: ${right.component_size ?? "-"} | Comparisons: ${right.comparison_count ?? 0}`;

        const leftStatsEl = document.getElementById("left-stats-line");
        const rightStatsEl = document.getElementById("right-stats-line");
        if (leftStatsEl) {
            leftStatsEl.textContent = statsText;
        }
        if (rightStatsEl) {
            rightStatsEl.textContent = rightStatsText;
        }

        // Update borders based on comparison type
        els.leftCard.classList.remove("collapsible-card", "regular-card", "extra-card");
        els.rightCard.classList.remove("collapsible-card", "regular-card", "extra-card");

        const isExtra = left.component_id === right.component_id
            && (this.currentPair.debug?.left_extremes?.top === 1 && this.currentPair.debug?.left_extremes?.bottom === 1);

        if (collapsable) {
            els.leftCard.classList.add("collapsible-card");
            els.rightCard.classList.add("collapsible-card");
        } else if (isExtra) {
            els.leftCard.classList.add("extra-card");
            els.rightCard.classList.add("extra-card");
        } else {
            els.leftCard.classList.add("regular-card");
            els.rightCard.classList.add("regular-card");
        }

        // Debug/Pair Meta Rationale - Enhanced Grid Alignment
        const rationale = pair_meta || this.currentPair.rationale;
        if (els.debugContent && rationale) {
            els.debugContent.innerHTML = `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8 w-full items-start">
                    <table class="w-full text-left border-collapse">
                        <thead><tr><th colspan="2" class="text-purple-400 border-b border-purple-500/20 pb-2 mb-2 uppercase tracking-widest text-[9px]">Selection Logic</th></tr></thead>
                        <tbody class="text-[10px]">
                            <tr class="h-6">
                                <td class="text-gray-500 pr-4">Strategy</td>
                                <td class="text-white font-bold">${rationale.pair_type || "Standard"}</td>
                            </tr>
                            <tr class="h-6">
                                <td class="text-gray-500 pr-4">Chain Level</td>
                                <td class="text-white">${(rationale.chain_level === -1 || rationale.chain_level === undefined) ? "Auto" : rationale.chain_level}</td>
                            </tr>
                            <tr class="h-6">
                                <td class="text-gray-500 pr-4">Comp Size</td>
                                <td class="text-white">${rationale.left_component_size || "-"} vs ${rationale.right_component_size || "-"}</td>
                            </tr>
                        </tbody>
                    </table>
                    <table class="w-full text-left border-collapse">
                        <thead>
                            <tr>
                                <th class="text-purple-400 border-b border-purple-500/20 pb-2 mb-2 uppercase tracking-widest text-[9px]">Node Metrics</th>
                                <th class="text-[8px] text-gray-500 text-right uppercase tracking-tighter">Left</th>
                                <th class="text-[8px] text-gray-500 text-right uppercase tracking-tighter">Right</th>
                            </tr>
                        </thead>
                        <tbody class="text-[10px]">
                            <tr class="h-6 border-b border-white/5">
                                <td class="text-gray-500 pr-4">Score</td>
                                <td class="text-white text-right font-mono">${Utils.formatScore(left.score)}</td>
                                <td class="text-white text-right font-mono">${Utils.formatScore(right.score)}</td>
                            </tr>
                            <tr class="h-6 border-b border-white/5">
                                <td class="text-gray-500 pr-4">Chain Depth</td>
                                <td class="text-white text-right">${left.chain_length || 0}</td>
                                <td class="text-white text-right">${right.chain_length || 0}</td>
                            </tr>
                            <tr class="h-6 border-b border-white/5">
                                <td class="text-gray-500 pr-4">Comparisons</td>
                                <td class="text-white text-right">${left.comparison_count || 0}</td>
                                <td class="text-white text-right">${right.comparison_count || 0}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            `;

            if (rationale.refinement_details) {
                const rd = rationale.refinement_details;

                // Extract filenames from node objects, handling both old string format and new {filename, score} format
                const getFilename = node => typeof node === "string" ? node : node.filename;

                // Explicit cross-side matching for reliability
                const leftSub = rd.node_0.sub_chain_nodes || [];
                const rightSub = rd.node_1.sub_chain_nodes || [];
                const leftFull = rd.node_0.full_chain_nodes || [];
                const rightFull = rd.node_1.full_chain_nodes || [];

                const leftSubFilenames = new Set(leftSub.map(getFilename));
                const rightSubFilenames = new Set(rightSub.map(getFilename));
                const leftFullFilenames = new Set(leftFull.map(getFilename));
                const rightFullFilenames = new Set(rightFull.map(getFilename));

                const isFilenameInBothSubs = fn => leftSubFilenames.has(fn) && rightSubFilenames.has(fn);
                const isFilenameInBothFulls = fn => leftFullFilenames.has(fn) && rightFullFilenames.has(fn);

                // Helper to get consistent styling for nodes
                const getNodeStyle = (filename, currentActive, otherActive) => {
                    // 1. Priority: Active Node on THIS side
                    if (filename === currentActive) {
                        return { bullet: "bg-purple-500 shadow-[0_0_10px_rgba(168,85,247,0.6)]", text: "text-purple-400 font-black bg-purple-500/20 px-1 rounded ring-1 ring-purple-500/30", isInline: false };
                    }

                    // 2. Secondary: Active Node of the OTHER side (if it exists in our list)
                    if (filename === otherActive) {
                        return { bullet: "bg-pink-500 animate-pulse", text: "text-pink-400 font-bold bg-pink-500/20 px-1 rounded italic", isInline: false };
                    }

                    // 3. Shared in the specific GAP/Subchain (Blue shades)
                    if (isFilenameInBothSubs(filename) || rd.common_subchain_nodes?.includes(filename)) {
                        const hash = [...filename].reduce((a, c) => a + c.charCodeAt(0), 0);
                        const hue = 190 + (hash % 50);
                        return {
                            bullet: `background-color: hsl(${hue}, 80%, 50%); box-shadow: 0 0 8px hsla(${hue}, 80%, 50%, 0.5)`,
                            text: `color: hsl(${hue}, 80%, 75%); background-color: hsla(${hue}, 80%, 50%, 0.2); padding: 0 4px; border-radius: 3px; font-weight: 600;`,
                            isInline: true,
                            rawColor: `hsl(${hue}, 80%, 50%)`,
                        };
                    }

                    // 4. Shared in the FULL chain (Red waypoints)
                    if (isFilenameInBothFulls(filename) || rd.common_nodes?.includes(filename)) {
                        return { bullet: "bg-red-500 shadow-[0_0_5px_rgba(239,68,68,0.4)]", text: "text-red-400 font-bold bg-red-900/40 px-1 rounded", isInline: false };
                    }

                    return { bullet: "bg-gray-800", text: "text-gray-500 opacity-60", isInline: false };
                };

                const renderNodeList = (nodes, currentActive, otherActive) => nodes.map((n) => {
                    const filename = getFilename(n);
                    const score = typeof n === "string" ? null : n.score;
                    const s = getNodeStyle(filename, currentActive, otherActive);
                    const bStyle = s.isInline ? `style="${s.bullet}"` : `class="w-1.5 h-1.5 rounded-full ${s.bullet}"`;
                    const tStyle = s.isInline ? `style="${s.text}"` : `class="${s.text}"`;
                    const scoreDisplay = score !== null ? ` [${Utils.formatScore(score)}]` : "";
                    return `
                            <li class="flex items-center gap-2 py-1">
                                ${s.isInline ? `<span class="w-1.5 h-1.5 rounded-full" ${bStyle}></span>` : `<span ${bStyle}></span>`}
                                <span ${tStyle} class="truncate text-[8px]">${filename}<span class="opacity-70">${scoreDisplay}</span></span>
                            </li>`;
                })
                    .join("");

                els.debugContent.innerHTML += `
                    <div class="col-span-1 md:col-span-2 mt-4 p-4 bg-purple-500/5 rounded-2xl border border-purple-500/20 shadow-2xl backdrop-blur-sm">
                        <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-4 border-b border-purple-500/10 pb-3">
                            <div>
                                <h5 class="text-purple-400 text-[10px] uppercase font-black tracking-[0.2em]">Refinement Topology</h5>
                                <p class="text-[8px] text-gray-500 mt-1">Cross-chain filename matching enabled</p>
                            </div>
                            <div class="flex flex-wrap gap-4 text-[7px] uppercase tracking-widest font-black">
                                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-red-500 shadow-[0_0_5px_red]"></span><span class="text-red-400">Waypoint</span></div>
                                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-blue-400 shadow-[0_0_5px_cyan]"></span><span class="text-blue-400">Shared Gap</span></div>
                                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-purple-500 shadow-[0_0_5px_purple]"></span><span class="text-purple-400">Selected</span></div>
                                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-pink-500 animate-pulse"></span><span class="text-pink-400">Other Side</span></div>
                            </div>
                        </div>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                             <!-- Left Data -->
                             <div class="space-y-4">
                                <div class="p-2 bg-black/20 rounded-lg border border-white/5">
                                    <p class="text-[9px] text-purple-300 font-bold mb-2 flex items-center gap-2">
                                        <span class="w-2 h-2 rounded-full bg-purple-500"></span> LEFT NODE CONTEXT
                                    </p>
                                    <div class="grid grid-cols-2 gap-2 text-[8px]">
                                        <div class="text-gray-500">Main Chain Pos: <span class="text-white font-mono">${rd.node_0.pos_in_chain}/${rd.node_0.chain_len}</span></div>
                                        <div class="text-gray-500">Gap Pos: <span class="text-white font-mono">${rd.node_0.pos_in_sub_chain}/${rd.node_0.sub_chain_len}</span></div>
                                    </div>
                                </div>
                                <div class="grid grid-cols-1 gap-4">
                                    <div>
                                        <p class="text-[7px] text-gray-500 uppercase mb-1.5 font-black tracking-tighter">Subchain List (${rd.node_0.sub_chain_len})</p>
                                        <div class="max-h-32 overflow-y-auto bg-black/60 p-2 rounded-xl border border-white/5 custom-scrollbar">
                                            <ul class="list-none font-mono">
                                                ${renderNodeList(rd.node_0.sub_chain_nodes, rd.node_0.filename, rd.node_1.filename)}
                                            </ul>
                                        </div>
                                    </div>
                                    <div>
                                        <p class="text-[7px] text-gray-500 uppercase mb-1.5 font-black tracking-tighter">Full Chain List (${rd.node_0.chain_len})</p>
                                        <div class="max-h-32 overflow-y-auto bg-black/60 p-2 rounded-xl border border-white/5 custom-scrollbar">
                                            <ul class="list-none font-mono">
                                                ${renderNodeList(rd.node_0.full_chain_nodes, rd.node_0.filename, rd.node_1.filename)}
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                             </div>
                             <!-- Right Data -->
                             <div class="space-y-4">
                                <div class="p-2 bg-black/20 rounded-lg border border-white/5">
                                    <p class="text-[9px] text-pink-300 font-bold mb-2 flex items-center gap-2">
                                        <span class="w-2 h-2 rounded-full bg-pink-500"></span> RIGHT NODE CONTEXT
                                    </p>
                                    <div class="grid grid-cols-2 gap-2 text-[8px]">
                                        <div class="text-gray-500">Main Chain Pos: <span class="text-white font-mono">${rd.node_1.pos_in_chain}/${rd.node_1.chain_len}</span></div>
                                        <div class="text-gray-500">Gap Pos: <span class="text-white font-mono">${rd.node_1.pos_in_sub_chain}/${rd.node_1.sub_chain_len}</span></div>
                                    </div>
                                </div>
                                <div class="grid grid-cols-1 gap-4">
                                    <div>
                                        <p class="text-[7px] text-gray-500 uppercase mb-1.5 font-black tracking-tighter">Subchain List (${rd.node_1.sub_chain_len})</p>
                                        <div class="max-h-32 overflow-y-auto bg-black/60 p-2 rounded-xl border border-white/5 custom-scrollbar">
                                            <ul class="list-none font-mono">
                                                ${renderNodeList(rd.node_1.sub_chain_nodes, rd.node_1.filename, rd.node_0.filename)}
                                            </ul>
                                        </div>
                                    </div>
                                    <div>
                                        <p class="text-[7px] text-gray-500 uppercase mb-1.5 font-black tracking-tighter">Full Chain List (${rd.node_1.chain_len})</p>
                                        <div class="max-h-32 overflow-y-auto bg-black/60 p-2 rounded-xl border border-white/5 custom-scrollbar">
                                            <ul class="list-none font-mono">
                                                ${renderNodeList(rd.node_1.full_chain_nodes, rd.node_1.filename, rd.node_0.filename)}
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                             </div>
                        </div>
                        <div class="mt-4 pt-3 border-t border-purple-500/20 flex justify-between items-center text-[9px] text-gray-400 font-black tracking-widest uppercase">
                            <span>Gap Index: ${rd.sub_chain_idx + 1} / ${rd.total_sub_chains}</span>
                            <span class="text-purple-400/80">Δ Score: ${rd.score_diff}</span>
                        </div>
                    </div>
                `;
            }
        }

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

    async submitVote(winner) {
        const winnerFilename = winner === "left" ? this.currentPair.left.filename : this.currentPair.right.filename;
        const filenameA = this.currentPair.left.filename;
        const filenameB = this.currentPair.right.filename;

        const leftSrc = this.currentPair._preloadedLeft?.src || `/images/${encodeURIComponent(filenameA)}`;
        const rightSrc = this.currentPair._preloadedRight?.src || `/images/${encodeURIComponent(filenameB)}`;

        this._submissionQueue.push({
            filenameA, filenameB, winnerFilename,
            leftSrc, rightSrc,
            queuedAt: Date.now(),
        });
        this._updateQueueIndicator();
        this._showUndoButton();

        // Start the processing loop
        this._startProcessLoop();

        // Immediately advance to next pair without waiting
        this.loadPair();
    }

    async _startProcessLoop() {
        if (this._processLoopRunning) {
            return;
        }
        this._processLoopRunning = true;
        this._processLoop();
    }

    async _processLoop() {
        while (this._processLoopRunning) {
            if (this._submissionQueue.length === 0) {
                this._processLoopRunning = false;
                this._startPrefetch();
                return;
            }

            const front = this._submissionQueue[0];
            const elapsed = Date.now() - front.queuedAt;
            const remaining = this._UNDO_DELAY_MS - elapsed;

            if (remaining <= 0) {
                // Time's up, submit it
                this._submissionQueue.shift();
                this._outgoingQueue.push(front);
                this._updateQueueIndicator();
                this._updateUndoUI();

                try {
                    await api.submitComparison(front.filenameA, front.filenameB, front.winnerFilename);
                    this._hasSubmitted = true;
                } catch (e) {
                    Utils.showToast("Failed to submit: " + e.message, "error");
                } finally {
                    this._outgoingQueue.shift();
                    this._updateQueueIndicator();
                    this._updateUndoUI();
                }
            } else {
                // Wait for remaining time, then loop again
                await new Promise(r => setTimeout(r, remaining));
            }
        }
    }

    _undoLast() {
        if (this._submissionQueue.length === 0) {
            return;
        }

        this._submissionQueue.pop();

        // Reset timer on the new last item so user gets another 10s to undo it
        if (this._submissionQueue.length > 0) {
            this._submissionQueue[this._submissionQueue.length - 1].queuedAt = Date.now();
        }

        this._updateQueueIndicator();
        this._updateUndoUI();
        Utils.showToast("Vote undone", "info");
    }

    _updateUndoUI() {
        const btn = this.elements.undoButton;
        if (!btn) {
            return;
        }

        if (this._submissionQueue.length === 0) {
            btn.style.display = "none";
            return;
        }

        btn.style.display = "block";

        const last = this._submissionQueue[this._submissionQueue.length - 1];
        const isLeftWinner = last.winnerFilename === last.filenameA;
        const leftStyle = isLeftWinner
            ? "width:72px;height:72px;object-fit:cover;border-radius:6px;border:2px solid #10b981;box-shadow:0 0 8px rgba(16,185,129,0.5);"
            : "width:72px;height:72px;object-fit:cover;border-radius:6px;border:1px solid rgba(255,255,255,0.1);opacity:0.5;";
        const rightStyle = !isLeftWinner
            ? "width:72px;height:72px;object-fit:cover;border-radius:6px;border:2px solid #10b981;box-shadow:0 0 8px rgba(16,185,129,0.5);"
            : "width:72px;height:72px;object-fit:cover;border-radius:6px;border:1px solid rgba(255,255,255,0.1);opacity:0.5;";
        this._undoThumbsEl.innerHTML = `
            <img src="${last.leftSrc}" style="${leftStyle}" />
            <img src="${last.rightSrc}" style="${rightStyle}" />
        `;

        // Animate progress bar and timer for the last item
        const start = last.queuedAt;
        const tick = () => {
            if (this._submissionQueue.length === 0 || this._submissionQueue[this._submissionQueue.length - 1] !== last) {
                return;
            }
            const elapsed = Date.now() - start;
            const pct = Math.min(elapsed / this._UNDO_DELAY_MS, 1);
            this._undoProgressEl.style.width = `${pct * 100}%`;
            const secs = Math.max(0, Math.ceil((this._UNDO_DELAY_MS - elapsed) / 1000));
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

    _startPrefetch() {
        const hasPendingSubmissions = this._outgoingQueue.length > 0 || this._processingQueue;
        if (hasPendingSubmissions || this._prefetching || this._pairCache.length >= this._targetCacheSize) {
            return;
        }
        this._prefetching = true;
        this._prefetchLoop();
    }

    async _prefetchLoop() {
        while (this._prefetching && this._pairCache.length < this._targetCacheSize) {
            const hasPendingSubmissions = this._outgoingQueue.length > 0 || this._processingQueue;
            if (hasPendingSubmissions) {
                await new Promise(r => setTimeout(r, 200));
                continue;
            }

            try {
                const pair = await api.getNextPair();
                if (!pair) {
                    break;
                }
                const leftUrl = `/images/${encodeURIComponent(pair.left.filename)}`;
                const rightUrl = `/images/${encodeURIComponent(pair.right.filename)}`;
                const leftImg = new Image();
                const rightImg = new Image();
                await Promise.all([
                    new Promise((r, j) => {
                        leftImg.onload = r; leftImg.onerror = j; leftImg.src = leftUrl;
                    }),
                    new Promise((r, j) => {
                        rightImg.onload = r; rightImg.onerror = j; rightImg.src = rightUrl;
                    }),
                ]);
                this._pairCache.push({ pair, leftImg, rightImg });
                this._updateQueueIndicator();
            } catch (e) {
                console.warn("Prefetch error:", e);
                break;
            }
        }
        this._prefetching = false;
    }

    showLoading(show) {
        const overlay = this.elements.loadingOverlay;
        if (!overlay) {
            return;
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

    async resetCache() {
        try {
            await api.resetCache();
            this._pairCache = [];
            this._prefetching = false;
            this._submissionQueue = [];
            this._outgoingQueue = [];
            this._processingQueue = false;
            this._processLoopRunning = false;
            this._updateQueueIndicator();
            this._updateUndoUI();
            Utils.showToast("Cache cleared!", "success");
        } catch (e) {
            Utils.showToast("Failed to reset cache: " + e.message, "error");
        }
    }

    async updateStats() {
        try {
            const status = await api.getStatus();
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
                els.statsComponents.textContent = status.total_components || 0;
            }
            if (els.statsChains) {
                els.statsChains.textContent = status.total_chains || 0;
            }
            if (els.statsSkipped) {
                els.statsSkipped.textContent = status.skipped_comparisons || 0;
            }
            if (els.statsLevel) {
                els.statsLevel.textContent = status.current_target;
            }
            if (els.statsActive) {
                els.statsActive.textContent = status.active_nodes || 0;
            }
            if (els.statsNextLevel) {
                els.statsNextLevel.textContent = status.next_level_count || 0;
            }
            if (els.statsNextLevelNum) {
                els.statsNextLevelNum.textContent = status.current_target + 1;
            }
            if (els.statsBaseLevel) {
                els.statsBaseLevel.textContent = status.base_level ?? status.baseline_comparisons;
            }
        } catch (e) { }
    }
}

window.compareMode = new CompareMode();
