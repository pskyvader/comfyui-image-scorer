/**
 * Comparison Mode Logic
 */

class CompareMode {
    constructor() {
        this.currentPair = null;
        this.nextPair = null;
        this._submitting = false;
        
        // These will be populated in init()
        this.elements = {};
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
            collapsibleIndicator: document.getElementById("collapsible-indicator")
        };
    }

    attachEventListeners() {
        const { leftButton, rightButton, skipButton, debugBtn, leftCard, rightCard } = this.elements;

        leftButton.onclick = () => this.submitVote("left");
        rightButton.onclick = () => this.submitVote("right");
        
        if (leftCard) leftCard.onclick = (e) => { if (e.target !== leftButton) this.submitVote("left"); };
        if (rightCard) rightCard.onclick = (e) => { if (e.target !== rightButton) this.submitVote("right"); };
        
        if (skipButton) skipButton.onclick = () => this.loadPair();
        if (debugBtn) debugBtn.onclick = () => this.elements.debugPanel.classList.toggle("hidden");
    }

    async init(params) {
        console.log("Initializing CompareMode...");
        this.cacheElements();
        this.attachEventListeners();
        
        // Reset pairs on load
        try {
            await fetch('/api/v2/ranking/reset', { method: 'POST' });
        } catch (e) {
            console.warn("Could not reset pairs:", e);
        }
        
        const leftFile = params?.get('left');
        const rightFile = params?.get('right');

        if (leftFile && rightFile) {
            this.updateStatus("Loading requested pair...");
            try {
                const [leftData, rightData] = await Promise.all([
                    api.getImage(leftFile),
                    api.getImage(rightFile)
                ]);
                this.currentPair = {
                    left: leftData,
                    right: rightData,
                    rationale: {
                        strategy: "Manual Selection (Chain Map)",
                        score_diff: Math.abs(leftData.score - rightData.score).toFixed(4),
                        seed: leftFile
                    }
                };
                await this.renderPair();
            } catch (e) {
                console.error("Manual load failed:", e);
                await this.loadPair();
            }
        } else {
            await this.loadPair();
        }

        this.preloadNextPair();
        this.updateStats();
    }

    async loadPair() {
        this.showLoading(true);
        this.updateStatus("Finding next pair...");

        try {
            const pair = await api.getNextPair();
            if (!pair) {
                this.updateStatus("No more pairs to compare!");
                this.elements.leftButton.disabled = true;
                this.elements.rightButton.disabled = true;
                this.showLoading(false);
                return;
            }
            this.currentPair = pair;
            this.renderPair();
            this.showLoading(false);
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

            if (totalEl) totalEl.textContent = gs.total_images.toLocaleString();
            if (rankedEl) rankedEl.textContent = (gs.level_count ?? 0).toLocaleString();
            if (levelEl) levelEl.textContent = gs.target_level;
            if (compsEl) compsEl.textContent = gs.total_comparisons.toLocaleString();
            if (compsStatEl) compsStatEl.textContent = (gs.total_components ?? 0).toLocaleString();
            if (chainsEl) chainsEl.textContent = (gs.total_chains ?? 0).toLocaleString();
            if (skippedEl) skippedEl.textContent = (gs.skipped_comparisons ?? 0).toLocaleString();
            if (activeEl) activeEl.textContent = (gs.active_nodes ?? 0).toLocaleString();
            if (nextLevelEl) nextLevelEl.textContent = (gs.next_level_count ?? 0).toLocaleString();
            if (nextLevelNumEl) nextLevelNumEl.textContent = gs.target_level + 1;
            if (baseLevelEl) baseLevelEl.textContent = gs.base_level ?? (gs.target_level - 1);
        }

        // Single line stats separated by |
        const statsText = `Score: ${Utils.formatScore(left.score)} | Chain: ${left.chain_length ?? "-"} | Component: ${left.component_size ?? "-"} | Comparisons: ${left.comparison_count ?? 0}`;
        const rightStatsText = `Score: ${Utils.formatScore(right.score)} | Chain: ${right.chain_length ?? "-"} | Component: ${right.component_size ?? "-"} | Comparisons: ${right.comparison_count ?? 0}`;
        
        const leftStatsEl = document.getElementById("left-stats-line");
        const rightStatsEl = document.getElementById("right-stats-line");
        if (leftStatsEl) leftStatsEl.textContent = statsText;
        if (rightStatsEl) rightStatsEl.textContent = rightStatsText;

        // Update borders based on comparison type
        els.leftCard.classList.remove("collapsible-card", "regular-card", "extra-card");
        els.rightCard.classList.remove("collapsible-card", "regular-card", "extra-card");

        const isExtra = left.component_id === right.component_id && 
                      (this.currentPair.debug?.left_extremes?.top === 1 && this.currentPair.debug?.left_extremes?.bottom === 1);

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
                                <td class="text-white font-bold">${rationale.pair_type || 'Standard'}</td>
                            </tr>
                            <tr class="h-6">
                                <td class="text-gray-500 pr-4">Chain Level</td>
                                <td class="text-white">${(rationale.chain_level === -1 || rationale.chain_level === undefined) ? 'Auto' : rationale.chain_level}</td>
                            </tr>
                            <tr class="h-6">
                                <td class="text-gray-500 pr-4">Comp Size</td>
                                <td class="text-white">${rationale.left_component_size || '-'} vs ${rationale.right_component_size || '-'}</td>
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
        }

        els.leftImg.src = `/images/${encodeURIComponent(left.filename)}`;
        els.rightImg.src = `/images/${encodeURIComponent(right.filename)}`;

        await Promise.all([
            Utils.preloadImage(els.leftImg.src),
            Utils.preloadImage(els.rightImg.src)
        ]);

        els.leftImg.classList.remove("opacity-0");
        els.rightImg.classList.remove("opacity-0");
        els.leftImg.classList.add("opacity-100");
        els.rightImg.classList.add("opacity-100");
    }

    async submitVote(winner) {
        if (this._submitting) return;
        this._submitting = true;
        
        const previousPair = this.currentPair;
        const winnerFilename = winner === "left" ? this.currentPair.left.filename : this.currentPair.right.filename;
        const filenameA = this.currentPair.left.filename;
        const filenameB = this.currentPair.right.filename;
        
        if (this.nextPair && !this.isSamePair(previousPair, this.nextPair)) {
            this.currentPair = this.nextPair;
            this.nextPair = null;
            this._preloadLeft = null;
            this._preloadRight = null;
            this.renderPair();
            this.preloadNextPair();
        } else {
            this.loadPair();
        }
        
        api.submitComparison(filenameA, filenameB, winnerFilename)
            .then(() => {
                Utils.showToast("Vote recorded!", "success");
            })
            .catch(e => {
                Utils.showToast("Failed to submit: " + e.message, "error");
            });
        
        this._submitting = false;
    }

    async preloadNextPair() {
        try {
            this.nextPair = await api.getNextPair();
            if (this.nextPair) {
                // Cancel previous preload images
                if (this._preloadLeft) this._preloadLeft.onload = this._preloadLeft.onerror = null;
                if (this._preloadRight) this._preloadRight.onload = this._preloadRight.onerror = null;
                
                this._preloadLeft = new Image();
                this._preloadRight = new Image();
                this._preloadLeft.src = `/images/${encodeURIComponent(this.nextPair.left.filename)}`;
                this._preloadRight.src = `/images/${encodeURIComponent(this.nextPair.right.filename)}`;
            }
        } catch (e) { console.warn("Preload failed", e); }
    }

    showLoading(show) {
        const overlay = this.elements.loadingOverlay;
        if (!overlay) return;
        
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
        if (this.elements.status) this.elements.status.textContent = msg;
    }

    async updateStats() {
        try {
            const status = await api.getStatus();
            const els = this.elements;
            if (els.statsTotal) els.statsTotal.textContent = status.total_images;
            if (els.statsRanked) els.statsRanked.textContent = status.ranked_images;
            if (els.statsComparisons) els.statsComparisons.textContent = status.total_comparisons;
            if (els.statsComponents) els.statsComponents.textContent = status.total_components || 0;
            if (els.statsChains) els.statsChains.textContent = status.total_chains || 0;
            if (els.statsSkipped) els.statsSkipped.textContent = status.skipped_comparisons || 0;
            if (els.statsLevel) els.statsLevel.textContent = status.current_target;
            if (els.statsActive) els.statsActive.textContent = status.active_nodes || 0;
            if (els.statsNextLevel) els.statsNextLevel.textContent = status.next_level_count || 0;
            if (els.statsNextLevelNum) els.statsNextLevelNum.textContent = status.current_target + 1;
            if (els.statsBaseLevel) els.statsBaseLevel.textContent = status.base_level ?? status.baseline_comparisons;
        } catch (e) {}
    }
}

window.compareMode = new CompareMode();
