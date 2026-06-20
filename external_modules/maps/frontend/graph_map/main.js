class ChainMapUI {
    constructor() {
        this.rawData = null;
        this.renderer = null;
        this.selectedNodes = [];
        this.labelsOverridden = false;
        this._showMainChains = true;
        this._showRegularLinks = true;
        this.width = 800;
        this.height = 600;
        this._simLinks = [];
        this._simNodes = [];
    }

    cacheElements() {
        this.container = document.getElementById("graph-container");
        this.loader = document.getElementById("loader");
        this.tooltip = document.getElementById("tooltip");
        this.refreshBtn = document.getElementById("refresh-btn");
        this.resetViewBtn = document.getElementById("reset-view-btn");

        this.minCompFilter = document.getElementById("min-comp-filter");
        this.minCompVal = document.getElementById("min-comp-val");
        this.maxCompFilter = document.getElementById("max-comp-filter");
        this.maxCompVal = document.getElementById("max-comp-val");

        this.minChainFilter = document.getElementById("min-chain-filter");
        this.minChainVal = document.getElementById("min-chain-val");
        this.maxChainFilter = document.getElementById("max-chain-filter");
        this.maxChainVal = document.getElementById("max-chain-val");

        this.minCompCountFilter = document.getElementById("min-comp-count-filter");
        this.minCompCountVal = document.getElementById("min-comp-count-val");
        this.maxCompCountFilter = document.getElementById("max-comp-count-filter");
        this.maxCompCountVal = document.getElementById("max-comp-count-val");

        this.linkLengthFilter = document.getElementById("link-length-filter");
        this.linkLengthVal = document.getElementById("link-length-val");

        this.collapsibleFilter = document.getElementById("collapsible-filter");
        this.nodeTypeFilter = document.getElementById("node-type-filter");

        this.statNodes = document.getElementById("stat-nodes");
        this.statComponents = document.getElementById("stat-components");
        this.statComparisons = document.getElementById("stat-comparisons");
        this.statChains = document.getElementById("stat-chains");

        this.nodeDetails = document.getElementById("node-details");
        this.compareSelectedBtn = document.getElementById("compare-selected-btn");
        this.selectedCountEl = document.getElementById("selected-count");
        this.zoomScaleEl = document.getElementById("zoom-scale");
        this.viewCoordsEl = document.getElementById("view-coords");

        if (this.container.clientWidth > 0) {
            this.width = this.container.clientWidth;
            this.height = this.container.clientHeight;
        }
    }

    _adjustContainerHeight() {
        if (!this.container) return;
        this.container.style.height = `${Math.round(window.innerHeight * 0.65)}px`;
    }

    async init() {
        console.log("Initializing ChainMapUI...");
        this.cacheElements();
        if (!this.container) return;
        this._adjustContainerHeight();

        this.loadFiltersFromStorage();
        this.setupThreeRenderer();
        this.renderer.resize();
        const self = this;
        window.addEventListener("resize", () => {
            self._adjustContainerHeight();
            self.renderer.resize();
        });
        this.attachEventListeners();
        await this.loadData();
        requestAnimationFrame(() => {
            if (this.renderer) this.renderer.resize();
        });
    }

    attachEventListeners() {
        if (this.refreshBtn) this.refreshBtn.onclick = () => this.loadData();
        if (this.resetViewBtn) this.resetViewBtn.onclick = () => this.resetView();

        const handleFilterInput = (minEl, maxEl, minValEl, maxValEl, cfg, isMin) => {
            let minPos = parseInt(minEl.value);
            let maxPos = parseInt(maxEl.value);

            if (minPos > maxPos) {
                if (isMin) maxEl.value = minPos;
                else minEl.value = maxPos;
                minPos = parseInt(minEl.value);
                maxPos = parseInt(maxEl.value);
            }

            const minVal = sliderToValue(minPos, cfg);
            const maxVal = sliderToValue(maxPos, cfg);

            if (minValEl) minValEl.textContent = minVal;
            if (maxValEl) maxValEl.textContent = maxPos >= cfg.steps ? "Max" : maxVal;

            this.saveFilters();

            if (this.filterTimer) {
                clearTimeout(this.filterTimer);
                this.filterTimer = null;
                const container = document.getElementById("filter-delay-container");
                if (container) container.classList.add("hidden");
                if (this.loader) this.loader.classList.add("hidden");
            }
        };

        if (this.minCompFilter) {
            this.minCompFilter.oninput = () => handleFilterInput(this.minCompFilter, this.maxCompFilter, this.minCompVal, this.maxCompVal, SLIDER.comp, true);
            this.minCompFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        }
        if (this.maxCompFilter) {
            this.maxCompFilter.oninput = () => handleFilterInput(this.minCompFilter, this.maxCompFilter, this.minCompVal, this.maxCompVal, SLIDER.comp, false);
            this.maxCompFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        }
        if (this.minChainFilter) {
            this.minChainFilter.oninput = () => handleFilterInput(this.minChainFilter, this.maxChainFilter, this.minChainVal, this.maxChainVal, SLIDER.chain, true);
            this.minChainFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        }
        if (this.maxChainFilter) {
            this.maxChainFilter.oninput = () => handleFilterInput(this.minChainFilter, this.maxChainFilter, this.minChainVal, this.maxChainVal, SLIDER.chain, false);
            this.maxChainFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        }

        if (this.minCompCountFilter) {
            this.minCompCountFilter.oninput = () => handleFilterInput(this.minCompCountFilter, this.maxCompCountFilter, this.minCompCountVal, this.maxCompCountVal, SLIDER.compCount, true);
            this.minCompCountFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        }
        if (this.maxCompCountFilter) {
            this.maxCompCountFilter.oninput = () => handleFilterInput(this.minCompCountFilter, this.maxCompCountFilter, this.minCompCountVal, this.maxCompCountVal, SLIDER.compCount, false);
            this.maxCompCountFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        }
        if (this.linkLengthFilter) {
            this.linkLengthFilter.oninput = () => {
                const val = sliderToValue(parseInt(this.linkLengthFilter.value), SLIDER.linkLength);
                if (this.linkLengthVal) this.linkLengthVal.textContent = val;
                this.saveFilters();
            };
        }
        if (this.collapsibleFilter) {
            this.collapsibleFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        }
        if (this.nodeTypeFilter) {
            this.nodeTypeFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        }

        if (this.compareSelectedBtn) {
            this.compareSelectedBtn.onclick = () => this.compareSelected();
        }

        const toggleLabelsBtn = document.getElementById("toggle-labels-btn");
        if (toggleLabelsBtn) {
            toggleLabelsBtn.onclick = () => {
                this.labelsOverridden = true;
                const current = this.renderer.detailLevel.showLabels;
                const newState = !current;
                this.renderer.setDetailLevel({ ...this.renderer.detailLevel, showLabels: newState, labelCap: newState ? RENDER.label.labelCap : 0 });
                this.syncButtonStates();
            };
        }

        const toggleMainChainsBtn = document.getElementById("toggle-main-chains-btn");
        if (toggleMainChainsBtn) {
            toggleMainChainsBtn.onclick = () => {
                this._showMainChains = !this._showMainChains;
                this._applyChainVisibility();
                this.syncButtonStates();
            };
        }

        const toggleRegularLinksBtn = document.getElementById("toggle-regular-links-btn");
        if (toggleRegularLinksBtn) {
            toggleRegularLinksBtn.onclick = () => {
                this._showRegularLinks = !this._showRegularLinks;
                this._applyChainVisibility();
                this.syncButtonStates();
            };
        }

        const zoomInBtn = document.getElementById("zoom-in-btn");
        if (zoomInBtn) {
            zoomInBtn.onclick = () => {
                const sub = this.renderer && this.renderer.subRenderer;
                if (!sub) return;
                if (sub._fallback) {
                    sub._zoomLevel = Math.min(sub._zoomLevel * 1.5, 50);
                    sub._renderFallback();
                } else if (sub.camera && sub.controls) {
                    sub.camera.zoom = Math.min(sub.camera.zoom * 1.5, 50);
                    sub.camera.updateProjectionMatrix();
                    sub.controls.update();
                }
            };
        }

        const zoomOutBtn = document.getElementById("zoom-out-btn");
        if (zoomOutBtn) {
            zoomOutBtn.onclick = () => {
                const sub = this.renderer && this.renderer.subRenderer;
                if (!sub) return;
                if (sub._fallback) {
                    sub._zoomLevel = Math.max(sub._zoomLevel / 1.5, 0.05);
                    sub._renderFallback();
                } else if (sub.camera && sub.controls) {
                    sub.camera.zoom = Math.max(sub.camera.zoom / 1.5, 0.01);
                    sub.camera.updateProjectionMatrix();
                    sub.controls.update();
                }
            };
        }
    }

    updateDetailsPosition() {
        if (!this._activeDetailsNode || !this.nodeDetails || this.nodeDetails.classList.contains("hidden")) return;
        if (!this.renderer) return;

        const { x: screenX, y: screenY } = this.renderer.worldToScreen(
            this._activeDetailsNode.x, this._activeDetailsNode.y
        );

        const r = this._activeDetailsNode._radius || 4;
        const offset = Math.max(r, 20) + 10;

        this.nodeDetails.style.left = `${Math.round(screenX + offset)}px`;
        this.nodeDetails.style.top = `${Math.round(screenY + offset)}px`;
        this.nodeDetails.classList.remove('top-4', 'left-4');
    }

    _applyChainVisibility() {
        if (!this.renderer) return;
        this.renderer.setLinkVisibility(this._showMainChains, this._showRegularLinks);
    }

    render(nodes, links, stats) {
        if (this._renderTimeout) {
            clearTimeout(this._renderTimeout);
            this._renderTimeout = null;
        }

        if (nodes.length === 0) {
            if (this.statNodes) this.statNodes.textContent = "0";
            if (this.statComparisons) this.statComparisons.textContent = "0";
            if (this.statComponents) this.statComponents.textContent = "0";
            if (this.statChains) this.statChains.textContent = "0";
            this.renderer.render({ nodes: [], links: [], profile: {}, selectedIds: [], world: { x: 0, y: 0, width: 800, height: 600 } });
            if (this.loader) this.loader.classList.add("hidden");
            return;
        }

        for (let i = 0; i < nodes.length; i++) {
            const n = nodes[i];
            if (n.comparison_count === undefined) {
                throw new Error("node[" + i + "] (" + n.id + ") is missing comparison_count");
            }
            if (n.score === undefined) {
                throw new Error("node[" + i + "] (" + n.id + ") is missing score");
            }
        }

        const distinctComponents = new Set(nodes.map(n => String(n.component)));
        const totalChains = stats.total_chains;

        if (this.statNodes) this.statNodes.textContent = nodes.length.toLocaleString();
        if (this.statComparisons) this.statComparisons.textContent = links.length.toLocaleString();
        if (this.statComponents) this.statComponents.textContent = distinctComponents.size.toLocaleString();
        if (this.statChains) this.statChains.textContent = totalChains.toLocaleString();

        const componentSizeMap = {};
        if (this.rawData.components) {
            for (const [compId, members] of Object.entries(this.rawData.components)) {
                componentSizeMap[compId] = members.length;
            }
        }

        const colorScale = d3.scaleLinear()
            .domain(RENDER.node.colorDomain)
            .range(RENDER.node.colorRange);

        const simNodes = nodes.map(d => ({
            ...d,
            _radius: 0,
            _fill: colorScale(d.score),
            _label: d.id.split('/').pop(),
            _shortLabel: d.id.split('/').pop().substring(0, RENDER.node.labelTruncateLength) + "...",
            _component_size: componentSizeMap[String(d.component)]
        }));

        const nodeMap = new Map(simNodes.map(n => [n.id, n]));

        const mainChainEdges = new Set();
        const chainNodeIds = new Set();
        const chainInfo = new Map();

        if (this.rawData.chains) {
            for (const chain of this.rawData.chains) {
                const chainNodes = chain.nodes;
                for (let i = 0; i < chainNodes.length; i++) {
                    const id = chainNodes[i];
                    chainNodeIds.add(id);

                    const existingInfo = chainInfo.get(id);
                    if (!existingInfo) {
                        chainInfo.set(id, {
                            chainId: chain.id,
                            prev: i > 0 ? chainNodes[i - 1] : null,
                            next: i < chainNodes.length - 1 ? chainNodes[i + 1] : null,
                            chainIndex: i,
                            chainLength: chainNodes.length,
                            allChains: [chain.id]
                        });
                    } else {
                        existingInfo.allChains.push(chain.id);
                        if (existingInfo.chainLength < chainNodes.length) {
                            existingInfo.chainId = chain.id;
                            existingInfo.prev = i > 0 ? chainNodes[i - 1] : null;
                            existingInfo.next = i < chainNodes.length - 1 ? chainNodes[i + 1] : null;
                            existingInfo.chainIndex = i;
                            existingInfo.chainLength = chainNodes.length;
                        }
                    }

                    if (i < chainNodes.length - 1) {
                        const a = chainNodes[i];
                        const b = chainNodes[i + 1];
                        mainChainEdges.add(`${a}|${b}`);
                        mainChainEdges.add(`${b}|${a}`);
                    }
                }
            }
        }

        for (const n of simNodes) {
            const info = chainInfo.get(n.id);
            if (info) {
                n._chainPrev = info.prev;
                n._chainNext = info.next;
                n._chainId = info.chainId;
                n._chainIndex = info.chainIndex;
                n._chainLength = info.chainLength;
                n._allChains = info.allChains;
            }
        }

        const simLinks = [];
        const existingLinks = new Set();

        for (const d of links) {
            const source = nodeMap.get(d.source);
            const target = nodeMap.get(d.target);
            if (source && target) {
                const isMainChain = mainChainEdges.has(`${d.source}|${d.target}`);
                simLinks.push({
                    source: source,
                    target: target,
                    _opacity: RENDER.node.defaultOpacity,
                    isMainChain: isMainChain
                });
                existingLinks.add(`${d.source}|${d.target}`);
                existingLinks.add(`${d.target}|${d.source}`);
            }
        }

        if (this.rawData.chains) {
            for (const chain of this.rawData.chains) {
                const chainNodes = chain.nodes;
                for (let i = 0; i < chainNodes.length - 1; i++) {
                    const a = chainNodes[i];
                    const b = chainNodes[i + 1];
                    const source = nodeMap.get(a);
                    const target = nodeMap.get(b);

                    if (source && target && !existingLinks.has(`${a}|${b}`)) {
                        simLinks.push({
                            source: source,
                            target: target,
                            _opacity: RENDER.node.defaultOpacity,
                            isMainChain: true,
                            isSynthetic: true
                        });
                        existingLinks.add(`${a}|${b}`);
                        existingLinks.add(`${b}|${a}`);
                    }
                }
            }
        }

        this._simNodes = simNodes;
        this._simLinks = simLinks;

        // --- Chain diagnostics ---
        const nodesWithMainChain = new Set();
        const allNodeIds = new Set();
        for (const l of simLinks) {
            allNodeIds.add(l.source.id);
            allNodeIds.add(l.target.id);
            if (l.isMainChain) {
                nodesWithMainChain.add(l.source.id);
                nodesWithMainChain.add(l.target.id);
            }
        }
        const mainChainCount = simLinks.filter(l => l.isMainChain).length;
        const pct = simLinks.length > 0 ? (mainChainCount / simLinks.length * 100).toFixed(1) : "0";
        const noChainNodes = [...allNodeIds].filter(id => !nodesWithMainChain.has(id));
        console.log(`[chain] ${mainChainCount}/${simLinks.length} links are main chain (${pct}%). ` +
            `${nodesWithMainChain.size}/${allNodeIds.size} nodes have ≥1 main chain link.`);
        const rawEdgeNodes = new Set();
        if (this.rawData && this.rawData.edges) {
            for (const e of this.rawData.edges) {
                rawEdgeNodes.add(e.source);
                rawEdgeNodes.add(e.target);
            }
        }
        const chainMissing = [...rawEdgeNodes].filter(id => !chainNodeIds.has(id));
        if (chainMissing.length > 0) {
            console.error(`[chain] BACKEND BUG: ${chainMissing.length}/${rawEdgeNodes.size} nodes from raw edges are MISSING from all chains.`);
            const samples = chainMissing.slice(0, 10);
            for (const nid of samples) {
                console.error(`  ${nid} has comparisons but is in NO chain`);
            }
        } else if (rawEdgeNodes.size > 0) {
            console.log(`[chain] Backend OK: all ${rawEdgeNodes.size} nodes with comparisons belong to a chain.`);
        }
        if (noChainNodes.length > 0) {
            const sample = noChainNodes.slice(0, 20);
            console.warn(`[chain] ${noChainNodes.length} filtered nodes have ZERO main chain links. Sample:`);
            for (const nid of sample) {
                const info = chainInfo.get(nid);
                if (info) {
                    console.warn(`  ${nid}: chain=${info.chainId}, prev=${info.prev}, next=${info.next}, len=${info.chainLength}`);
                } else {
                    console.warn(`  ${nid}: NOT IN ANY CHAIN`);
                }
            }
            if (noChainNodes.length > 20) {
                console.warn(`  ... and ${noChainNodes.length - 20} more`);
            }
        }

        // --- Node sizing ---
        const degreeMap = new Map();
        for (const link of simLinks) {
            degreeMap.set(link.source.id, (degreeMap.get(link.source.id) || 0) + 1);
            degreeMap.set(link.target.id, (degreeMap.get(link.target.id) || 0) + 1);
        }
        const baseRadius = RENDER.node.baseRadius;
        const maxRadius = baseRadius * 10;
        for (const n of simNodes) {
            const degree = degreeMap.get(n.id) || 0;
            n._radius = Math.min(maxRadius, baseRadius + Math.pow(degree, 1.5) * RENDER.node.radiusMultiplier);
        }

        // --- Deterministic layout: arrange by component (X) and score (Y) ---
        const componentIds = [...distinctComponents].sort();
        const compSpacing = 300;
        const scoreScale = 500;
        const jitter = 25;

        for (const n of simNodes) {
            const compIdx = componentIds.indexOf(String(n.component));
            const compX = (compIdx - (componentIds.length - 1) / 2) * compSpacing;
            const scoreY = (parseFloat(n.score) - 0.5) * scoreScale;
            n.x = compX + (Math.random() - 0.5) * jitter;
            n.y = scoreY + (Math.random() - 0.5) * jitter;
        }

        // --- World bounds ---
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const n of simNodes) {
            if (n.x < minX) minX = n.x;
            if (n.x > maxX) maxX = n.x;
            if (n.y < minY) minY = n.y;
            if (n.y > maxY) maxY = n.y;
        }
        const padding = RENDER.bounds.padding;
        const world = {
            x: minX - padding,
            y: minY - padding,
            width: (maxX - minX) + padding * 2,
            height: (maxY - minY) + padding * 2,
        };

        // --- Render (async — chunked for 2D fallback) ---
        this.renderer.render({
            nodes: simNodes,
            links: simLinks,
            profile: { drawLinks: true },
            selectedIds: this.selectedNodes,
            world,
        }, () => {
            if (this.loader) this.loader.classList.add("hidden");
            this._applyChainVisibility();
            this.syncButtonStates();
        });
    }

    calculateWorldBounds(nodes) {
        if (!nodes || !nodes.length) return { x: -2500, y: -2500, width: 5000, height: 5000, padding: 0 };
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        nodes.forEach(n => {
            minX = Math.min(minX, n.x);
            maxX = Math.max(maxX, n.x);
            minY = Math.min(minY, n.y);
            maxY = Math.max(maxY, n.y);
        });
        return {
            x: minX - RENDER.bounds.padding,
            y: minY - RENDER.bounds.padding,
            width: (maxX - minX) + RENDER.bounds.padding * 2,
            height: (maxY - minY) + RENDER.bounds.padding * 2,
            padding: RENDER.bounds.padding
        };
    }

    cleanup() {
        if (this._renderTimeout) {
            clearTimeout(this._renderTimeout);
            this._renderTimeout = null;
        }
        if (this.renderer) {
            this.renderer.destroy();
            this.renderer = null;
        }
        this.rawData = null;
        this.selectedNodes = [];
        this._simNodes = [];
        this._simLinks = [];
    }

    syncButtonStates() {
        if (this.renderer) {
            const showLabels = this.renderer.detailLevel.showLabels;
            const iconOn = document.getElementById("tag-icon-on");
            const iconOff = document.getElementById("tag-icon-off");
            if (iconOn) iconOn.classList.toggle("hidden", !showLabels);
            if (iconOff) iconOff.classList.toggle("hidden", showLabels);

            const mainChainsIconOn = document.getElementById("main-chains-icon-on");
            const mainChainsIconOff = document.getElementById("main-chains-icon-off");
            if (mainChainsIconOn) mainChainsIconOn.classList.toggle("hidden", !this._showMainChains);
            if (mainChainsIconOff) mainChainsIconOff.classList.toggle("hidden", this._showMainChains);

            const regularLinksIconOn = document.getElementById("regular-links-icon-on");
            const regularLinksIconOff = document.getElementById("regular-links-icon-off");
            if (regularLinksIconOn) regularLinksIconOn.classList.toggle("hidden", !this._showRegularLinks);
            if (regularLinksIconOff) regularLinksIconOff.classList.toggle("hidden", this._showRegularLinks);
        }
    }
}

window.chainMapUI = new ChainMapUI();
window.Sections = window.Sections || {};
window.Sections.chains = ChainMapUI;
