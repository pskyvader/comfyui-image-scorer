/**
 * Chain Map UI (Main Entry Point)
 */

class ChainMapUI {
    constructor() {
        this.rawData = null;
        this.chainSim = null;
        this.renderer = null;
        this.selectedNodes = [];
        this.labelsOverridden = false;
        this.linksOverridden = false;
        this._showLinks = true;
        this._showChainLinks = true;
        this._toggleChainBtn = null;
        this._selectedChainId = null;
        this.width = 800;
        this.height = 600;
    }

    cacheElements() {
        this.container = document.getElementById("graph-container");
        this.loader = document.getElementById("loader");
        this.tooltip = document.getElementById("tooltip");
        this.refreshBtn = document.getElementById("refresh-btn");
        this.resetViewBtn = document.getElementById("reset-view-btn");

        // Filters
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

        this.collapsibleFilter = document.getElementById("collapsible-filter");
        this.nodeTypeFilter = document.getElementById("node-type-filter");

        this.playPauseBtn = document.getElementById("play-pause-btn");
        this.playIcon = document.getElementById("play-icon");
        this.pauseIcon = document.getElementById("pause-icon");

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

    async init() {
        console.log("Initializing Split-module ChainMapUI...");
        this.cacheElements();
        if (!this.container) return;

        this.loadFiltersFromStorage();
        this.setupSVGAndRenderer();
        this.renderer.resize();
        window.addEventListener("resize", () => this.renderer.resize());
        this.attachEventListeners();
        await this.loadData();
    }

    attachEventListeners() {
        if (this.refreshBtn) this.refreshBtn.onclick = () => this.loadData();
        if (this.resetViewBtn) this.resetViewBtn.onclick = () => this.resetView();
        if (this.playPauseBtn) this.playPauseBtn.onclick = () => this.toggleSimulation();

        const handleFilterInput = (minEl, maxEl, minValEl, maxValEl, keyPrefix, isMin) => {
            let min = parseInt(minEl.value);
            let max = parseInt(maxEl.value);
            const sliderMax = parseInt(maxEl.max);

            if (min > max) {
                if (isMin) maxEl.value = min;
                else minEl.value = max;
                min = parseInt(minEl.value);
                max = parseInt(maxEl.value);
            }

            if (minValEl) minValEl.textContent = min;
            if (maxValEl) maxValEl.textContent = max >= sliderMax ? "Max" : max;

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
            this.minCompFilter.oninput = () => handleFilterInput(this.minCompFilter, this.maxCompFilter, this.minCompVal, this.maxCompVal, "comp", true);
            this.minCompFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        }
        if (this.maxCompFilter) {
            this.maxCompFilter.oninput = () => handleFilterInput(this.minCompFilter, this.maxCompFilter, this.minCompVal, this.maxCompVal, "comp", false);
            this.maxCompFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        }
        if (this.minChainFilter) {
            this.minChainFilter.oninput = () => handleFilterInput(this.minChainFilter, this.maxChainFilter, this.minChainVal, this.maxChainVal, "chain", true);
            this.minChainFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        }
        if (this.maxChainFilter) {
            this.maxChainFilter.oninput = () => handleFilterInput(this.minChainFilter, this.maxChainFilter, this.minChainVal, this.maxChainVal, "chain", false);
            this.maxChainFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        }

        if (this.minCompCountFilter) {
            this.minCompCountFilter.oninput = () => handleFilterInput(this.minCompCountFilter, this.maxCompCountFilter, this.minCompCountVal, this.maxCompCountVal, "compcount", true);
            this.minCompCountFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        }
        if (this.maxCompCountFilter) {
            this.maxCompCountFilter.oninput = () => handleFilterInput(this.minCompCountFilter, this.maxCompCountFilter, this.minCompCountVal, this.maxCompCountVal, "compcount", false);
            this.maxCompCountFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
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
        const iconOn = document.getElementById("tag-icon-on");
        const iconOff = document.getElementById("tag-icon-off");

        if (toggleLabelsBtn) {
            toggleLabelsBtn.onclick = () => {
                this.labelsOverridden = true;
                const current = this.renderer.detailLevel.showLabels;
                const newState = !current;
                this.renderer.setDetailLevel({ ...this.renderer.detailLevel, showLabels: newState, labelCap: newState ? RENDER.label.labelCap : 0 });
                this.syncButtonStates();
            };
        }

        if (this.renderer.detailLevel) {
            this.syncButtonStates();
        }

        const toggleLinksBtn = document.getElementById("toggle-links-btn");

        // Change master link toggle icon to arrow
        if (toggleLinksBtn) {
            toggleLinksBtn.innerHTML =
                '<svg id="links-icon-on" class="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
                '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14M12 5l7 7-7 7" />' +
                '</svg>' +
                '<svg id="links-icon-off" class="w-5 h-5 hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
                '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14M12 5l7 7-7 7" />' +
                '</svg>';
            toggleLinksBtn.onclick = () => {
                this.linksOverridden = true;
                this._showLinks = !this._showLinks;
                this._updateDetailShowLinks();
                this._applyLinkVisibility();
                this.syncButtonStates();
            };
        }

        // Helper to create a toggle button with on/off SVGs
        const createToggleBtn = (title, onPath, offPath) => {
            const btn = document.createElement("button");
            btn.className = "p-2 bg-black/60 hover:bg-black/80 rounded-lg text-white border border-white/10";
            btn.title = title;
            btn.innerHTML =
                '<svg class="icon-on w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">' + onPath + '</svg>' +
                '<svg class="icon-off w-5 h-5 hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24">' + offPath + '</svg>';
            return btn;
        };

        // Chain links toggle (blue chain links)
        this._toggleChainBtn = createToggleBtn(
            "Toggle Chain Links",
            '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />',
            '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6" />',
        );
        this._toggleChainBtn.onclick = () => {
            this._showChainLinks = !this._showChainLinks;
            this._updateDetailShowLinks();
            this._applyLinkVisibility();
            this.syncButtonStates();
        };

        // Insert chain toggle after the master toggle
        if (toggleLinksBtn && toggleLinksBtn.parentNode) {
            toggleLinksBtn.parentNode.insertBefore(this._toggleChainBtn, toggleLinksBtn.nextSibling);
        }

        if (this.renderer.detailLevel) {
            this.syncButtonStates();
        }
    }

    _updateDetailShowLinks() {
        this.renderer.setDetailLevel({
            ...this.renderer.detailLevel,
            showLinks: this._showLinks || this._showChainLinks,
        });
    }

    selectChain(chainId) {
        this._selectedChainId = chainId;
        const nodeSet = this.chainSim && this.chainSim._chainNodeIds ? this.chainSim._chainNodeIds.get(chainId) : null;
        this.renderer.setHighlightChain(chainId, nodeSet);
    }

    clearChainSelection() {
        if (this._selectedChainId === null) return;
        this._selectedChainId = null;
        this.renderer.clearHighlightChain();
    }

    _applyLinkVisibility() {
        if (!this.chainSim) return;

        const showLinks = this._showLinks;
        const showChain = this._showChainLinks;
        const links = this.chainSim.links;

        if (showChain && showLinks) {
            this.renderer.setVisibleLinkCount(links.length);
        } else if (showChain) {
            // Only chain links: sort chain-first
            links.sort((a, b) => {
                if (a._isChainLink && !b._isChainLink) return -1;
                if (!a._isChainLink && b._isChainLink) return 1;
                return 0;
            });
            this.renderer.setVisibleLinkCount(this.chainSim._chainLinkCount);
        } else if (showLinks) {
            // Only cross-chain links: sort cross-chain-first
            links.sort((a, b) => {
                if (!a._isChainLink && b._isChainLink) return -1;
                if (a._isChainLink && !b._isChainLink) return 1;
                return 0;
            });
            this.renderer.setVisibleLinkCount(links.length - this.chainSim._chainLinkCount);
        } else {
            this.renderer.setVisibleLinkCount(0);
        }

        this.renderer.update(this.chainSim.nodes, links);
    }

    _syncToggleStyle(btn, isOn) {
        if (!btn) return;
        const iconOn = btn.querySelector(".icon-on");
        const iconOff = btn.querySelector(".icon-off");
        if (iconOn) iconOn.classList.toggle("hidden", !isOn);
        if (iconOff) iconOff.classList.toggle("hidden", isOn);
    }

    render(nodes, links, stats) {
        if (this._renderTimeout) {
            clearTimeout(this._renderTimeout);
            this._renderTimeout = null;
        }
        if (this.chainSim) {
            console.log("RENDER stopping old simulation");
            this.chainSim.stop();
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

        const colorScale = d3.scaleLinear()
            .domain(RENDER.node.colorDomain)
            .range(RENDER.node.colorRange);

        const componentSizeMap = {};
        if (this.rawData.components) {
            for (const [compId, members] of Object.entries(this.rawData.components)) {
                componentSizeMap[compId] = members.length;
            }
        }

        const simNodes = nodes.map(d => ({
            ...d,
            _radius: RENDER.node.baseRadius + Math.sqrt(d.comparison_count) * RENDER.node.radiusMultiplier,
            _fill: colorScale(d.score),
            _label: d.id.split('/').pop(),
            _shortLabel: d.id.split('/').pop().substring(0, RENDER.node.labelTruncateLength) + "...",
            _component_size: componentSizeMap[String(d.component)]
        }));

        const nodeMap = new Map(simNodes.map(n => [n.id, n]));
        const simLinks = links.map(d => ({
            source: nodeMap.get(d.source),
            target: nodeMap.get(d.target),
            _opacity: RENDER.node.defaultOpacity
        })).filter(l => l.source && l.target);

        const chains = (this.rawData && this.rawData.chains) ? this.rawData.chains.map(c => ({
            id: c.id,
            component: c.component,
            nodes: c.nodes,
        })) : [];

        const self = this;

        this.chainSim = new ChainSimulation(simNodes, simLinks, chains, [], {
            width: this.width,
            height: this.height,
            onTick: (data) => {
                self.renderer.setVisibleNodeCount(data.visibleNodeCount);
                self.renderer.setVisibleLinkCount(data.visibleLinkCount);
                self.renderer.update(simNodes, simLinks);

                if (data.done) {
                    if (self.loader) self.loader.classList.add("hidden");

                    if (self.chainSim.runtime.startPaused) {
                        self.chainSim.pause();
                        self.labelsOverridden = true;
                        self.linksOverridden = true;
                        self.renderer.setDetailLevel({
                            ...self.renderer.detailLevel,
                            showLabels: false,
                            showLinks: false,
                        });
                    }

                    self._applyLinkVisibility();
                    self.syncButtonStates();

                    const worldW = self.chainSim.effectiveWidth;
                    const worldH = self.chainSim.effectiveHeight;
                    const scale = Math.min(CAMERA.maxFitScale, CAMERA.fitPadding / Math.max(worldW / self.width, worldH / self.height));
                    const transform = d3.zoomIdentity
                        .translate(self.width / 2, self.height / 2)
                        .scale(scale)
                        .translate(-worldW / 2, -worldH / 2);
                    if (isFinite(transform.x) && isFinite(transform.y) && isFinite(transform.k)) {
                        d3.select(self.container).call(self.zoom.transform, transform);
                        self.renderer.setTransform(transform);
                    }
                }
            }
        });

        this.chainSim.initialize();

        this.renderer.render({
            nodes: simNodes,
            links: simLinks,
            profile: this.chainSim.runtime,
            selectedIds: this.selectedNodes,
            world: {
                x: 0,
                y: 0,
                width: this.chainSim.effectiveWidth,
                height: this.chainSim.effectiveHeight
            }
        });
        this.renderer.setVisibleNodeCount(0);
        this.renderer.setVisibleLinkCount(0);

        // Fit world to viewport immediately (before phases begin)
        const worldW = this.chainSim.effectiveWidth;
        const worldH = this.chainSim.effectiveHeight;
        const initScale = Math.min(CAMERA.maxFitScale, CAMERA.fitPadding / Math.max(worldW / this.width, worldH / this.height));
        const initTransform = d3.zoomIdentity
            .translate(this.width / 2, this.height / 2)
            .scale(initScale)
            .translate(-worldW / 2, -worldH / 2);
        if (isFinite(initTransform.x) && isFinite(initTransform.y) && isFinite(initTransform.k)) {
            d3.select(this.container).call(this.zoom.transform, initTransform);
            this.renderer.setTransform(initTransform);
        }

        this.syncButtonStates();

        // Begin phased chain-based layout
        this.chainSim.start();
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
        if (this.chainSim) this.chainSim.stop();
        if (this.renderer) {
            this.renderer.destroy();
            this.renderer = null;
        }
        if (this.svg) {
            this.svg.remove();
            this.svg = null;
        }
        this.rawData = null;
        this.selectedNodes = [];
    }

    syncButtonStates() {
        const paused = this.chainSim && this.chainSim.isPaused;
        if (this.playIcon && this.pauseIcon) {
            this.playIcon.classList.toggle("hidden", !paused);
            this.pauseIcon.classList.toggle("hidden", paused);
        }

        if (this.renderer) {
            const showLabels = this.renderer.detailLevel.showLabels;
            const iconOn = document.getElementById("tag-icon-on");
            const iconOff = document.getElementById("tag-icon-off");
            if (iconOn) iconOn.classList.toggle("hidden", !showLabels);
            if (iconOff) iconOff.classList.toggle("hidden", showLabels);

            const linksIconOn = document.getElementById("links-icon-on");
            const linksIconOff = document.getElementById("links-icon-off");
            if (linksIconOn) linksIconOn.classList.toggle("hidden", !this._showLinks);
            if (linksIconOff) linksIconOff.classList.toggle("hidden", this._showLinks);
        }

        this._syncToggleStyle(this._toggleChainBtn, this._showChainLinks);
    }
}

window.chainMapUI = new ChainMapUI();
