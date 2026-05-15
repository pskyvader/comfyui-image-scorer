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
        const linksIconOn = document.getElementById("links-icon-on");
        const linksIconOff = document.getElementById("links-icon-off");

        if (toggleLinksBtn) {
            toggleLinksBtn.onclick = () => {
                this.linksOverridden = true;
                const current = this.renderer.detailLevel.showLinks;
                const newState = !current;
                this.renderer.setDetailLevel({ ...this.renderer.detailLevel, showLinks: newState });
                this.syncButtonStates();
            };
        }

        if (this.renderer.detailLevel) {
            this.syncButtonStates();
        }
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
            throw new Error("render received 0 nodes — filters may have excluded everything");
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

        this.chainSim = new ChainSimulation(simNodes, simLinks, [], {
            width: this.width,
            height: this.height,
            onTick: () => {
                this.renderer.setVisibleNodeCount(this.chainSim._activeNodeCount);
                this.renderer.setVisibleLinkCount(simLinks.length);
                this.renderer.update(simNodes, simLinks);
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
        this.renderer.setVisibleNodeCount(this.chainSim._activeNodeCount);
        this.renderer.setVisibleLinkCount(simLinks.length);

        if (this.loader) this.loader.classList.add("hidden");

        const isLargeMap = this.chainSim.runtime.startPaused;

        if (isLargeMap) {
            this.chainSim.pause();
            this.labelsOverridden = true;
            this.linksOverridden = true;
            this.renderer.setDetailLevel({
                ...this.renderer.detailLevel,
                showLabels: false,
                showLinks: false,
            });
        }

        this.syncButtonStates();

        const self = this;
        const worldW = this.chainSim.effectiveWidth;
        const worldH = this.chainSim.effectiveHeight;
        const scale = Math.min(CAMERA.maxFitScale, CAMERA.fitPadding / Math.max(worldW / this.width, worldH / this.height));
        const transform = d3.zoomIdentity
            .translate(this.width / 2, this.height / 2)
            .scale(scale)
            .translate(-worldW / 2, -worldH / 2);
        if (isFinite(transform.x) && isFinite(transform.y) && isFinite(transform.k)) {
            d3.select(this.container).call(this.zoom.transform, transform);
            this.renderer.setTransform(transform);
        }

        this._renderTimeout = setTimeout(function () {
            if (self.chainSim && !self.chainSim.runtime.startPaused) {
                self.chainSim.play();
                self.syncButtonStates();
            }
        }, CONTROLS.transitionDuration + 100);
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

            const showLinks = this.renderer.detailLevel.showLinks;
            const linksIconOn = document.getElementById("links-icon-on");
            const linksIconOff = document.getElementById("links-icon-off");
            if (linksIconOn) linksIconOn.classList.toggle("hidden", !showLinks);
            if (linksIconOff) linksIconOff.classList.toggle("hidden", showLinks);
        }
    }
}

window.chainMapUI = new ChainMapUI();
