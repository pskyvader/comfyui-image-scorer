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

        this.playPauseBtn = document.getElementById("play-pause-btn");
        this.playIcon = document.getElementById("play-icon");
        this.pauseIcon = document.getElementById("pause-icon");

        this.statNodes = document.getElementById("stat-nodes");
        this.statEdges = document.getElementById("stat-edges");
        this.statComponents = document.getElementById("stat-components");
        this.statComparisons = document.getElementById("stat-comparisons");

        this.nodeDetails = document.getElementById("node-details");
        this.compareSelectedBtn = document.getElementById("compare-selected-btn");
        this.selectedCountEl = document.getElementById("selected-count");
        this.zoomScaleEl = document.getElementById("zoom-scale");
        this.viewCoordsEl = document.getElementById("view-coords");

        this.width = this.container?.clientWidth || 800;
        this.height = this.container?.clientHeight || 600;
    }

    async init() {
        console.log("Initializing Split-module ChainMapUI...");
        this.cacheElements();
        if (!this.container) return;

        this.loadFiltersFromStorage();
        this.setupSVGAndRenderer();
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

        if (this.compareSelectedBtn) {
            this.compareSelectedBtn.onclick = () => this.compareSelected();
        }

        const toggleLabelsBtn = document.getElementById("toggle-labels-btn");
        const iconOn = document.getElementById("tag-icon-on");
        const iconOff = document.getElementById("tag-icon-off");

        if (toggleLabelsBtn) {
            toggleLabelsBtn.onclick = () => {
                this.labelsOverridden = true;
                const current = this.renderer?.detailLevel?.showLabels;
                const newState = !current;
                this.renderer?.setDetailLevel({ ...this.renderer.detailLevel, showLabels: newState, labelCap: newState ? 1000 : 0 });

                if (iconOn) iconOn.classList.toggle("hidden", !newState);
                if (iconOff) iconOff.classList.toggle("hidden", newState);
            };
        }

        if (this.renderer?.detailLevel) {
            const isShown = this.renderer.detailLevel.showLabels;
            if (iconOn) iconOn.classList.toggle("hidden", !isShown);
            if (iconOff) iconOff.classList.toggle("hidden", isShown);
        }
    }

    render(nodes, links) {
        if (this.chainSim) this.chainSim.stop();

        const distinctComponents = new Set(nodes.map(n => String(n.component ?? "")));
        const totalComparisons = nodes.reduce((sum, n) => sum + (n.comparison_count || 0), 0);

        if (this.statNodes) this.statNodes.textContent = nodes.length.toLocaleString();
        if (this.statEdges) this.statEdges.textContent = links.length.toLocaleString();
        if (this.statComparisons) this.statComparisons.textContent = totalComparisons.toLocaleString();
        if (this.statComponents) this.statComponents.textContent = distinctComponents.size.toLocaleString();

        const colorScale = d3.scaleLinear()
            .domain([0, 0.4, 0.5, 0.6, 1])
            .range(["#ef4444", "#facc15", "#facc15", "#facc15", "#22c55e"]);

        const simNodes = nodes.map(d => ({
            ...d,
            _radius: 3 + Math.sqrt(d.comparison_count || 0) * 2,
            _fill: colorScale(d.score),
            _label: d.id.split('/').pop(),
            _shortLabel: d.id.split('/').pop().substring(0, 8) + "..."
        }));

        const nodeMap = new Map(simNodes.map(n => [n.id, n]));
        const simLinks = links.map(d => ({
            source: nodeMap.get(d.source),
            target: nodeMap.get(d.target),
            _opacity: 0.3
        })).filter(l => l.source && l.target);

        this.chainSim = new ChainSimulation(simNodes, simLinks, null, {
            width: this.width,
            height: this.height,
            onTick: () => this.renderer.update(simNodes, simLinks)
        });

        this.renderer.render({
            nodes: simNodes,
            links: simLinks,
            profile: (typeof MAP_VISUALS !== 'undefined') ? MAP_VISUALS : {},
            selectedIds: this.selectedNodes,
            world: this.calculateWorldBounds(simNodes)
        });

        if (this.loader) this.loader.classList.add("hidden");

        if (this.chainSim) {
            this.chainSim.initialize();
            this.chainSim.play();
        }
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
        const padding = 150;
        return {
            x: minX - padding,
            y: minY - padding,
            width: (maxX - minX) + padding * 2,
            height: (maxY - minY) + padding * 2,
            padding: padding
        };
    }

    cleanup() {
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
}

window.chainMapUI = new ChainMapUI();
