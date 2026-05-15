/**
 * Filter and Control Logic for ChainMapUI
 */

ChainMapUI.prototype.loadFiltersFromStorage = function () {
    const loadFilter = (el, valEl, key, defaultVal, isMin) => {
        if (!el) {
            return;
        }
        const saved = localStorage.getItem(key);
        const val = saved !== null ? parseInt(saved) : defaultVal;
        el.value = val;
        if (valEl) {
            const isMax = val >= parseInt(el.max);
            valEl.textContent = (isMax && !isMin) ? "Max" : val;
        }
    };

    loadFilter(this.minCompFilter, this.minCompVal, "chainmap_minComp", 1, true);
    loadFilter(this.maxCompFilter, this.maxCompVal, "chainmap_maxComp", 30, false);
    loadFilter(this.minChainFilter, this.minChainVal, "chainmap_minChain", 1, true);
    loadFilter(this.maxChainFilter, this.maxChainVal, "chainmap_maxChain", 30, false);
    loadFilter(this.minCompCountFilter, this.minCompCountVal, "chainmap_minCompCount", 0, true);
    loadFilter(this.maxCompCountFilter, this.maxCompCountVal, "chainmap_maxCompCount", 100, false);

    if (this.collapsibleFilter) {
        const saved = localStorage.getItem("chainmap_collapsible");
        if (saved) {
            this.collapsibleFilter.value = saved;
        }
    }
    if (this.nodeTypeFilter) {
        const saved = localStorage.getItem("chainmap_nodeType");
        if (saved) {
            this.nodeTypeFilter.value = saved;
        }
    }
};

ChainMapUI.prototype.saveFilters = function () {
    localStorage.setItem("chainmap_minComp", this.minCompFilter.value);
    localStorage.setItem("chainmap_maxComp", this.maxCompFilter.value);
    localStorage.setItem("chainmap_minChain", this.minChainFilter.value);
    localStorage.setItem("chainmap_maxChain", this.maxChainFilter.value);
    localStorage.setItem("chainmap_minCompCount", this.minCompCountFilter.value);
    localStorage.setItem("chainmap_maxCompCount", this.maxCompCountFilter.value);
    if (this.collapsibleFilter) {
        localStorage.setItem("chainmap_collapsible", this.collapsibleFilter.value);
    }
    if (this.nodeTypeFilter) {
        localStorage.setItem("chainmap_nodeType", this.nodeTypeFilter.value);
    }
};

ChainMapUI.prototype.debouncedApplyFilters = function () {
    if (this.filterTimer) {
        clearTimeout(this.filterTimer);
    }

    const bar = document.getElementById("filter-delay-bar");
    const container = document.getElementById("filter-delay-container");

    if (container) {
        container.classList.remove("hidden");
    }
    if (this.loader) {
        this.loader.classList.remove("hidden");
    }
    if (bar) {
        bar.style.transition = "none";
        bar.style.width = "0%";
        void bar.offsetWidth;
        bar.style.transition = "width 3000ms linear";
        bar.style.width = "100%";
    }

    this.filterTimer = setTimeout(() => {
        this.applyFilters();
        if (container) {
            container.classList.add("hidden");
        }
        this.filterTimer = null;
    }, 3000);
};

ChainMapUI.prototype.applyFilters = function () {
    if (!this.rawData) {
        return;
    }
    if (this.loader) {
        this.loader.classList.remove("hidden");
    }

    const minComp = parseInt(this.minCompFilter.value);
    const maxComp = parseInt(this.maxCompFilter.value);
    const minChain = parseInt(this.minChainFilter.value);
    const maxChain = parseInt(this.maxChainFilter.value);

    const minCompCount = this.minCompCountFilter ? parseInt(this.minCompCountFilter.value) : 0;
    const maxCompCount = this.maxCompCountFilter ? parseInt(this.maxCompCountFilter.value) : 10;
    const collapsibleMode = this.collapsibleFilter ? this.collapsibleFilter.value : "all";
    const nodeTypeMode = this.nodeTypeFilter ? this.nodeTypeFilter.value : "all";

    this.saveFilters();

    const useMinComp = minComp > 1;
    const useMinChain = minChain > 1;
    const useMinCompCount = minCompCount > 0;
    const useMaxComp = maxComp < parseInt(this.maxCompFilter.max);
    const useMaxChain = maxChain < parseInt(this.maxChainFilter.max);
    const maxCompCountMax = parseInt(this.maxCompCountFilter.max);
    const useMaxCompCount = maxCompCount < maxCompCountMax;

    const validComponents = new Set();
    const nodeMap = new Map(this.rawData.nodes.map(n => [n.id, n]));

    Object.entries(this.rawData.components)
        .forEach(([id, members]) => {
            const size = members.length;

            let maxH = 0;
            let maxTopBottomH = 0;
            let filteredMaxH = 0;
            let topNodeCount = 0;
            let bottomNodeCount = 0;
            let filteredNodeCount = 0;
            let anyNodeMatchesCompCount = false;
            let anyFilteredMatchesCompCount = false;
            let topBottomCount = 0;

            const isNodeIncluded = (n) => {
                if (nodeTypeMode === "all") return true;
                if (nodeTypeMode === "top") return n.is_top;
                if (nodeTypeMode === "bottom") return n.is_bottom;
                if (nodeTypeMode === "extremes") return n.is_top || n.is_bottom;
                return true;
            };

            members.forEach((nid) => {
                const n = nodeMap.get(nid);
                if (n) {
                    if (n.height > maxH) {
                        maxH = n.height;
                    }
                    if (n.is_top || n.is_bottom) {
                        topBottomCount++;
                        if (n.height > maxTopBottomH) {
                            maxTopBottomH = n.height;
                        }
                    }
                    if (n.is_top) {
                        topNodeCount++;
                    }
                    if (n.is_bottom) {
                        bottomNodeCount++;
                    }

                    if (isNodeIncluded(n)) {
                        filteredNodeCount++;
                        if (n.height > filteredMaxH) {
                            filteredMaxH = n.height;
                        }
                    }

                    const meetsMinC = !useMinCompCount || n.comparison_count >= minCompCount;
                    const meetsMaxC = !useMaxCompCount || n.comparison_count <= maxCompCount;
                    if (meetsMinC && meetsMaxC) {
                        anyNodeMatchesCompCount = true;
                        if (isNodeIncluded(n)) {
                            anyFilteredMatchesCompCount = true;
                        }
                    }
                }
            });

            const meetsMinComp = !useMinComp || size >= minComp;
            const meetsMaxComp = !useMaxComp || size <= maxComp;

            const effectiveMaxH = collapsibleMode === "only" ? maxTopBottomH : filteredMaxH;
            const meetsMinChain = !useMinChain || effectiveMaxH >= minChain;
            const meetsMaxChain = !useMaxChain || effectiveMaxH <= maxChain;

            const isCollapsible = topNodeCount >= 2 || bottomNodeCount >= 2;
            let meetsCollapsible = true;
            if (collapsibleMode === "only") {
                meetsCollapsible = isCollapsible;
            }
            if (collapsibleMode === "exclude") {
                meetsCollapsible = !isCollapsible;
            }

            const compCountMatches = nodeTypeMode === "all" 
                ? anyNodeMatchesCompCount 
                : anyFilteredMatchesCompCount;

            if (meetsMinComp && meetsMaxComp && meetsMinChain && meetsMaxChain && meetsCollapsible && compCountMatches) {
                validComponents.add(id.toString());
            }
        });

    const filteredNodes = this.rawData.nodes.filter((n) => {
        const compId = String(n.component);
        return validComponents.has(compId);
    });

    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredEdges = this.rawData.edges.filter(e =>
        nodeIds.has(e.source) && nodeIds.has(e.target),
    );

    this.render(filteredNodes, filteredEdges, this.rawData.stats);
};

ChainMapUI.prototype.toggleSimulation = function () {
    if (this.chainSim.isPaused) {
        this.chainSim.play();
    } else {
        this.chainSim.pause();
    }
    this.syncButtonStates();
};

ChainMapUI.prototype.resetView = function () {
    const w = this.width || this.container.clientWidth || 800;
    const h = this.height || this.container.clientHeight || 600;
    const worldW = this.chainSim && this.chainSim.effectiveWidth;
    const worldH = this.chainSim && this.chainSim.effectiveHeight;

    if (!worldW || !worldH || !w || !h) {
        const t = d3.zoomIdentity.translate(w / 2, h / 2).scale(CAMERA.fallbackScale);
        if (isFinite(t.x) && isFinite(t.y) && isFinite(t.k)) {
            d3.select(this.container).call(this.zoom.transform, t);
            this.renderer.setTransform(t);
        }
        return;
    }

    const scale = Math.min(CAMERA.maxFitScale, CAMERA.fitPadding / Math.max(worldW / w, worldH / h));
    const t = d3.zoomIdentity.translate(w / 2, h / 2).scale(scale).translate(-worldW / 2, -worldH / 2);
    if (isFinite(t.x) && isFinite(t.y) && isFinite(t.k)) {
        d3.select(this.container).call(this.zoom.transform, t);
        this.renderer.setTransform(t);
    }
};

ChainMapUI.prototype.zoomToFitNodes = function () {
    const activeCount = this.chainSim._activeNodeCount;
    const nodes = this.chainSim.nodes.slice(0, activeCount);
    const bounds = this.calculateWorldBounds(nodes);
    const bw = bounds.width || 1;
    const bh = bounds.height || 1;

    if (!this.width || !this.height) return;

    const scale = Math.min(CAMERA.maxFitScale, CAMERA.fitPadding / Math.max(bw / this.width, bh / this.height));
    const cx = bounds.x + bw / 2;
    const cy = bounds.y + bh / 2;

    d3.select(this.container).call(
        this.zoom.transform,
        d3.zoomIdentity
            .translate(this.width / 2, this.height / 2)
            .scale(scale)
            .translate(-cx, -cy),
    );
};
