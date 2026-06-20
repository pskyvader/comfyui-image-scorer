/**
 * Filter and Control Logic for ChainMapUI
 */

ChainMapUI.prototype.loadFiltersFromStorage = function () {
    const loadFilter = (el, valEl, key, cfg, isMin) => {
        if (!el) {
            return;
        }
        const saved = localStorage.getItem(key);
        const val = saved !== null ? parseInt(saved) : (isMin ? cfg.min : cfg.max);
        el.value = valueToSlider(val, cfg);
        if (valEl) {
            const isMax = val >= cfg.max;
            valEl.textContent = (isMax && !isMin) ? "Max" : val;
        }
    };

    loadFilter(this.minCompFilter, this.minCompVal, "chainmap_minComp", SLIDER.comp, true);
    loadFilter(this.maxCompFilter, this.maxCompVal, "chainmap_maxComp", SLIDER.comp, false);
    loadFilter(this.minChainFilter, this.minChainVal, "chainmap_minChain", SLIDER.chain, true);
    loadFilter(this.maxChainFilter, this.maxChainVal, "chainmap_maxChain", SLIDER.chain, false);
    loadFilter(this.minCompCountFilter, this.minCompCountVal, "chainmap_minCompCount", SLIDER.compCount, true);
    loadFilter(this.maxCompCountFilter, this.maxCompCountVal, "chainmap_maxCompCount", SLIDER.compCount, false);
    loadFilter(this.linkLengthFilter, this.linkLengthVal, "chainmap_linkLength", SLIDER.linkLength, true);

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
    localStorage.setItem("chainmap_minComp", sliderToValue(parseInt(this.minCompFilter.value), SLIDER.comp));
    localStorage.setItem("chainmap_maxComp", sliderToValue(parseInt(this.maxCompFilter.value), SLIDER.comp));
    localStorage.setItem("chainmap_minChain", sliderToValue(parseInt(this.minChainFilter.value), SLIDER.chain));
    localStorage.setItem("chainmap_maxChain", sliderToValue(parseInt(this.maxChainFilter.value), SLIDER.chain));
    localStorage.setItem("chainmap_minCompCount", sliderToValue(parseInt(this.minCompCountFilter.value), SLIDER.compCount));
    localStorage.setItem("chainmap_maxCompCount", sliderToValue(parseInt(this.maxCompCountFilter.value), SLIDER.compCount));
    if (this.linkLengthFilter) {
        localStorage.setItem("chainmap_linkLength", sliderToValue(parseInt(this.linkLengthFilter.value), SLIDER.linkLength));
    }
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

    const minComp = sliderToValue(parseInt(this.minCompFilter.value), SLIDER.comp);
    const maxComp = sliderToValue(parseInt(this.maxCompFilter.value), SLIDER.comp);
    const minChain = sliderToValue(parseInt(this.minChainFilter.value), SLIDER.chain);
    const maxChain = sliderToValue(parseInt(this.maxChainFilter.value), SLIDER.chain);

    const minCompCount = this.minCompCountFilter ? sliderToValue(parseInt(this.minCompCountFilter.value), SLIDER.compCount) : 0;
    const maxCompCount = this.maxCompCountFilter ? sliderToValue(parseInt(this.maxCompCountFilter.value), SLIDER.compCount) : SLIDER.compCount.max;
    const collapsibleMode = this.collapsibleFilter ? this.collapsibleFilter.value : "all";
    const nodeTypeMode = this.nodeTypeFilter ? this.nodeTypeFilter.value : "all";

    this.saveFilters();

    const useMinComp = minComp > SLIDER.comp.min;
    const useMinChain = minChain > SLIDER.chain.min;
    const useMinCompCount = minCompCount > SLIDER.compCount.min;
    const useMaxComp = maxComp < SLIDER.comp.max;
    const useMaxChain = maxChain < SLIDER.chain.max;
    const useMaxCompCount = maxCompCount < SLIDER.compCount.max;

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
    // Simulation removed — no-op
};

ChainMapUI.prototype.resetView = function () {
    if (!this.renderer || !this.renderer.subRenderer) return;
    const sub = this.renderer.subRenderer;
    if (sub.worldBounds) {
        sub._fitToWorld(sub.worldBounds);
    }
};

ChainMapUI.prototype.zoomToFitNodes = function () {
    const nodes = this._simNodes;
    if (!nodes || !nodes.length) return;
    if (!this.renderer || !this.renderer.subRenderer) return;
    const bounds = this.calculateWorldBounds(nodes);
    this.renderer.subRenderer._fitToWorld(bounds);
};
