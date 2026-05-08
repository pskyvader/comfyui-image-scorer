/**
 * Filter and Control Logic for ChainMapUI
 */

ChainMapUI.prototype.loadFiltersFromStorage = function() {
    const loadFilter = (el, valEl, key, defaultVal, isMin) => {
        if (!el) return;
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
    loadFilter(this.maxCompCountFilter, this.maxCompCountVal, "chainmap_maxCompCount", 10, false);

    if (this.collapsibleFilter) {
        const saved = localStorage.getItem("chainmap_collapsible");
        if (saved) this.collapsibleFilter.value = saved;
    }
};

ChainMapUI.prototype.saveFilters = function() {
    localStorage.setItem("chainmap_minComp", this.minCompFilter?.value || 1);
    localStorage.setItem("chainmap_maxComp", this.maxCompFilter?.value || 30);
    localStorage.setItem("chainmap_minChain", this.minChainFilter?.value || 1);
    localStorage.setItem("chainmap_maxChain", this.maxChainFilter?.value || 30);
    localStorage.setItem("chainmap_minCompCount", this.minCompCountFilter?.value || 0);
    localStorage.setItem("chainmap_maxCompCount", this.maxCompCountFilter?.value || 10);
    if (this.collapsibleFilter) {
        localStorage.setItem("chainmap_collapsible", this.collapsibleFilter.value);
    }
};

ChainMapUI.prototype.debouncedApplyFilters = function() {
    if (this.filterTimer) clearTimeout(this.filterTimer);

    const bar = document.getElementById("filter-delay-bar");
    const container = document.getElementById("filter-delay-container");

    if (container) container.classList.remove("hidden");
    if (this.loader) this.loader.classList.remove("hidden"); 
    if (bar) {
        bar.style.transition = 'none';
        bar.style.width = '0%';
        void bar.offsetWidth;
        bar.style.transition = 'width 3000ms linear';
        bar.style.width = '100%';
    }

    this.filterTimer = setTimeout(() => {
        this.applyFilters();
        if (container) container.classList.add("hidden");
        this.filterTimer = null;
    }, 3000);
};

ChainMapUI.prototype.applyFilters = function() {
    if (!this.rawData) return;
    if (this.loader) this.loader.classList.remove("hidden");

    const minComp = parseInt(this.minCompFilter.value);
    const maxComp = parseInt(this.maxCompFilter.value);
    const minChain = parseInt(this.minChainFilter.value);
    const maxChain = parseInt(this.maxChainFilter.value);

    const minCompCount = this.minCompCountFilter ? parseInt(this.minCompCountFilter.value) : 0;
    const maxCompCount = this.maxCompCountFilter ? parseInt(this.maxCompCountFilter.value) : 10;
    const collapsibleMode = this.collapsibleFilter ? this.collapsibleFilter.value : 'all';

    const maxLimit = (typeof MAP_FILTERS !== 'undefined') ? MAP_FILTERS.sliderMax : 20;

    this.saveFilters();

    const useMinComp = minComp > 1;
    const useMaxComp = maxComp < maxLimit;
    const useMinChain = minChain > 1;
    const useMaxChain = maxChain < maxLimit;
    const useMinCompCount = minCompCount > 0;
    const useMaxCompCount = maxCompCount < 10;

    const validComponents = new Set();
    const nodeMap = new Map(this.rawData.nodes.map(n => [n.id, n]));

    Object.entries(this.rawData.components || {}).forEach(([id, members]) => {
        const size = members.length;
        const meetsMinComp = !useMinComp || size >= minComp;
        const meetsMaxComp = !useMaxComp || size <= maxComp;

        let maxH = 0;
        let topNodeCount = 0;
        let bottomNodeCount = 0;
        let anyNodeMatchesCompCount = false;

        members.forEach(nid => {
            const n = nodeMap.get(nid);
            if (n) {
                if ((n.height || 0) > maxH) maxH = n.height;
                if (n.is_top) topNodeCount++;
                if (n.is_bottom) bottomNodeCount++;

                const meetsMinC = !useMinCompCount || (n.comparison_count || 0) >= minCompCount;
                const meetsMaxC = !useMaxCompCount || (n.comparison_count || 0) <= maxCompCount;
                if (meetsMinC && meetsMaxC) anyNodeMatchesCompCount = true;
            }
        });

        const meetsMinChain = !useMinChain || maxH >= minChain;
        const meetsMaxChain = !useMaxChain || maxH <= maxChain;

        const isCollapsible = topNodeCount >= 2 || bottomNodeCount >= 2;
        let meetsCollapsible = true;
        if (collapsibleMode === 'only') meetsCollapsible = isCollapsible;
        if (collapsibleMode === 'exclude') meetsCollapsible = !isCollapsible;

        if (meetsMinComp && meetsMaxComp && meetsMinChain && meetsMaxChain && meetsCollapsible && anyNodeMatchesCompCount) {
            validComponents.add(id.toString());
        }
    });

    const filteredNodes = (this.rawData.nodes || []).filter(n => {
        const compId = String(n.component ?? "");
        return validComponents.has(compId);
    });

    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredEdges = (this.rawData.edges || []).filter(e =>
        nodeIds.has(e.source) && nodeIds.has(e.target)
    );

    this.render(filteredNodes, filteredEdges);
    setTimeout(() => this.resetView(), 100);
};

ChainMapUI.prototype.toggleSimulation = function() {
    if (this.chainSim.isPaused) {
        this.chainSim.play();
        this.playIcon.classList.add("hidden");
        this.pauseIcon.classList.remove("hidden");
    } else {
        this.chainSim.pause();
        this.playIcon.classList.remove("hidden");
        this.pauseIcon.classList.add("hidden");
    }
};

ChainMapUI.prototype.resetView = function() {
    if (!this.rawData || !this.chainSim || !this.chainSim.nodes.length) {
        this.svg.transition().duration(MAP_INTERACTION.transitionDuration).call(
            this.zoom.transform,
            d3.zoomIdentity.translate(this.width / 2, this.height / 2).scale(0.05)
        );
        return;
    }

    const nodes = this.chainSim.nodes;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    nodes.forEach(n => {
        minX = Math.min(minX, n.x);
        maxX = Math.max(maxX, n.x);
        minY = Math.min(minY, n.y);
        maxY = Math.max(maxY, n.y);
    });

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const dx = maxX - minX;
    const dy = maxY - minY;
    const scale = Math.min(4, 0.8 / Math.max(dx / this.width, dy / this.height)) || 0.05;

    this.svg.transition().duration(MAP_INTERACTION.transitionDuration).call(
        this.zoom.transform,
        d3.zoomIdentity
            .translate(this.width / 2, this.height / 2)
            .scale(scale)
            .translate(-centerX, -centerY)
    );
};
