/**
 * Chain Map UI Logic (Canvas-based for Performance)
 */

class ChainMapUI {
    static config = {
        visuals: {
            nodeMinRadius: 3,
            highlightColor: "#f472b6",
            linkVisibilityZoomThreshold: 0.1,
            linkGlobalOpacity: 0.25,
            drawLinks: true,
            progressiveReveal: true,
            enablePointerHits: true
        },
        interaction: {
            zoomExtent: [0.005, 5],
            interactThreshold: 0.01,
            transitionDuration: 750,
        },
        filters: {
            sliderMax: 30,
        },
    };

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
        this.statChains = document.getElementById("stat-chains");
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
        console.log("Initializing Canvas-based ChainMapUI...");
        this.cacheElements();
        if (!this.container) return;

        this.loadFiltersFromStorage();
        this.setupSVGAndRenderer();
        this.attachEventListeners();
        await this.loadData();
    }

    loadFiltersFromStorage() {
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
    }

    saveFilters() {
        localStorage.setItem("chainmap_minComp", this.minCompFilter?.value || 1);
        localStorage.setItem("chainmap_maxComp", this.maxCompFilter?.value || 30);
        localStorage.setItem("chainmap_minChain", this.minChainFilter?.value || 1);
        localStorage.setItem("chainmap_maxChain", this.maxChainFilter?.value || 30);
        localStorage.setItem("chainmap_minCompCount", this.minCompCountFilter?.value || 0);
        localStorage.setItem("chainmap_maxCompCount", this.maxCompCountFilter?.value || 10);
        if (this.collapsibleFilter) {
            localStorage.setItem("chainmap_collapsible", this.collapsibleFilter.value);
        }
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

    setupSVGAndRenderer() {
        if (this.svg) {
            this.svg.remove();
        }

        this.svg = d3.select("#graph-container")
            .append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .style("position", "absolute")
            .style("top", 0)
            .style("left", 0)
            .style("z-index", 5)
            .style("pointer-events", "none");

        this.g = this.svg.append("g");
        
        this.zoom = d3.zoom()
            .scaleExtent(ChainMapUI.config.interaction.zoomExtent)
            .filter((event) => {
                // If it's a mousedown and we hit a node, don't pan (let drag handle it)
                if (event.type === "mousedown" || event.type === "touchstart") {
                    const canvas = d3.select(this.renderer.canvas);
                    const p = d3.pointer(event, canvas.node());
                    const hit = this.renderer?.hitTest({ x: p[0], y: p[1] });
                    if (hit) return false;
                }
                return !event.ctrlKey && !event.button; // Standard zoom filter
            })
            .on("zoom", (event) => {
                this.g.attr("transform", event.transform);
                this.updateHUD(event.transform);
                
                // Update detail levels based on zoom thresholds
                const k = event.transform.k;
                this.renderer?.setDetailLevel({
                    showLabels: this.labelsOverridden ? this.renderer.detailLevel.showLabels : k > 0.25,
                    showArrows: k > 0.15,
                    showNodeBorders: k > 0.2,
                    showLinks: k > 0.05,
                    labelCap: 500,
                    arrowCap: 2000,
                    zoom: k
                });
                this.renderer?.setTransform(event.transform);
            });
    }

    updateHUD(transform) {
        if (this.zoomScaleEl) this.zoomScaleEl.textContent = `${transform.k.toFixed(2)}x`;
        if (this.viewCoordsEl) {
            const centerX = Math.round((-transform.x + (this.width / 2)) / transform.k);
            const centerY = Math.round((-transform.y + (this.height / 2)) / transform.k);
            this.viewCoordsEl.textContent = `X: ${centerX}, Y: ${centerY}`;
        }
    }

        this.svg.call(this.zoom);
        // Initialize high-performance Canvas renderer
        this.renderer = new ChainMapRenderer({
            container: this.container,
            svg: this.svg,
            g: this.g,
            interaction: ChainMapUI.config.interaction
        });

        // Bind zoom, drag and click to CANVAS
        const canvas = d3.select(this.renderer.canvas);
        
        canvas.call(this.zoom)
            .on("click", (event) => {
                if (event.defaultPrevented) return; // ignore drag clicks
                const p = d3.pointer(event, canvas.node());
                const node = this.renderer?.hitTest({ x: p[0], y: p[1] });
                if (node) {
                    this.showNodeDetails(node);
                    // Do not toggle selection automatically to avoid accidental comparisons
                }
            })
            .on("dblclick.zoom", null);

        canvas.call(d3.drag()
            .container(canvas.node())
            .subject((event) => {
                const sourceEvent = event.sourceEvent || event;
                const p = d3.pointer(sourceEvent, canvas.node());
                return this.renderer?.hitTest({ x: p[0], y: p[1] });
            })
            .on("start", (event) => {
                if (this.chainSim?.isPaused) {
                    this.toggleSimulation(); // Unpause to allow physics to react
                }
                if (!event.active) {
                    this.chainSim?.simulation?.alphaTarget(0.3).restart();
                }
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            })
            .on("drag", (event) => {
                const transform = d3.zoomTransform(canvas.node());
                event.subject.fx = (event.x - transform.x) / transform.k;
                event.subject.fy = (event.y - transform.y) / transform.k;
            })
            .on("end", (event) => {
                if (!event.active) {
                    this.chainSim?.simulation?.alphaTarget(0);
                }
                event.subject.fx = null;
                event.subject.fy = null;
            })
        );

        canvas.on("mousemove", (event) => {
            const p = d3.pointer(event, canvas.node());
            const node = this.renderer?.hitTest({ x: p[0], y: p[1] });
            if (node) {
                this.showTooltip(event, node);
                canvas.style("cursor", "pointer");
            } else {
                this.hideTooltip();
                canvas.style("cursor", "default");
            }
        });
    }

    attachEventListeners() {
        this.refreshBtn.onclick = () => this.loadData();
        this.resetViewBtn.onclick = () => this.resetView();
        this.playPauseBtn.onclick = () => this.toggleSimulation();

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

        this.minCompFilter.oninput = () => handleFilterInput(this.minCompFilter, this.maxCompFilter, this.minCompVal, this.maxCompVal, "comp", true);
        this.maxCompFilter.oninput = () => handleFilterInput(this.minCompFilter, this.maxCompFilter, this.minCompVal, this.maxCompVal, "comp", false);
        this.minChainFilter.oninput = () => handleFilterInput(this.minChainFilter, this.maxChainFilter, this.minChainVal, this.maxChainVal, "chain", true);
        this.maxChainFilter.oninput = () => handleFilterInput(this.minChainFilter, this.maxChainFilter, this.minChainVal, this.maxChainVal, "chain", false);
        
        if (this.minCompCountFilter) {
            this.minCompCountFilter.oninput = () => handleFilterInput(this.minCompCountFilter, this.maxCompCountFilter, this.minCompCountVal, this.maxCompCountVal, "compcount", true);
            this.maxCompCountFilter.oninput = () => handleFilterInput(this.minCompCountFilter, this.maxCompCountFilter, this.minCompCountVal, this.maxCompCountVal, "compcount", false);
        }

        this.minCompFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        this.maxCompFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        this.minChainFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        this.maxChainFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        if (this.minCompCountFilter) this.minCompCountFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        if (this.maxCompCountFilter) this.maxCompCountFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };
        if (this.collapsibleFilter) this.collapsibleFilter.onchange = () => { this.saveFilters(); this.debouncedApplyFilters(); };

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
        
        // Initial sync of label button icons
        if (this.renderer?.detailLevel) {
            const isShown = this.renderer.detailLevel.showLabels;
            if (iconOn) iconOn.classList.toggle("hidden", !isShown);
            if (iconOff) iconOff.classList.toggle("hidden", isShown);
        }
    }

    async loadData() {
        this.loader.classList.remove("hidden");
        try {
            const data = await api.getGraphData();
            this.rawData = data;
            this.applyFilters();
        } catch (e) {
            Utils.showToast("Failed to load graph data", "error");
        } finally {
            this.loader.classList.add("hidden");
        }
    }

    debouncedApplyFilters() {
        if (this.filterTimer) clearTimeout(this.filterTimer);
        
        const bar = document.getElementById("filter-delay-bar");
        const container = document.getElementById("filter-delay-container");
        
        if (container) container.classList.remove("hidden");
        if (this.loader) this.loader.classList.remove("hidden"); // Show main loader early too
        if (bar) {
            bar.style.transition = 'none';
            bar.style.width = '0%';
            // Trigger reflow
            void bar.offsetWidth;
            bar.style.transition = 'width 3000ms linear';
            bar.style.width = '100%';
        }

        this.filterTimer = setTimeout(() => {
            this.applyFilters();
            if (container) container.classList.add("hidden");
            this.filterTimer = null;
        }, 3000);
    }

    applyFilters() {
        if (!this.rawData) return;
        if (this.loader) this.loader.classList.remove("hidden"); // Show loader when filtering starts

        const minComp = parseInt(this.minCompFilter.value);
        const maxComp = parseInt(this.maxCompFilter.value);
        const minChain = parseInt(this.minChainFilter.value);
        const maxChain = parseInt(this.maxChainFilter.value);
        
        const minCompCount = this.minCompCountFilter ? parseInt(this.minCompCountFilter.value) : 0;
        const maxCompCount = this.maxCompCountFilter ? parseInt(this.maxCompCountFilter.value) : 10;
        const collapsibleMode = this.collapsibleFilter ? this.collapsibleFilter.value : 'all';
        
        const maxLimit = ChainMapUI.config.filters.sliderMax;

        // Save to localStorage
        localStorage.setItem("chainmap_minComp", minComp);
        localStorage.setItem("chainmap_maxComp", maxComp);
        localStorage.setItem("chainmap_minChain", minChain);
        localStorage.setItem("chainmap_maxChain", maxChain);
        localStorage.setItem("chainmap_minCompCount", minCompCount);
        localStorage.setItem("chainmap_maxCompCount", maxCompCount);
        localStorage.setItem("chainmap_collapsible", collapsibleMode);

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
            let anyNodeMatchesCompCount = false;
            
            members.forEach(nid => {
                const n = nodeMap.get(nid);
                if (n) {
                    if ((n.height || 0) > maxH) maxH = n.height;
                    if (n.is_top) topNodeCount++;
                    
                    const meetsMinC = !useMinCompCount || (n.comparison_count || 0) >= minCompCount;
                    const meetsMaxC = !useMaxCompCount || (n.comparison_count || 0) <= maxCompCount;
                    if (meetsMinC && meetsMaxC) anyNodeMatchesCompCount = true;
                }
            });

            const meetsMinChain = !useMinChain || maxH >= minChain;
            const meetsMaxChain = !useMaxChain || maxH <= maxChain;
            
            const isCollapsible = topNodeCount >= 2;
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
        setTimeout(() => this.resetView(), 100); // Auto reset view after filtering
    }

    render(nodes, links) {
        if (this.chainSim) this.chainSim.stop();

        this.statNodes.textContent = nodes.length;
        this.statEdges.textContent = links.length;
        const distinctComponents = new Set(nodes.map(n => String(n.component ?? "")));
        const totalComparisons = nodes.reduce((sum, n) => sum + (n.comparison_count || 0), 0);
        
        if (this.statComparisons) this.statComparisons.textContent = totalComparisons.toLocaleString();
        if (this.statComponents) this.statComponents.textContent = distinctComponents.size;

        // Bottom stats
        const nodesBottom = document.getElementById("stat-nodes-bottom");
        const edgesBottom = document.getElementById("stat-edges-bottom");
        const compsBottom = document.getElementById("stat-comparisons-bottom");
        const componentsBottom = document.getElementById("stat-components-bottom");
        
        if (nodesBottom) nodesBottom.textContent = nodes.length.toLocaleString();
        if (edgesBottom) edgesBottom.textContent = links.length.toLocaleString();
        if (compsBottom) compsBottom.textContent = totalComparisons.toLocaleString();
        if (componentsBottom) componentsBottom.textContent = distinctComponents.size.toLocaleString();

        // Clone for simulation
        // Custom color scale matching legend: High (>0.6) Green, Neutral (0.4-0.6) Yellow, Low (<0.4) Red
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
        
        // Link sources/targets must match node objects for D3 Force
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
            profile: ChainMapUI.config.visuals,
            selectedIds: this.selectedNodes,
            world: this.calculateWorldBounds(simNodes)
        });

        if (this.loader) this.loader.classList.add("hidden");
        
        if (this.chainSim) {
            this.chainSim.initialize();
            this.chainSim.play();
        }
    }

    toggleSimulation() {
        if (this.chainSim.isPaused) {
            this.chainSim.play();
            this.playIcon.classList.add("hidden");
            this.pauseIcon.classList.remove("hidden");
        } else {
            this.chainSim.pause();
            this.playIcon.classList.remove("hidden");
            this.pauseIcon.classList.add("hidden");
        }
    }

    resetView() {
        if (!this.rawData || !this.chainSim || !this.chainSim.nodes.length) {
             this.svg.transition().duration(ChainMapUI.config.interaction.transitionDuration).call(
                this.zoom.transform,
                d3.zoomIdentity.translate(this.width/2, this.height/2).scale(0.05)
            );
            return;
        }

        // Center on the average position of filtered nodes
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

        this.svg.transition().duration(ChainMapUI.config.interaction.transitionDuration).call(
            this.zoom.transform,
            d3.zoomIdentity
                .translate(this.width/2, this.height/2)
                .scale(scale)
                .translate(-centerX, -centerY)
        );
    }

    showTooltip(event, node) {
        this.tooltip.classList.remove("hidden");
        this.tooltip.style.left = `${event.pageX + 10}px`;
        this.tooltip.style.top = `${event.pageY + 10}px`;
        this.tooltip.innerHTML = `
            <div class="font-bold mb-1">${node.id.split('/').pop()}</div>
            <div>Score: ${node.score.toFixed(3)}</div>
            <div>Chain: ${node.height || node.chain_length || 0}</div>
            <div class="text-[9px] text-gray-500 mt-1">Click for details</div>
        `;
    }

    hideTooltip() {
        this.tooltip.classList.add("hidden");
    }

    showNodeDetails(d) {
        this.nodeDetails.classList.remove("hidden");
        const filename = d.id.split('/').pop();
        this.nodeDetails.innerHTML = `
            <div class="flex justify-between items-start mb-2">
                <h3 class="text-white font-bold text-[10px] truncate max-w-[80%]">${filename}</h3>
                <button onclick="document.getElementById('node-details').classList.add('hidden')" class="text-gray-500 hover:text-white p-1">&times;</button>
            </div>
            <div class="flex flex-col gap-2">
                <img src="/images/${encodeURIComponent(d.id)}" class="w-full h-32 md:h-48 object-contain rounded bg-black/40 border border-white/5" onerror="this.src='/output/ranked/${encodeURIComponent(d.id)}'">
                
                <div class="flex flex-wrap gap-x-3 gap-y-1 text-[9px] text-gray-300 justify-center">
                    <span class="flex items-center gap-1">Score: <b class="text-purple-400">${d.score.toFixed(3)}</b></span>
                    <span class="flex items-center gap-1">Chain: <b class="text-purple-400">${d.height || d.chain_length || 0}</b></span>
                    <span class="flex items-center gap-1">Comp: <b class="text-purple-400">${d.component_size || 0}</b></span>
                </div>

                <div class="flex gap-2">
                    <button onclick="window.chainMapUI.toggleSelection('${d.id}')" class="flex-1 py-1.5 bg-pink-600 hover:bg-pink-500 rounded text-[9px] font-bold transition">
                        Select
                    </button>
                    <button onclick="window.chainMapUI.focusNode('${d.id}')" class="px-3 py-1.5 bg-white/10 hover:bg-white/20 rounded text-[9px] transition">
                        Focus
                    </button>
                </div>
            </div>
        `;
    }

    focusNode(id) {
        const node = this.chainSim.nodes.find(n => n.id === id);
        if (!node) return;
        
        this.svg.transition().duration(750).call(
            this.zoom.transform,
            d3.zoomIdentity
                .translate(this.width/2, this.height/2)
                .scale(1.5)
                .translate(-node.x, -node.y)
        );
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

    toggleSelection(id) {
        const idx = this.selectedNodes.indexOf(id);
        if (idx > -1) this.selectedNodes.splice(idx, 1);
        else {
            if (this.selectedNodes.length >= 2) this.selectedNodes.shift();
            this.selectedNodes.push(id);
        }
        this.updateSelectionUI();
        this.renderer.updateSelection(this.selectedNodes);
    }

    updateSelectionUI() {
        const count = this.selectedNodes.length;
        this.selectedCountEl.textContent = count;
        this.compareSelectedBtn.classList.toggle("hidden", count < 2);
    }

    compareSelected() {
        const [left, right] = this.selectedNodes;
        window.location.hash = `#compare?left=${encodeURIComponent(left)}&right=${encodeURIComponent(right)}`;
    }
}

window.chainMapUI = new ChainMapUI();
