// Logic for chain map visualization using D3.js

class ChainMapUI {
    static config = {
        visuals: {
            nodeMinRadius: 3,
            nodeConfidenceScale: 30,
            scoreColors: ['#ef4444', '#eab308', '#22c55e'],
            scoreDomain: [0, 0.5, 1],
            highlightColor: '#f472b6'
        },
        interaction: {
            zoomExtent: [0.001, 2],
            interactThreshold: 0.1,
            transitionDuration: 750,
            zoomToScale: 2
        },
        filters: {
            minLengthSliderMax: 10
        }
    };

    constructor() {
        console.log('ChainMapUI initializing...');
        this.container = document.getElementById('graph-container');
        if (!this.container) {
            console.error('Critical: #graph-container not found');
            return;
        }

        this.loader = document.getElementById('loader');
        this.tooltip = document.getElementById('tooltip');
        this.refreshBtn = document.getElementById('refresh-btn');
        this.resetViewBtn = document.getElementById('reset-view-btn');
        this.minLengthFilter = document.getElementById('min-length-filter');
        this.playPauseBtn = document.getElementById('play-pause-btn');
        this.playIcon = document.getElementById('play-icon');
        this.pauseIcon = document.getElementById('pause-icon');

        this.statNodes = document.getElementById('stat-nodes');
        this.statEdges = document.getElementById('stat-edges');
        this.statChains = document.getElementById('stat-chains');

        this.zoomInBtn = document.getElementById('zoom-in-btn');
        this.zoomOutBtn = document.getElementById('zoom-out-btn');
        this.zoomScaleEl = document.getElementById('zoom-scale');
        this.nodeDetails = document.getElementById('node-details');
        this.minLengthVal = document.getElementById('min-length-val');
        this.compareSelectedBtn = document.getElementById('compare-selected-btn');
        this.selectedCountEl = document.getElementById('selected-count');

        this.width = this.container.clientWidth || 800;
        this.height = this.container.clientHeight || 600;

        this.rawData = null;
        this.chainSim = null;
        this.selectedNodes = [];

        if (typeof d3 === 'undefined') {
            console.error('Critical: D3 library not loaded');
            Utils.showToast('D3 library failed to load', 'error');
            return;
        }

        this.svg = d3.select('#graph-container')
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', [0, 0, this.width, this.height]);

        this.g = this.svg.append('g');
        this.zoom = d3.zoom()
            .extent([[0, 0], [this.width, this.height]])
            .scaleExtent(ChainMapUI.config.interaction.zoomExtent)
            .filter((event) => {
                if (event.type === 'click' || event.type === 'dblclick') return false;
                if (event.button === 0 && event.target.tagName === 'circle') return false;
                return true;
            })
            .on('zoom', (event) => this.handleZoom(event));
        this.svg.call(this.zoom);

        this.defs = this.svg.append('defs');
        this.setupMarker('arrowhead-mid', '#999', 10, 0.8);
        this.setupMarker('arrowhead-legend', '#666', 12, 1);
        this.resetView();
        this.init();
    }

    setupMarker(id, fill, size, opacity) {
        this.defs.append('marker')
            .attr('id', id)
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 5)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', size)
            .attr('markerHeight', size)
            .append('path')
            .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
            .attr('fill', fill)
            .style('opacity', opacity);
    }

    async init() {
        this.refreshBtn.addEventListener('click', () => this.loadData());
        this.resetViewBtn.addEventListener('click', () => this.resetView(true));

        const savedMinLen = localStorage.getItem('chainMap_minLen');
        if (savedMinLen) {
            this.minLengthFilter.value = savedMinLen;
            this.minLengthVal.textContent = savedMinLen;
        }

        this.minLengthFilter.addEventListener('change', () => {
            this.minLengthVal.textContent = this.minLengthFilter.value;
            localStorage.setItem('chainMap_minLen', this.minLengthFilter.value);
            this.applyFilters();
        });

        this.minLengthFilter.addEventListener('input', () => {
            this.minLengthVal.textContent = this.minLengthFilter.value;
        });

        this.playPauseBtn.addEventListener('click', () => this.toggleSimulation());

        if (this.compareSelectedBtn) {
            this.compareSelectedBtn.addEventListener('click', () => this.compareSelected());
        }

        if (this.zoomInBtn) this.zoomInBtn.addEventListener('click', () => this.zoomBy(1.5));
        if (this.zoomOutBtn) this.zoomOutBtn.addEventListener('click', () => this.zoomBy(0.66));

        this.container.addEventListener('click', (e) => {
            if (e.target === this.container || e.target.tagName === 'svg') {
                this.nodeDetails.classList.add('hidden');
            }
        });

        window.addEventListener('resize', () => {
            this.width = this.container.clientWidth;
            this.height = this.container.clientHeight;
            this.svg.attr('viewBox', [0, 0, this.width, this.height]);
        });

        await this.loadData();
    }

    async loadData() {
        console.log('Loading graph data...');
        this.loader.classList.remove('hidden');
        try {
            const data = await api.getGraphData();
            console.log('Graph data received:', data);
            this.rawData = data;

            this.minLengthFilter.max = ChainMapUI.config.filters.minLengthSliderMax;

            this.applyFilters();
        } catch (e) {
            console.error('Failed to load graph data:', e);
            Utils.showToast('Failed to load graph data', 'error');
        } finally {
            this.loader.classList.add('hidden');
        }
    }

    applyFilters() {
        if (!this.rawData) return;

        const minLen = parseInt(this.minLengthFilter.value) || 1;

        const validComponents = new Set(
            Object.entries(this.rawData.components || {})
                .filter(([id, members]) => members.length >= minLen)
                .map(([id]) => id.toString())
        );

        const filteredNodes = (this.rawData.nodes || []).filter(n => {
            const compId = (n.component !== undefined && n.component !== null) ? n.component.toString() : "";
            return validComponents.has(compId);
        });

        const nodeIds = new Set(filteredNodes.map(n => n.id));
        const filteredEdges = (this.rawData.edges || this.rawData.links || []).filter(e =>
            nodeIds.has(e.source) && nodeIds.has(e.target)
        );

        this.render({
            nodes: filteredNodes,
            edges: filteredEdges,
            componentCount: validComponents.size,
            componentList: validComponents
        });
    }

    resetView(animate = false) {
        const targetScale = 0.1;
        const centerX = -ChainSimulation.defaults.simWidth / 4;
        const centerY = -ChainSimulation.defaults.simHeight / 4;
        const animation = animate ? ChainMapUI.config.interaction.transitionDuration : 0;

        this.svg.transition()
            .duration(animation)
            .call(
                this.zoom.transform,
                d3.zoomIdentity
                    //.translate(0, 0) // 1. Move to center of container
                    .scale(targetScale)           // 2. Scale
                    .translate(centerX, centerY)            // 3. Optional: point in map to center on
                //.translate(0, 0)
            );

        this.nodeDetails.classList.add('hidden');
    }

    zoomBy(factor) {
        this.svg.transition().duration(300).call(this.zoom.scaleBy, factor);
    }

    handleZoom(event) {
        this.g.attr('transform', event.transform);
        this.updateZoomScale(event.transform.k);

        const nodeGroup = this.g.select('.nodes');
        if (!nodeGroup.empty()) {
            nodeGroup.classed('no-interact', event.transform.k < ChainMapUI.config.interaction.interactThreshold);
        }
    }

    updateZoomScale(k) {
        if (this.zoomScaleEl) {
            this.zoomScaleEl.textContent = k.toFixed(2) + 'x';
        }
    }

    toggleSimulation() {
        if (!this.chainSim) return;
        if (this.chainSim.isPaused) {
            this.chainSim.play();
            this.playIcon.classList.add('hidden');
            this.pauseIcon.classList.remove('hidden');
        } else {
            this.chainSim.pause();
            this.playIcon.classList.remove('hidden');
            this.pauseIcon.classList.add('hidden');
        }
    }

    render(data) {
        if (this.chainSim) this.chainSim.stop();
        this.g.selectAll('*').remove();

        const nodes = data.nodes.map(d => ({ ...d }));
        const links = data.edges.map(d => ({ ...d }));
        const components = data.componentList;

        this.statNodes.textContent = nodes.length;
        this.statEdges.textContent = links.length;
        this.statChains.textContent = data.componentCount;

        if (nodes.length === 0) {
            this.g.append('text')
                .attr('x', this.width / 2)
                .attr('y', this.height / 2)
                .attr('text-anchor', 'middle')
                .attr('fill', '#666')
                .text('No chains match the current filters.');
            return;
        }

        this.chainSim = new ChainSimulation(nodes, links, components, {
            width: this.width,
            height: this.height,
            onTick: (simNodes, simLinks) => this.updateRender(simNodes, simLinks),
            onEnd: () => this.onSimulationEnd()
        });

        this.chainSim.initialize();

        const link = this.g.append('g')
            .attr('class', 'links')
            .selectAll('path')
            .data(links)
            .join('path')
            .attr('class', 'link')
            .attr('fill', 'none')
            .attr('stroke', '#555')
            .attr('stroke-opacity', 0.4)
            .attr('marker-mid', nodes.length < 2000 ? 'url(#arrowhead-mid)' : null);

        const node = this.g.append('g')
            .attr('class', 'nodes')
            .selectAll('g')
            .data(nodes)
            .join('g')
            .attr('class', 'node')
            .call(this.drag());

        const customColor = d3.scaleLinear()
            .domain(ChainMapUI.config.visuals.scoreDomain)
            .range(ChainMapUI.config.visuals.scoreColors)
            .interpolate(d3.interpolateHcl);

        node.append('circle')
            .attr('r', d => ChainMapUI.config.visuals.nodeMinRadius + (d.confidence * ChainMapUI.config.visuals.nodeConfidenceScale))
            .attr('fill', d => customColor(d.score))
            .attr('stroke', '#fff')
            .attr('stroke-width', nodes.length < 5000 ? 1.5 : 0.5)
            .style('cursor', 'pointer')
            .on('mouseover', (event, d) => this.showTooltip(event, d))
            .on('mousemove', (event) => this.updateTooltipPos(event))
            .on('mouseout', () => this.tooltip.classList.add('hidden'))
            .on('click', (event, d) => this.handleNodeClick(event, d));

        if (nodes.length < 1000) {
            node.append('text')
                .attr('x', 12)
                .attr('y', 4)
                .text(d => d.id.split('_').pop().slice(0, 15))
                .style('font-size', '8px')
                .style('opacity', '0.7');
        }

        this.updateUIState(nodes.length);
    }

    updateRender(nodes, links) {
        this.g.select('.links').selectAll('path').attr('d', d => {
            const mx = (d.source.x + d.target.x) / 2;
            const my = (d.source.y + d.target.y) / 2;
            return `M${d.source.x},${d.source.y} L${mx},${my} L${d.target.x},${d.target.y}`;
        });

        this.g.select('.nodes').selectAll('g')
            .attr('transform', d => `translate(${d.x},${d.y})`);
    }

    onSimulationEnd() {
        console.log('Simulation settled');
    }

    updateUIState(nodeCount) {
        if (nodeCount > 1000) {
            this.chainSim.pause();
            this.playIcon.classList.remove('hidden');
            this.pauseIcon.classList.add('hidden');
        } else {
            this.playIcon.classList.add('hidden');
            this.pauseIcon.classList.remove('hidden');
        }
    }

    showTooltip(event, d) {
        if (!this.nodeDetails.classList.contains('hidden')) return;
        this.tooltip.classList.remove('hidden');
        this.tooltip.innerHTML = `
            <div class="font-bold text-white mb-1">${d.id.split('/').pop()}</div>
            <div class="text-purple-300 text-xs">Score: ${d.score.toFixed(3)}</div>
            <div class="text-blue-300 text-xs">Confidence: ${d.confidence.toFixed(3)}</div>
        `;
        this.updateTooltipPos(event);
    }

    updateTooltipPos(event) {
        this.tooltip.style.left = (event.pageX + 10) + 'px';
        this.tooltip.style.top = (event.pageY + 10) + 'px';
    }

    handleNodeClick(event, d) {
        event.stopPropagation();
        if (event.shiftKey || event.ctrlKey || event.metaKey) {
            this.toggleNodeSelection(d.id);
        } else {
            this.showNodeDetails(d);
        }
    }

    drag() {
        return d3.drag()
            .on('start', (event) => {
                const transform = d3.zoomTransform(this.svg.node());
                if (transform.k < ChainMapUI.config.interaction.interactThreshold) return;
                if (!event.active && this.chainSim && !this.chainSim.isPaused) this.chainSim.simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            })
            .on('drag', (event) => {
                const transform = d3.zoomTransform(this.svg.node());
                if (transform.k < ChainMapUI.config.interaction.interactThreshold) return;
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            })
            .on('end', (event) => {
                if (!event.active && this.chainSim && !this.chainSim.isPaused) this.chainSim.simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            });
    }

    showNodeDetails(d) {
        this.tooltip.classList.add('hidden');
        this.nodeDetails.classList.remove('hidden');

        const filename = d.id;
        const imageUrl = `/images/${encodeURIComponent(filename)}?score=${d.score}`;

        this.nodeDetails.innerHTML = `
            <div class="flex justify-between items-start mb-2">
                <h3 class="text-white font-bold text-[10px] truncate pr-2" title="${filename}">${filename.split('/').pop()}</h3>
                <button onclick="document.getElementById('node-details').classList.add('hidden')" class="text-gray-500 hover:text-white text-lg leading-none">&times;</button>
            </div>
            <div class="aspect-square bg-black/40 rounded-lg overflow-hidden mb-2 border border-white/5">
                <img src="${imageUrl}" class="w-full h-full object-contain" onerror="this.src='/api/v2/placeholder'"/>
            </div>
            <div class="grid grid-cols-2 gap-1.5 text-[9px]">
                <div class="bg-purple-500/10 p-1.5 rounded">
                    <div class="text-purple-400 uppercase font-bold">Score</div>
                    <div class="text-white text-xs font-mono">${d.score.toFixed(4)}</div>
                </div>
                <div class="bg-blue-500/10 p-1.5 rounded">
                    <div class="text-blue-400 uppercase font-bold">Conf</div>
                    <div class="text-white text-xs font-mono">${d.confidence.toFixed(4)}</div>
                </div>
            </div>
            <div class="mt-2 pt-2 border-t border-white/5 flex gap-2">
                <button onclick="window.chainMapUI.toggleNodeSelection(decodeURIComponent('${encodeURIComponent(filename)}'))" class="flex-1 text-center py-1 bg-pink-600 hover:bg-pink-500 text-white rounded text-[9px] font-bold transition">Select</button>
                <button onclick="window.chainMapUI.zoomToNode(decodeURIComponent('${encodeURIComponent(filename)}'))" class="flex-1 text-center py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-[9px] font-bold transition">Focus</button>
            </div>
        `;
    }

    zoomToNode(nodeId) {
        if (!this.chainSim) return;
        const simNode = this.chainSim.getSimNode(nodeId);
        if (!simNode) return;

        this.svg.transition().duration(ChainMapUI.config.interaction.transitionDuration).call(
            this.zoom.transform,
            d3.zoomIdentity.translate(this.width / 2, this.height / 2).scale(ChainMapUI.config.interaction.zoomToScale).translate(-simNode.x, -simNode.y)
        );
    }

    toggleNodeSelection(nodeId) {
        const index = this.selectedNodes.indexOf(nodeId);
        if (index > -1) {
            this.selectedNodes.splice(index, 1);
        } else {
            if (this.selectedNodes.length >= 2) {
                this.selectedNodes.shift();
            }
            this.selectedNodes.push(nodeId);
        }
        this.updateSelectionUI();
    }

    updateSelectionUI() {
        this.g.selectAll('.node circle')
            .classed('highlight', d => this.selectedNodes.includes(d.id))
            .attr('stroke-width', d => this.selectedNodes.includes(d.id) ? 3 : 1.5)
            .attr('stroke', d => this.selectedNodes.includes(d.id) ? ChainMapUI.config.visuals.highlightColor : '#fff');

        if (this.selectedNodes.length === 2) {
            this.compareSelectedBtn.classList.remove('hidden');
            this.selectedCountEl.textContent = this.selectedNodes.length;
        } else {
            this.compareSelectedBtn.classList.add('hidden');
            this.selectedCountEl.textContent = this.selectedNodes.length;
        }

        if (this.selectedNodes.length === 1) {
            Utils.showToast('Selected one image. Shift+Click another to compare.', 'info');
        }
    }

    compareSelected() {
        if (this.selectedNodes.length !== 2) return;
        const [left, right] = this.selectedNodes;
        window.location.href = `/?left=${encodeURIComponent(left)}&right=${encodeURIComponent(right)}`;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.chainMapUI = new ChainMapUI();
});