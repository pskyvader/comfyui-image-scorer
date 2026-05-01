// Logic for chain map visualization using D3.js

class ChainMap {
    static config = {
        simulation: {
            linkDistance: 100,       // Base distance between connected nodes
            linkDistanceMin: 50,      // Line length for nodes with few connections
            linkDistanceMax: 200,     // Line length for nodes with many connections
            useDynamicDistance: true, // If true, lines are LONGER for hubs and SHORTER for isolated pairs
            linkStrength: 0.1,        // Base stiffness of the lines
            scoreSimilarityStrength: 0.9, // Extra attraction between connected nodes with similar scores (0 to 1)
            chargeStrength: -300,     // Base repulsion for all nodes
            disconnectedChargeMultiplier: 2.0, // Multiplier for repulsion on nodes with no connections
            chargeDistanceMax: 1000,  // Maximum distance over which nodes repel each other
            componentRepulsion: 0.7,  // Extra force to push separate chains away from each other (0 to 1)
            collisionRadius: 25,      // Minimum gap between node centers to prevent overlap
            forceXStrength: 0.03,     // Gravity toward the horizontal center
            forceYStrength: 0.5,      // Gravity toward the vertical position (Higher scores = higher up, lower = lower down)
            velocityDecay: 0.2,       // Friction (0 to 1). Higher = slower movement
            alphaDecay: 0.02,         // How fast the simulation cools down and stops
            yMargin: 0.01,             // Vertical padding (%) where images won't be placed
            scoreSpread: 2.0          // Push scores toward the poles (1.0 = linear, > 1.0 = more extreme separation)
        },
        visuals: {
            nodeMinRadius: 2,         // Base size of node circles
            nodeConfidenceScale: 20,  // Multiplier for confidence-based sizing
            scoreColors: ['#ef4444', '#eab308', '#22c55e'], // Red, Yellow, Green gradient
            scoreDomain: [0, 0.5, 1], // Thresholds for the color gradient
            highlightColor: '#f472b6' // Border color for selected nodes
        },
        interaction: {
            zoomExtent: [0.001, 2],   // [min, max] zoom levels
            interactThreshold: 0.2,   // Zoom level below which dragging/tooltips are disabled
            transitionDuration: 750,  // Animation speed (ms) for view resets
            zoomToScale: 2            // Target zoom level when focusing on a node
        }
    };

    constructor() {
        console.log('ChainMap initializing...');
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
        this.simulation = null;
        this.isPaused = false;
        this.selectedNodes = []; // Track selected node IDs for comparison

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

        // Add zoom behavior
        this.g = this.svg.append('g');
        this.zoom = d3.zoom()
            .extent([[0, 0], [this.width, this.height]])
            .scaleExtent(ChainMap.config.interaction.zoomExtent)
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
                this.updateZoomScale(event.transform.k);

                // Toggle node interaction based on zoom level (Point 2)
                const nodeGroup = this.g.select('.nodes');
                if (!nodeGroup.empty()) {
                    nodeGroup.classed('no-interact', event.transform.k < ChainMap.config.interaction.interactThreshold);
                }
            });
        this.svg.call(this.zoom);

        // Add arrowhead marker definition (for middle of line)
        this.svg.append('defs').append('marker')
            .attr('id', 'arrowhead-mid')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 5)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 5)
            .attr('markerHeight', 5)
            .append('path')
            .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
            .attr('fill', '#999')
            .style('opacity', '0.8');

        // Legend colors matching marker
        this.svg.append('defs').append('marker')
            .attr('id', 'arrowhead-legend')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 5)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .append('path')
            .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
            .attr('fill', '#666');

        this.init();
    }

    async init() {
        this.refreshBtn.addEventListener('click', () => this.loadData());
        this.resetViewBtn.addEventListener('click', () => this.resetView());

        // Restore from localStorage
        const savedMinLen = localStorage.getItem('chainMap_minLen');
        if (savedMinLen) {
            this.minLengthFilter.value = savedMinLen;
            this.minLengthVal.textContent = savedMinLen;
        }

        // Update on release (change event)
        this.minLengthFilter.addEventListener('change', () => {
            this.minLengthVal.textContent = this.minLengthFilter.value;
            localStorage.setItem('chainMap_minLen', this.minLengthFilter.value);
            this.applyFilters();
        });

        // Live feedback for value but no render
        this.minLengthFilter.addEventListener('input', () => {
            this.minLengthVal.textContent = this.minLengthFilter.value;
        });

        this.playPauseBtn.addEventListener('click', () => this.toggleSimulation());

        if (this.compareSelectedBtn) {
            this.compareSelectedBtn.addEventListener('click', () => this.compareSelected());
        }

        if (this.zoomInBtn) this.zoomInBtn.addEventListener('click', () => this.zoomBy(1.5));
        if (this.zoomOutBtn) this.zoomOutBtn.addEventListener('click', () => this.zoomBy(0.66));

        // Close details on click outside
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

            // Dynamic slider max based on largest component
            const componentSizes = Object.values(this.rawData.components).map(m => m.length);
            const maxLen = Math.max(10, ...componentSizes);
            this.minLengthFilter.max = maxLen;

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

        // Filter components by size - Using strings for robust matching
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
            componentCount: validComponents.size
        });
    }

    resetView() {
        this.svg.transition().duration(ChainMap.config.interaction.transitionDuration).call(
            this.zoom.transform,
            d3.zoomIdentity
        );
        this.nodeDetails.classList.add('hidden');
    }

    zoomBy(factor) {
        this.svg.transition().duration(300).call(this.zoom.scaleBy, factor);
    }

    updateZoomScale(k) {
        if (this.zoomScaleEl) {
            this.zoomScaleEl.textContent = k.toFixed(2) + 'x';
        }
    }

    toggleSimulation() {
        if (!this.simulation) return;
        this.isPaused = !this.isPaused;
        if (this.isPaused) {
            this.simulation.stop();
            this.playIcon.classList.remove('hidden');
            this.pauseIcon.classList.add('hidden');
        } else {
            this.simulation.alpha(0.3).restart();
            this.playIcon.classList.add('hidden');
            this.pauseIcon.classList.remove('hidden');
        }
    }

    render(data) {
        if (this.simulation) this.simulation.stop();
        this.g.selectAll('*').remove();

        const nodes = data.nodes.map(d => ({ ...d }));
        const links = data.edges.map(d => ({ ...d }));

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

        // Calculate node degree (number of connections) for dynamic distance
        const nodeDegree = {};
        links.forEach(l => {
            nodeDegree[l.source] = (nodeDegree[l.source] || 0) + 1;
            nodeDegree[l.target] = (nodeDegree[l.target] || 0) + 1;
        });

        this.simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links)
                .id(d => d.id)
                .distance(d => {
                    if (!ChainMap.config.simulation.useDynamicDistance) return ChainMap.config.simulation.linkDistance;

                    // More connections = longer lines
                    const degA = nodeDegree[d.source.id || d.source] || 1;
                    const degB = nodeDegree[d.target.id || d.target] || 1;
                    const avgDeg = (degA + degB) / 2;

                    const maxDeg = 10;
                    const factor = Math.min(1, (avgDeg - 1) / (maxDeg - 1));
                    return ChainMap.config.simulation.linkDistanceMin + (factor * (ChainMap.config.simulation.linkDistanceMax - ChainMap.config.simulation.linkDistanceMin));
                })
                .strength(d => {
                    const base = ChainMap.config.simulation.linkStrength;
                    const bonus = ChainMap.config.simulation.scoreSimilarityStrength;
                    if (bonus <= 0) return base;
                    const scoreDiff = Math.abs((d.source.score || 0) - (d.target.score || 0));
                    const similarity = Math.max(0, 1 - scoreDiff);
                    return base + (similarity * bonus);
                }))
            .force('charge', d3.forceManyBody()
                .strength(d => {
                    const base = ChainMap.config.simulation.chargeStrength;
                    const deg = nodeDegree[d.id] || 0;
                    return deg === 0 ? base * ChainMap.config.simulation.disconnectedChargeMultiplier : base;
                })
                .distanceMax(ChainMap.config.simulation.chargeDistanceMax))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('x', d3.forceX(this.width / 2).strength(ChainMap.config.simulation.forceXStrength))
            .force('y', d3.forceY(d => {
                const margin = this.height * ChainMap.config.simulation.yMargin;
                const spread = ChainMap.config.simulation.scoreSpread || 1.0;

                // Stretch the score around the 0.5 center
                let s = d.score;
                if (spread !== 1.0) {
                    s = 0.5 + (s - 0.5) * spread;
                    s = Math.max(0, Math.min(1, s));
                }

                return margin + (1 - s) * (this.height - margin * 2);
            }).strength(ChainMap.config.simulation.forceYStrength));

        if (nodes.length < 5000) {
            this.simulation.force('collision', d3.forceCollide().radius(ChainMap.config.simulation.collisionRadius));
            this.simulation.force('component', this.getComponentForce());
        }

        // For all graphs, remove the horizontal center pull to allow clusters to spread out freely
        this.simulation.force('x', null);

        this.simulation
            .velocityDecay(ChainMap.config.simulation.velocityDecay)
            .alphaDecay(ChainMap.config.simulation.alphaDecay);

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
            .call(this.drag(this.simulation));

        const customColor = d3.scaleLinear()
            .domain(ChainMap.config.visuals.scoreDomain)
            .range(ChainMap.config.visuals.scoreColors)
            .interpolate(d3.interpolateHcl);

        node.append('circle')
            .attr('r', d => ChainMap.config.visuals.nodeMinRadius + (d.confidence * ChainMap.config.visuals.nodeConfidenceScale))
            .attr('fill', d => customColor(d.score))
            .attr('stroke', '#fff')
            .attr('stroke-width', nodes.length < 5000 ? 1.5 : 0.5)
            .style('cursor', 'pointer')
            .on('mouseover', (event, d) => {
                if (nodes.length > 10000 || !this.nodeDetails.classList.contains('hidden')) return;
                this.tooltip.classList.remove('hidden');
                this.tooltip.innerHTML = `
                    <div class="font-bold text-white mb-1">${d.id.split('/').pop()}</div>
                    <div class="text-purple-300 text-xs">Score: ${d.score.toFixed(3)}</div>
                    <div class="text-blue-300 text-xs">Confidence: ${d.confidence.toFixed(3)}</div>
                `;
                this.updateTooltipPos(event);
            })
            .on('mousemove', (event) => this.updateTooltipPos(event))
            .on('mouseout', () => this.tooltip.classList.add('hidden'))
            .on('click', (event, d) => {
                event.stopPropagation();
                if (event.shiftKey || event.ctrlKey || event.metaKey) {
                    this.toggleNodeSelection(d.id);
                } else {
                    this.showNodeDetails(d);
                }
            });

        // Skip labels for large graphs
        if (nodes.length < 500) {
            node.append('text')
                .attr('x', 12)
                .attr('y', 4)
                .text(d => d.id.split('_').pop().slice(0, 15))
                .style('font-size', '8px')
                .style('opacity', '0.7');
        }

        this.simulation.on('tick', () => {
            link.attr('d', d => {
                const mx = (d.source.x + d.target.x) / 2;
                const my = (d.source.y + d.target.y) / 2;
                return `M${d.source.x},${d.source.y} L${mx},${my} L${d.target.x},${d.target.y}`;
            });

            node
                .attr('transform', d => `translate(${d.x},${d.y})`);
        });

        // Initial UI state - Auto-pause if too many nodes
        if (nodes.length > 1000) {
            this.isPaused = true;
        }

        if (this.isPaused) {
            this.simulation.stop();
            this.playIcon.classList.remove('hidden');
            this.pauseIcon.classList.add('hidden');
        } else {
            this.playIcon.classList.add('hidden');
            this.pauseIcon.classList.remove('hidden');
        }
    }

    getComponentForce() {
        let nodes;
        const force = (alpha) => {
            if (ChainMap.config.simulation.componentRepulsion <= 0) return;

            const centers = {};
            const counts = {};
            nodes.forEach(n => {
                const c = n.component;
                if (!centers[c]) { centers[c] = { x: 0, y: 0 }; counts[c] = 0; }
                centers[c].x += n.x;
                centers[c].y += n.y;
                counts[c]++;
            });

            const compIds = Object.keys(centers);
            compIds.forEach(c => {
                centers[c].x /= counts[c];
                centers[c].y /= counts[c];
            });

            // Repel centers - Calculate net force per component
            const componentForces = {};
            compIds.forEach(c => componentForces[c] = { x: 0, y: 0 });

            for (let i = 0; i < compIds.length; i++) {
                for (let j = i + 1; j < compIds.length; j++) {
                    const id1 = compIds[i];
                    const id2 = compIds[j];
                    const c1 = centers[id1];
                    const c2 = centers[id2];
                    const dx = c1.x - c2.x;
                    const dy = c1.y - c2.y;
                    const distSq = dx * dx + dy * dy;
                    if (distSq === 0) continue;

                    const dist = Math.sqrt(distSq);
                    const strength = alpha * ChainMap.config.simulation.componentRepulsion * 20000;
                    const f = strength / (distSq + 100);
                    const fx = (dx / dist) * f;
                    const fy = (dy / dist) * f;

                    componentForces[id1].x += fx;
                    componentForces[id1].y += fy;
                    componentForces[id2].x -= fx;
                    componentForces[id2].y -= fy;
                }
            }

            // Apply forces to nodes in one pass
            nodes.forEach(n => {
                const f = componentForces[n.component];
                if (f) {
                    n.vx += f.x;
                    n.vy += f.y;
                }
            });
        };
        force.initialize = (_) => nodes = _;
        return force;
    }

    updateTooltipPos(event) {
        this.tooltip.style.left = (event.pageX + 10) + 'px';
        this.tooltip.style.top = (event.pageY + 10) + 'px';
    }

    drag(simulation) {
        return d3.drag()
            .on('start', (event) => {
                // Stricter dragging: deactivate if scale < threshold
                const transform = d3.zoomTransform(this.svg.node());
                if (transform.k < ChainMap.config.interaction.interactThreshold) return;

                if (!event.active && !this.isPaused) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            })
            .on('drag', (event) => {
                const transform = d3.zoomTransform(this.svg.node());
                if (transform.k < ChainMap.config.interaction.interactThreshold) return;

                event.subject.fx = event.x;
                event.subject.fy = event.y;
            })
            .on('end', (event) => {
                if (!event.active && !this.isPaused) simulation.alphaTarget(0);
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
            <div class="flex justify-between items-start mb-3">
                <h3 class="text-white font-bold text-xs truncate pr-4" title="${filename}">${filename.split('/').pop()}</h3>
                <button onclick="document.getElementById('node-details').classList.add('hidden')" class="text-gray-500 hover:text-white">&times;</button>
            </div>
            <div class="aspect-square bg-black/40 rounded-lg overflow-hidden mb-3 border border-white/5">
                <img src="${imageUrl}" class="w-full h-full object-contain" onerror="this.src='/api/v2/placeholder'"/>
            </div>
            <div class="grid grid-cols-2 gap-2 text-[10px]">
                <div class="bg-purple-500/10 p-2 rounded">
                    <div class="text-purple-400 uppercase font-bold">Score</div>
                    <div class="text-white text-sm font-mono">${d.score.toFixed(4)}</div>
                </div>
                <div class="bg-blue-500/10 p-2 rounded">
                    <div class="text-blue-400 uppercase font-bold">Conf</div>
                    <div class="text-white text-sm font-mono">${d.confidence.toFixed(4)}</div>
                </div>
            </div>
            <div class="mt-3 pt-3 border-t border-white/5 flex gap-2">
                <button onclick="window.chainMap.toggleNodeSelection(decodeURIComponent('${encodeURIComponent(filename)}'))" class="flex-1 text-center py-1.5 bg-pink-600 hover:bg-pink-500 text-white rounded text-[10px] font-bold transition">Select</button>
                <button onclick="window.chainMap.zoomToNode(decodeURIComponent('${encodeURIComponent(filename)}'))" class="flex-1 text-center py-1.5 bg-gray-700 hover:bg-gray-600 text-white rounded text-[10px] font-bold transition">Focus</button>
            </div>
        `;
    }

    zoomToNode(nodeId) {
        const node = this.rawData.nodes.find(n => n.id === nodeId);
        if (!node) return;

        // Find the actual simulated node data
        const simNode = this.simulation.nodes().find(n => n.id === nodeId);
        if (!simNode) return;

        this.svg.transition().duration(ChainMap.config.interaction.transitionDuration).call(
            this.zoom.transform,
            d3.zoomIdentity.translate(this.width / 2, this.height / 2).scale(ChainMap.config.interaction.zoomToScale).translate(-simNode.x, -simNode.y)
        );
    }

    toggleNodeSelection(nodeId) {
        const index = this.selectedNodes.indexOf(nodeId);
        if (index > -1) {
            this.selectedNodes.splice(index, 1);
        } else {
            if (this.selectedNodes.length >= 2) {
                // Replace the oldest selection or just block?
                // Let's replace the first one
                this.selectedNodes.shift();
            }
            this.selectedNodes.push(nodeId);
        }
        this.updateSelectionUI();
    }

    updateSelectionUI() {
        // Update nodes in SVG
        this.g.selectAll('.node circle')
            .classed('highlight', d => this.selectedNodes.includes(d.id))
            .attr('stroke-width', d => this.selectedNodes.includes(d.id) ? 3 : 1.5)
            .attr('stroke', d => this.selectedNodes.includes(d.id) ? ChainMap.config.visuals.highlightColor : '#fff');

        // Update button
        if (this.selectedNodes.length === 2) {
            this.compareSelectedBtn.classList.remove('hidden');
            this.selectedCountEl.textContent = this.selectedNodes.length;
        } else {
            this.compareSelectedBtn.classList.add('hidden');
            this.selectedCountEl.textContent = this.selectedNodes.length;
        }

        // Show toast for feedback
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
    window.chainMap = new ChainMap();
});
