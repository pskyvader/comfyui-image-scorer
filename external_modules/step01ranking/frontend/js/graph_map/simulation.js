/**
 * Crystal Structure Simulation (Chain-Based Diamond Layout with Phased Reveal)
 * Phases: 0=anchors, 1=nodes+chainLinks, 2=allLinks, 3=relaxation, 4=done
 */

class ChainSimulation {
    constructor(nodes, links, chains, components, options = {}) {
        this.nodes = nodes;
        this.links = links;
        this.chains = Array.isArray(chains) ? chains : [];
        this.components = Array.isArray(components) ? Array.from(components) : [];
        this._tickCount = 0;
        this._simStart = performance.now();

        this.config = { ...PHYSICS.link, ...PHYSICS.node, ...PHYSICS.world, ...RENDER.link, ...CHAIN, ...CONTROLS, ...options };

        this.onTick = options.onTick;
        this.onEnd = options.onEnd;
        this.isPaused = false;

        this.nodeById = new Map();
        this.nodeDegree = new Map();
        this.componentNodes = new Map();
        this.runtime = this.resolveRuntimeProfile();

        this._activeNodeCount = this.nodes.length;
        this._activeNodes = [...this.nodes];
        this._activeLinks = [...this.links];

        this.simulationLinks = this.links;
        this.effectiveWidth = this.config.minWorldSize;
        this.effectiveHeight = this.config.minWorldSize;

        this.hexRadius = 200;
        this.componentPadding = 40;

        // Chain data structures
        this.chainsByComponent = new Map();
        this.nodeToChain = new Map();
        this.nodeChainOrder = new Map();
        this.linkIsChainLink = new Map();
        this.linkMatchedChain = new Map();
        this._chainNodeIds = new Map();

        // Phase tracking
        this._phase = 5;
        this._phaseTimer = null;
        this._relaxTimer = null;
        this._relaxIter = 0;

        // Layout data
        this._componentLayouts = new Map();
        this._chainLayoutByComponent = new Map();
        this._chainInfoPerChain = new Map();
        this._anchorSet = new Set();
        this._anchorCount = 0;
        this._chainLinkCount = 0;
    }

    setVisibleCount(count) {
        return count;
    }

    addNodeBatch(count) {
        return 0;
    }

    // ── Initialization ──────────────────────────────────────────

    initialize() {
        this.indexNodes();
        this.groupNodesByComponent();
        this.calculateNodeDegree();
        this.buildChainData();
        this.prepareWorldBounds();
        this.prepareNodes();
        this.prepareLinks();
        this._activeNodes = [...this.nodes];
        this._activeLinks = [...this.links];
        this._countChainLinks();
        this.setupSimulation();
    }

    // ── Phase Management ────────────────────────────────────────

    start() {
        this._phase = 0;
        this._relaxIter = 0;
        this._advancePhase();
    }

    _advancePhase() {
        if (this.isPaused) return;
        if (this._phase >= 4) {
            this._complete();
            return;
        }

        switch (this._phase) {
            case 0: this._p0_placeAnchors(); this._sortNodes(); break;
            case 1: this._p1_placeChainNodes(); this._sortLinks(); break;
            case 2: break;
            case 3: this._p3_startRelaxation(); return;
        }

        this._emitTick();
        this._phase++;

        this._phaseTimer = setTimeout(() => this._advancePhase(), this._phase < 4 ? this.config.phaseDelay : this.config.relaxDelay);
    }

    _emitTick() {
        if (this.onTick) {
            this.onTick({
                phase: this._phase,
                visibleNodeCount: this._getVisibleNodeCount(),
                visibleLinkCount: this._getVisibleLinkCount(),
                done: false,
            });
        }
    }

    _getVisibleNodeCount() {
        switch (this._phase) {
            case 0: return this._anchorCount;
            case 1: return this.nodes.length;
            case 2: return this.nodes.length;
            case 3: return this.nodes.length;
            default: return this.nodes.length;
        }
    }

    _getVisibleLinkCount() {
        switch (this._phase) {
            case 0: return 0;
            case 1: return this._chainLinkCount;
            case 2: return this.links.length;
            case 3: return this.links.length;
            default: return this.links.length;
        }
    }

    _complete() {
        if (this._phaseTimer) { clearTimeout(this._phaseTimer); this._phaseTimer = null; }
        if (this._relaxTimer) { clearInterval(this._relaxTimer); this._relaxTimer = null; }
        this._phase = 4;
        this.isPaused = this.runtime && this.runtime.startPaused;
        if (this.onTick) {
            this.onTick({
                phase: this._phase,
                visibleNodeCount: this.nodes.length,
                visibleLinkCount: this.links.length,
                done: true,
            });
        }
        if (this.onEnd) this.onEnd(this.runtime);
    }

    // ── Phase 0: Place Anchors ──────────────────────────────────

    _p0_placeAnchors() {
        this._anchorSet.clear();
        this._anchorCount = 0;

        const groups = this._groupNodesByComponentRaw();
        for (const [compId, compNodes] of groups) {
            const layout = this._componentLayouts.get(compId);
            if (!layout) continue;

            let compTopScore = -Infinity;
            let compBottomScore = Infinity;
            for (const n of compNodes) {
                if (n.score > compTopScore) compTopScore = n.score;
                if (n.score < compBottomScore) compBottomScore = n.score;
            }

            const topThresh = compTopScore - this.config.anchorScoreThreshold;
            const bottomThresh = compBottomScore + this.config.anchorScoreThreshold;

            const topAnchors = [];
            const bottomAnchors = [];

            for (const n of compNodes) {
                if (n.score >= topThresh) topAnchors.push(n);
                if (n.score <= bottomThresh) bottomAnchors.push(n);
            }

            const topSpread = Math.min(layout.cellWidth * this.config.topBottomSpread, Math.max(300, topAnchors.length * 60));
            const bottomSpread = Math.min(layout.cellWidth * this.config.topBottomSpread, Math.max(300, bottomAnchors.length * 60));

            const topY = layout.cy - layout.cellHeight / 2 + layout.cellHeight * 0.02;
            topAnchors.forEach((node, i, arr) => {
                const frac = arr.length === 1 ? 0.5 : (i + 0.5) / arr.length;
                node.x = layout.cx + (frac - 0.5) * topSpread;
                node.y = topY;
                node._isAnchor = true;
                node._topX = node.x;
                node._topY = node.y;
                node._botX = node.x;
                node._botY = node.y;
                this._anchorSet.add(node.id);
                this._anchorCount++;
            });

            const bottomY = layout.cy + layout.cellHeight / 2 - layout.cellHeight * 0.02;
            bottomAnchors.forEach((node, i, arr) => {
                const frac = arr.length === 1 ? 0.5 : (i + 0.5) / arr.length;
                node.x = layout.cx + (frac - 0.5) * bottomSpread;
                node.y = bottomY;
                node._isAnchor = true;
                node._topX = node.x;
                node._topY = node.y;
                node._botX = node.x;
                node._botY = node.y;
                this._anchorSet.add(node.id);
                this._anchorCount++;
            });
        }
    }

    // ── Phase 1: Place Chain Nodes ──────────────────────────────

    _p1_placeChainNodes() {
        const groups = this._groupNodesByComponentRaw();

        for (const [compId, compNodes] of groups) {
            const layout = this._componentLayouts.get(compId);
            if (!layout) continue;

            // Compute component score extremes
            let compTopScore = -Infinity;
            let compBottomScore = Infinity;
            for (const n of compNodes) {
                if (n.score > compTopScore) compTopScore = n.score;
                if (n.score < compBottomScore) compBottomScore = n.score;
            }

            // Compute component-level convergence point: center of all anchors
            const topThresh = compTopScore - this.config.anchorScoreThreshold;
            const bottomThresh = compBottomScore + this.config.anchorScoreThreshold;
            const compTopAnchors = [];
            const compBottomAnchors = [];
            for (const n of compNodes) {
                if (this._anchorSet.has(n.id) && n.score >= topThresh) compTopAnchors.push(n);
                if (this._anchorSet.has(n.id) && n.score <= bottomThresh) compBottomAnchors.push(n);
            }

            const convTopX = compTopAnchors.length > 0
                ? compTopAnchors.reduce((s, a) => s + a._topX, 0) / compTopAnchors.length
                : layout.cx;
            const convTopY = compTopAnchors.length > 0
                ? compTopAnchors[0]._topY
                : (layout.cy - layout.cellHeight / 2);
            const convBottomX = compBottomAnchors.length > 0
                ? compBottomAnchors.reduce((s, a) => s + a._botX, 0) / compBottomAnchors.length
                : layout.cx;
            const convBottomY = compBottomAnchors.length > 0
                ? compBottomAnchors[0]._botY
                : (layout.cy + layout.cellHeight / 2);

            const chainLayouts = this._chainLayoutByComponent.get(compId) || [];

            for (const cl of chainLayouts) {
                for (const node of cl.nodes) {
                    // Only position/identify on primary chain to avoid overlap and overwrite
                    // Anchor nodes are exempt so they link all chains at convergence points
                    if (!this._anchorSet.has(node.id) && this.nodeToChain.get(node.id) !== cl.id) continue;

                    if (this._anchorSet.has(node.id)) {
                        // Anchor nodes keep their _p0 position; just store chain metadata
                        node._chainId = cl.id;
                        node._chainProgress = node.score >= cl.topScore ? 0 : 1;
                        node._topX = convTopX;
                        node._topY = convTopY;
                        node._botX = convBottomX;
                        node._botY = convBottomY;
                        continue;
                    }

                    const scoreRange = cl.scoreRange || 0.001;
                    const progress = (cl.topScore - node.score) / scoreRange;
                    const t = Math.min(1, Math.max(0, progress));

                    node._isAnchor = false;
                    node._chainId = cl.id;
                    node._chainProgress = t;

                    this._updateNodeFromProgress(node, cl, layout, convTopX, convTopY, convBottomX, convBottomY);
                }
            }

            // Position unchained nodes at center by score
            for (const node of compNodes) {
                if (node._chainId !== undefined) continue;
                if (this._anchorSet.has(node.id)) continue;
                const topY = layout.cy - layout.cellHeight / 2;
                const bottomY = layout.cy + layout.cellHeight / 2;
                const t = (compTopScore === compBottomScore) ? 0.5 : (compTopScore - node.score) / (compTopScore - compBottomScore);
                node.x = layout.cx;
                node.y = topY + Math.min(1, Math.max(0, t)) * (bottomY - topY);
                node._isAnchor = false;
                node._chainId = null;
                node._chainProgress = null;
            }
        }
    }

    _updateNodeFromProgress(node, cl, layout, topX, topY, bottomX, bottomY) {
        const t = node._chainProgress;
        const linearX = topX + t * (bottomX - topX);
        const linearY = topY + t * (bottomY - topY);
        const arcX = cl.side * cl.spreadFactor * cl.maxSpread * Math.sin(t * Math.PI);
        node.x = linearX + arcX;
        node.y = linearY;
        node._topX = topX;
        node._topY = topY;
        node._botX = bottomX;
        node._botY = bottomY;
    }

    // ── Phase 3: Relaxation ─────────────────────────────────────

    _p3_startRelaxation() {
        this._relaxIter = 0;
        this._emitTick();
        this._relaxTimer = setInterval(() => {
            if (this.isPaused) return;
            this._relaxStep();
            this._relaxIter++;
            if (this._relaxIter >= this.config.relaxIterations) {
                clearInterval(this._relaxTimer);
                this._relaxTimer = null;
                this._phase = 4;
                this._complete();
            } else {
                this._emitTick();
            }
        }, this.config.relaxDelay);
    }

    _relaxStep() {
        const step = this.config.relaxStepSize * Math.pow(this.config.relaxStepDecay, this._relaxIter);

        for (const link of this.links) {
            if (link._isChainLink) continue;

            const source = link.source;
            const target = link.target;
            if (!source || !target) continue;

            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 1) continue;

            this._relaxNode(source, dx, dy, dist, step);
            this._relaxNode(target, -dx, -dy, dist, step);
        }
    }

    _relaxNode(node, dx, dy, dist, step) {
        const chainId = node._chainId;
        if (chainId === undefined || chainId === null) return;
        if (node._isAnchor) return;

        const cl = this._chainInfoPerChain.get(chainId);
        if (!cl) return;

        const t = node._chainProgress;

        // Chain tangent direction at current progress
        // x(t) = topX + t * (bottomX - topX) + side * spreadFactor * maxSpread * sin(t * PI)
        // dx/dt = (bottomX - topX) + side * spreadFactor * maxSpread * PI * cos(t * PI)
        // y(t) = topY + t * (bottomY - topY)
        // dy/dt = (bottomY - topY)

        const dx_dt = (node._botX - node._topX) + cl.side * cl.spreadFactor * cl.maxSpread * Math.PI * Math.cos(t * Math.PI);
        const dy_dt = (node._botY - node._topY);

        if (Math.abs(dx_dt) < 0.001 && Math.abs(dy_dt) < 0.001) return;

        // Project link force onto chain tangent direction
        // cosθ = (dx,dy)·(dx_dt,dy_dt) / (|link| * |tangent|)
        // We want newT = t + step * cosθ  (gradient descent on link distance)
        const dot = dx * dx_dt + dy * dy_dt;
        const tangentMag = Math.sqrt(dx_dt * dx_dt + dy_dt * dy_dt);
        const cosTheta = dot / (dist * tangentMag);

        const newT = Math.min(1, Math.max(0, t + step * cosTheta));

        node._chainProgress = newT;

        // Recompute position using same formula as _updateNodeFromProgress
        const linearX = node._topX + newT * (node._botX - node._topX);
        const linearY = node._topY + newT * (node._botY - node._topY);
        const arcX = cl.side * cl.spreadFactor * cl.maxSpread * Math.sin(newT * Math.PI);
        node.x = linearX + arcX;
        node.y = linearY;
    }

    // ── Sorting for Phased Reveal ───────────────────────────────

    _sortNodes() {
        this.nodes.sort((a, b) => {
            const aA = this._anchorSet.has(a.id) ? 1 : 0;
            const bA = this._anchorSet.has(b.id) ? 1 : 0;
            return bA - aA;
        });
    }

    _sortLinks() {
        this.links.sort((a, b) => {
            const aC = a._isChainLink ? 1 : 0;
            const bC = b._isChainLink ? 1 : 0;
            return bC - aC;
        });
    }

    _countChainLinks() {
        let count = 0;
        for (const link of this.links) {
            if (link._isChainLink) count++;
        }
        this._chainLinkCount = count;
    }

    // ── Component Centers & Chain Layout Params ─────────────────

    _componentCenters(nodesByComponent) {
        const padding = this.runtime.boundaryPadding;
        const componentIds = this.components.map(c => String(c));
        const compCount = Math.max(1, componentIds.length);
        const columns = Math.max(1, Math.ceil(Math.sqrt(compCount)));
        const rows = Math.max(1, Math.ceil(compCount / columns));

        const usableW = Math.max(1, this.effectiveWidth - padding * 2);
        const usableH = Math.max(1, this.effectiveHeight - padding * 2);

        const cellWidth = usableW / columns;
        const cellHeight = usableH / rows;

        const centers = new Map();

        componentIds.forEach((id, i) => {
            const col = i % columns;
            const row = Math.floor(i / columns);

            centers.set(id, {
                cx: padding + (col + 0.5) * cellWidth,
                cy: padding + (row + 0.5) * cellHeight,
                cellWidth: cellWidth - this.componentPadding * 2,
                cellHeight: cellHeight - this.componentPadding * 2,
            });
        });
        return centers;
    }

    // ── Existing Core Methods ───────────────────────────────────

    buildChainData() {
        this.chainsByComponent.clear();
        for (const chain of this.chains) {
            const compId = String(chain.component);
            if (!this.chainsByComponent.has(compId)) {
                this.chainsByComponent.set(compId, []);
            }
            this.chainsByComponent.get(compId).push(chain);
        }

        this.nodeToChain.clear();
        this.nodeChainOrder.clear();
        for (const chain of this.chains) {
            for (let i = 0; i < chain.nodes.length; i++) {
                const nodeId = chain.nodes[i];
                if (!this.nodeToChain.has(nodeId)) {
                    this.nodeToChain.set(nodeId, chain.id);
                    this.nodeChainOrder.set(nodeId, i);
                }
            }
        }
    }

    indexNodes() {
        this.nodeById.clear();
        for (const node of this.nodes) {
            this.nodeById.set(node.id, node);
        }
    }

    groupNodesByComponent() {
        this.componentNodes.clear();
        for (const node of this.nodes) {
            const componentId = String(node.component);
            if (!this.componentNodes.has(componentId)) {
                this.componentNodes.set(componentId, []);
            }
            this.componentNodes.get(componentId).push(node);
        }
        if (!this.components.length) {
            this.components = Array.from(this.componentNodes.keys());
        }
    }

    _groupNodesByComponentRaw() {
        const map = new Map();
        for (const node of this.nodes) {
            const compId = String(node.component);
            if (!map.has(compId)) map.set(compId, []);
            map.get(compId).push(node);
        }
        return map;
    }

    calculateNodeDegree() {
        this.nodeDegree.clear();
        for (const link of this.links) {
            const sourceId = this.getNodeId(link.source);
            const targetId = this.getNodeId(link.target);
            const currentSource = this.nodeDegree.get(sourceId);
            this.nodeDegree.set(sourceId, currentSource === undefined ? 1 : currentSource + 1);
            const currentTarget = this.nodeDegree.get(targetId);
            this.nodeDegree.set(targetId, currentTarget === undefined ? 1 : currentTarget + 1);
        }
    }

    prepareWorldBounds() {
        const viewportBase = Math.max(this.config.width, this.config.height, this.config.minWorldSize);
        const scaledWorld = Math.ceil(Math.sqrt(Math.max(1, this.nodes.length)) * this.runtime.worldScale);
        const worldSize = Math.min(this.config.maxWorldSize, Math.max(viewportBase, scaledWorld));
        this.effectiveWidth = worldSize;
        this.effectiveHeight = worldSize;
    }

    prepareNodes() {
        for (const node of this.nodes) {
            node.x = 0;
            node.y = 0;
            node.vx = 0;
            node.vy = 0;
            node.fx = null;
            node.fy = null;
            node._charge = 0;
            node._scoreBias = 0;
            node._scoreForce = 0;
            node._isAnchor = false;
            node._chainId = undefined;
            node._chainProgress = null;
            node._topX = null;
            node._topY = null;
            node._botX = null;
            node._botY = null;
        }

        // Compute component grid layouts
        const groups = this._groupNodesByComponentRaw();
        this._componentLayouts = this._componentCenters(groups);

        // Build chain layout parameters per component
        this._chainLayoutByComponent.clear();
        this._chainInfoPerChain.clear();

        for (const [compId, compNodes] of groups) {
            const layout = this._componentLayouts.get(compId);
            if (!layout) continue;

            const compChains = this.chainsByComponent.get(compId) || [];
            const chainsWithNodes = [];

            for (const chain of compChains) {
                const chainNodes = [];
                for (const nid of chain.nodes) {
                    const n = this.nodeById.get(nid);
                    if (n) chainNodes.push(n);
                }
                if (chainNodes.length === 0) continue;

                chainNodes.sort((a, b) => b.score - a.score);
                const topScore = chainNodes[0].score;
                const bottomScore = chainNodes[chainNodes.length - 1].score;

                chainsWithNodes.push({
                    id: chain.id,
                    nodes: chainNodes,
                    topScore: topScore,
                    bottomScore: bottomScore,
                    scoreRange: topScore - bottomScore,
                    length: chainNodes.length,
                });
            }

            // Sort by length ascending, assign spreadFactor by rank
            chainsWithNodes.sort((a, b) => a.length - b.length);
            const numChains = chainsWithNodes.length;

            // Re-sort for diamond: spreadFactor ascending, then alternate left/right
            // (shortest chain = lowest spread = near center, longest = highest spread = outside)
            const maxSpread = Math.min(layout.cellWidth, layout.cellHeight) * this.config.maxSpreadFraction;

            chainsWithNodes.forEach((chain, idx) => {
                const spreadFactor = numChains === 1
                    ? 0
                    : this.config.minSpread + (idx / (numChains - 1)) * (1 - this.config.minSpread);
                chain.spreadFactor = spreadFactor;
                chain.maxSpread = maxSpread;
            });

            // Sort by spread descending for diamond (widest outside)
            chainsWithNodes.sort((a, b) => b.spreadFactor - a.spreadFactor);
            chainsWithNodes.forEach((chain, idx) => {
                chain.side = (idx % 2 === 0) ? -1 : 1;
            });

            this._chainLayoutByComponent.set(compId, chainsWithNodes);

            // Store per-chain info for relaxation
            for (const chain of chainsWithNodes) {
                this._chainInfoPerChain.set(chain.id, {
                    id: chain.id,
                    side: chain.side,
                    spreadFactor: chain.spreadFactor,
                    maxSpread: maxSpread,
                    topScore: chain.topScore,
                    bottomScore: chain.bottomScore,
                    scoreRange: chain.scoreRange,
                    layout: layout,
                    sortedNodeIds: chain.nodes.map(n => n.id),
                });
            }
        }
    }

    prepareLinks() {
        const baseDistance = this.config.linkDistance * this.runtime.linkDistanceScale;
        const scoreMultiplier = this.config.scoreDistanceMultiplier * this.runtime.scoreDistanceMultiplierScale;
        const countMultiplier = this.config.countDistanceMultiplier * this.runtime.countDistanceMultiplierScale;
        const maxLinkDistance = this.runtime.maxLinkDistance;

        // Classify chain links: find any chain where both nodes are consecutive in score order
        this.linkIsChainLink.clear();
        this.linkMatchedChain.clear();
        this._chainNodeIds.clear();
        for (let i = 0; i < this.links.length; i++) {
            const link = this.links[i];
            const sourceId = this.getNodeId(link.source);
            const targetId = this.getNodeId(link.target);
            let isChain = false;
            let matchChainId = undefined;
            for (const [chainId, chainInfo] of this._chainInfoPerChain) {
                const ids = chainInfo.sortedNodeIds;
                if (!ids) continue;
                const si = ids.indexOf(sourceId);
                if (si < 0) continue;
                const ti = ids.indexOf(targetId);
                if (ti < 0) continue;
                if (Math.abs(si - ti) === 1) {
                    isChain = true;
                    matchChainId = chainId;
                    break;
                }
            }
            this.linkIsChainLink.set(i, isChain);
            this.linkMatchedChain.set(i, matchChainId);
        }

        // Build chain node ID sets for highlighting
        for (const [chainId, chainInfo] of this._chainInfoPerChain) {
            if (!this._chainNodeIds.has(chainId)) {
                this._chainNodeIds.set(chainId, new Set());
            }
            const nset = this._chainNodeIds.get(chainId);
            for (const nid of chainInfo.sortedNodeIds) {
                nset.add(nid);
            }
        }

        for (let i = 0; i < this.links.length; i++) {
            const link = this.links[i];
            const source = this.nodeById.get(this.getNodeId(link.source));
            const target = this.nodeById.get(this.getNodeId(link.target));
            if (!source || !target) continue;

            const sourceDegree = this.nodeDegree.get(source.id);
            const targetDegree = this.nodeDegree.get(target.id);
            const scoreGap = Math.abs(
                this.assertFinite(source.score, "source.score") - this.assertFinite(target.score, "target.score"),
            );
            const countWeight = Math.log1p(Math.max(
                this.assertFinite(source.comparison_count, "source.comparison_count"),
                this.assertFinite(target.comparison_count, "target.comparison_count"),
            ));

            const countContribution = baseDistance * countWeight * countMultiplier;
            const targetDist = Math.min(
                maxLinkDistance,
                baseDistance * (1 + scoreGap * scoreMultiplier) + countContribution,
            );
            link.distance = targetDist;
            link._startDistance = targetDist;
            link._targetDistance = targetDist;
            link._baseStrength = Math.min(
                1,
                this.config.linkStrength
                * this.runtime.linkStrengthScale
                * (1 / Math.sqrt(Math.max(1, Math.min(sourceDegree, targetDegree)))),
            );
            link._opacity = this.runtime.usePerLinkOpacity
                ? Math.max(this.config.linkOpacityMin, 1 - (link.distance / (maxLinkDistance * this.config.maxDistanceMultiplier)))
                : this.config.linkOpacityMax;
            link._isChainLink = this.linkIsChainLink.get(i) || false;
            link._matchedChainId = this.linkMatchedChain.get(i);
        }

        if (this.runtime.simulationLinkStride === 1) {
            this.simulationLinks = this.links;
            return;
        }

        this.simulationLinks = this.links.filter((_, index) =>
            index % this.runtime.simulationLinkStride === 0,
        );
    }

    setupSimulation() {
        // No-op — layout is phase-driven, not force-driven
    }

    stop() {
        if (this._phaseTimer) { clearTimeout(this._phaseTimer); this._phaseTimer = null; }
        if (this._relaxTimer) { clearInterval(this._relaxTimer); this._relaxTimer = null; }
        this._phase = 4;
        this.isPaused = true;
    }

    pause() {
        this.isPaused = true;
    }

    play() {
        this.isPaused = false;
    }

    dragLinkedNodesWhilePaused(subject, dx, dy) {
        if (!this.isPaused) return;
        subject.x += this.assertFinite(dx, "dx");
        subject.y += this.assertFinite(dy, "dy");
    }

    // ── Utility ─────────────────────────────────────────────────

    getScoreBias(score) {
        const centeredScore = (this.assertFinite(score, "score") - 0.5) * 2;
        const magnitude = Math.abs(centeredScore);
        const deadZone = this.config.neutralDeadZone;
        if (magnitude <= deadZone) return 0;
        const normalized = (magnitude - deadZone) / (1 - deadZone);
        return Math.sign(centeredScore) * Math.pow(normalized, this.config.scoreExponent);
    }

    assertFinite(value, name) {
        const n = Number(value);
        if (!Number.isFinite(n)) {
            showError("ChainSim: " + name + " is not a finite number (got " + JSON.stringify(value) + ")");
            throw new Error("ChainSim: " + name + " not finite");
        }
        return n;
    }

    getNodeId(nodeOrId) {
        return typeof nodeOrId === "object" ? nodeOrId.id : nodeOrId;
    }

    getSimNode(id) {
        return this.nodeById.get(id);
    }

    clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    resolveRuntimeProfile() {
        const nodeCount = this.nodes.length;
        const linkCount = this.links.length;
        const t = PERFORMANCE;
        const workUnits = nodeCount + (linkCount * t.graph.linkWorkWeight);
        const mediumGraph = workUnits >= t.graph.mediumGraphMinWorkUnits;
        const largeGraph = workUnits >= t.graph.largeGraphMinWorkUnits;
        const sizeKey = largeGraph ? "large" : (mediumGraph ? "medium" : "small");
        const PROFILE_SRC = { link: RENDER.link, render: PERFORMANCE.render };
        const getProfileValue = (category, key) => {
            const src = PROFILE_SRC[category];
            const multKey = sizeKey + key.charAt(0).toUpperCase() + key.slice(1) + "Multiplier";
            const mult = src[multKey];
            if (mult !== undefined) {
                return src[key] * mult;
            }
            const directKey = key + sizeKey.charAt(0).toUpperCase() + sizeKey.slice(1);
            const direct = src[directKey];
            return direct !== undefined ? direct : src[key];
        };
        const s = SCALE;
        CONTROLS.enablePointerHits = nodeCount < t.node.pointerHitsMaxNodes;

        return {
            canvasMode: true,
            enableCollision: true,
            enableMarkers: false,
            enableLabels: nodeCount < t.node.labelsMaxNodes,
            enableDrag: true,
            enablePausedGroupDrag: true,
            enableTooltips: nodeCount < t.node.tooltipsMaxNodes,
            drawLinks: linkCount < t.link.drawLinksMaxLinks,
            progressiveReveal: mediumGraph || nodeCount > t.batch.progressiveRevealMinNodes,
            usePerLinkOpacity: linkCount <= t.link.perLinkOpacityMaxLinks,
            linkVisibilityZoomThreshold: getProfileValue("link", "linkVisibilityZoomThreshold"),
            renderIntervalMs: getProfileValue("render", "renderIntervalMs"),
            chargeDistanceMax: this.config.chargeDistanceMax * s.node.chargeDistanceMultiplier,
            gravityStrength: this.config.gravityStrength * s.node.gravityMultiplier,
            velocityDecay: this.config.velocityDecay * s.world.velocityDecayMultiplier,
            alphaDecay: this.config.alphaDecay * s.world.alphaDecayMultiplier,
            linkDistanceScale: s.link.distanceMultiplier,
            linkStrengthScale: s.link.strengthMultiplier,
            scoreDistanceMultiplierScale: s.link.scoreDistMultiplier,
            countDistanceMultiplierScale: s.link.countDistMultiplier,
            maxLinkDistance: this.config.maxLinkDistance * s.link.maxDistanceMultiplier,
            worldScale: this.config.worldScale * s.world.worldScaleMultiplier,
            boundaryPadding: this.config.boundaryPadding * s.world.boundaryPaddingMultiplier,
            simulationLinkStride: largeGraph || mediumGraph
                ? Math.max(1, Math.ceil(linkCount / (PERFORMANCE.link.linkStrideDivisor * (largeGraph ? PERFORMANCE.link.linkStrideLargeMultiplier : 1))))
                : 1,
            constraintPhases: largeGraph || mediumGraph
                ? Math.max(1, Math.ceil(linkCount / (PERFORMANCE.constraint.constraintPhasesDivisor * (largeGraph ? PERFORMANCE.constraint.constraintPhasesLargeMultiplier : 1))))
                : 1,
            constraintPull: this.config.linkConstraintPull * s.link.constraintPullMultiplier,
            startPaused: workUnits >= t.graph.startPausedWorkUnits,
            statusLabel: largeGraph ? "Chain / Reduced Physics" : (mediumGraph ? "Chain / Scaled" : "Chain"),
        };
    }
}
