/**
 * Buoyancy Simulation with Link Constraints
 *
 * Higher scores float, lower scores sink. Score 0.5 stays neutral.
 * Links maintain a target distance based on score difference.
 * Nodes never overlap. Components repel. Bouncy borders.
 */

class ChainSimulation {
    constructor(nodes, links, components, options = {}) {
        this.nodes = nodes;
        this.links = links;
        this.components = Array.isArray(components) ? Array.from(components) : [];
        this._tickCount = 0;
        this._simStart = performance.now();

        this.onTick = options.onTick;
        this.onEnd = options.onEnd;
        this.isPaused = false;

        this.nodeById = new Map();
        this._activeNodeCount = this.nodes.length;
        this._activeNodes = [...this.nodes];
        this._activeLinks = [...this.links];
        this.simulationLinks = this.links;

        this.effectiveWidth = 2000;
        this.effectiveHeight = 2000;

        // Physics parameters
        this._verticalScale = 500;
        this._buoyancyStrength = 0.08;
        this._velocityDecay = 0.88;
        this._collisionStrength = 0.8;
        this._compRepelStrength = 0.4;
        this._compRepelRange = 400;
        this._bouncyBounds = true;
        this._restitution = 0.5;
        this._boundaryPadding = 60;

        // Link constraint parameters
        this._baseLinkLength = 40;
        this._scoreDistanceScale = 300;
        this._linkStrength = 0.4;

        // Alpha cooling — faster decay for large datasets
        const nodeCount = this.nodes.length;
        this._alphaDecay = nodeCount > 10000 ? 0.025 : nodeCount > 5000 ? 0.015 : nodeCount > 1000 ? 0.008 : 0.003;
        this._alpha = 1;
        this._alphaMin = 0.002;

        // Time-based force ramping (sine curve: 0 → peak → 0)
        this._rampTotal = options.rampTotal || 0;
        this._maxAlpha = options.maxAlpha || this._alpha;

        // When true, buoyancy uses per-component centers instead of absolute map center
        this._noCenterAttract = options.noCenterAttract !== false;

        // Grid & spatial hash
        this._maxNodeRadius = 8;
        this._cellSize = 30;
        this._gap = 5;

        this._running = false;
        this._timer = null;
        this._phase = 2;

        // Component data
        this._compNodesMap = new Map();

        this.runtime = {
            canvasMode: true,
            enableCollision: true,
            enableMarkers: false,
            enableLabels: true,
            enableDrag: true,
            enablePausedGroupDrag: true,
            enableTooltips: true,
            drawLinks: true,
            progressiveReveal: false,
            usePerLinkOpacity: false,
            linkVisibilityZoomThreshold: 0.01,
            startPaused: false,
            statusLabel: "Buoyancy",
        };
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
        this._computeMaxRadius();
        this._groupByComponent();
        this._sizedWorld();
        this._prepareLinks();
        this._initPositions();
    }

    _computeMaxRadius() {
        this._maxNodeRadius = 8;
        for (const n of this.nodes) {
            if (n._radius && n._radius > this._maxNodeRadius) {
                this._maxNodeRadius = n._radius;
            }
        }
        this._gap = 5;
        this._cellSize = this._maxNodeRadius * 2 + this._gap;
    }

    indexNodes() {
        this.nodeById.clear();
        for (const n of this.nodes) {
            this.nodeById.set(n.id, n);
        }
    }

    _groupByComponent() {
        this._compNodesMap.clear();
        if (!this.components.length) {
            const seen = new Set();
            for (const n of this.nodes) {
                const cid = String(n.component);
                seen.add(cid);
            }
            this.components = Array.from(seen);
        }
        for (const cid of this.components) {
            this._compNodesMap.set(String(cid), []);
        }
        for (const n of this.nodes) {
            const cid = String(n.component);
            if (!this._compNodesMap.has(cid)) {
                this._compNodesMap.set(cid, []);
            }
            this._compNodesMap.get(cid).push(n);
        }
    }

    _prepareLinks() {
        for (const link of this.links) {
            const sId = this.getNodeId(link.source);
            const tId = this.getNodeId(link.target);
            const source = this.nodeById.get(sId);
            const target = this.nodeById.get(tId);
            if (!source || !target) continue;

            const scoreDiff = Math.abs(source.score - target.score);
            link._targetDistance = this._baseLinkLength + scoreDiff * this._scoreDistanceScale;
        }
    }

    _sizedWorld() {
        const cellSize = this._cellSize;
        const compIds = Array.from(this._compNodesMap.keys());
        const compCols = Math.max(1, Math.ceil(Math.sqrt(compIds.length)));
        const compRows = compIds.length > 0 ? Math.ceil(compIds.length / compCols) : 1;

        let maxCompWidth = 0;
        for (const [, compNodes] of this._compNodesMap) {
            const cols = Math.max(1, Math.ceil(Math.sqrt(compNodes.length)));
            const w = cols * cellSize;
            if (w > maxCompWidth) maxCompWidth = w;
        }

        const gridWidth = compCols * (maxCompWidth + cellSize) + this._boundaryPadding * 2;
        const scoreRange = this._verticalScale * 2;
        const rowGap = this._verticalScale * 0.2;
        const gridHeight = scoreRange + Math.max(0, compRows - 1) * rowGap + this._boundaryPadding * 4;

        const size = Math.min(1200000, Math.max(2000, gridWidth, gridHeight));
        this.effectiveWidth = size;
        this.effectiveHeight = size;
    }

    _initPositions() {
        const cellSize = this._cellSize;
        const cy = this.effectiveHeight / 2;
        const compIds = Array.from(this._compNodesMap.keys());
        if (compIds.length === 0) return;

        const compCols = Math.max(1, Math.ceil(Math.sqrt(compIds.length)));

        for (let ci = 0; ci < compIds.length; ci++) {
            const compId = compIds[ci];
            const compNodes = this._compNodesMap.get(compId);
            if (!compNodes || !compNodes.length) continue;

            const col = ci % compCols;
            const row = Math.floor(ci / compCols);

            compNodes.sort((a, b) => b.score - a.score);
            const count = compNodes.length;
            const nodeCols = Math.max(1, Math.ceil(Math.sqrt(count)));
            const totalW = nodeCols * cellSize;

            const compX = this._boundaryPadding + col * (totalW + cellSize);

            for (let i = 0; i < count; i++) {
                const n = compNodes[i];
                const nc = i % nodeCols;
                const nr = Math.floor(i / nodeCols);
                n.x = compX + nc * cellSize + cellSize / 2;
                n.y = cy + (0.5 - n.score) * this._verticalScale + row * this._verticalScale * 0.2 + nr;
                n.vx = 0;
                n.vy = 0;
            }
        }

        const halfW = this.effectiveWidth / 2;
        const halfH = this.effectiveHeight / 2;
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const n of this.nodes) {
            if (n.x < minX) minX = n.x;
            if (n.x > maxX) maxX = n.x;
            if (n.y < minY) minY = n.y;
            if (n.y > maxY) maxY = n.y;
        }
        const cx = (minX + maxX) / 2;
        const cy2 = (minY + maxY) / 2;
        const dx = halfW - cx;
        const dy = halfH - cy2;
        if (Math.abs(dx) > 1 || Math.abs(dy) > 1) {
            for (const n of this.nodes) {
                n.x += dx;
                n.y += dy;
            }
        }
    }

    // ── Simulation Loop ─────────────────────────────────────────

    start() {
        this._phase = 1;
        this._alpha = this._rampTotal > 0 ? 0 : 1;
        this._simStart = performance.now();
        this._running = true;
        this._emitTick();
        this._tick();
    }

    _tick() {
        if (!this._running || this.isPaused) return;

        try {
            if (this._rampTotal > 0) {
                const elapsed = (performance.now() - this._simStart) / 1000;
                if (elapsed >= this._rampTotal) {
                    this._alpha = 0;
                } else {
                    const raw = Math.sin((elapsed / this._rampTotal) * Math.PI);
                    this._alpha = Math.max(raw, 0.03) * this._maxAlpha;
                }
            } else {
                this._alpha *= (1 - this._alphaDecay);
            }

            this._applyBuoyancy();
            this._applyLinkConstraints();
            this._resolveCollisions();
            this._applyComponentRepulsion();
            this._applyMotion();
            this._applyBoundaryBounce();

            this._tickCount++;

            this._emitTick();
        } catch (e) {
            console.error("Simulation tick error:", e);
            this._alpha = 0;
        }

        if (this._alpha < this._alphaMin) {
            this._complete();
            return;
        }

        this._timer = requestAnimationFrame(() => this._tick());
    }

    _emitTick() {
        if (this.onTick) {
            this.onTick({
                phase: this._phase,
                visibleNodeCount: this.nodes.length,
                visibleLinkCount: this.links.length,
                done: this._phase === 2,
            });
        }
    }

    _complete() {
        this._phase = 2;
        this._running = false;
        if (this._timer) {
            cancelAnimationFrame(this._timer);
            this._timer = null;
        }
        this._emitTick();
        if (this.onEnd) this.onEnd(this.runtime);
    }

    // ── Physics Steps ───────────────────────────────────────────

    _applyBuoyancy() {
        if (this._buoyancyStrength <= 0) return;
        const strength = this._buoyancyStrength * this._alpha;

        if (this._noCenterAttract) {
            // Score-based vertical drift without any center anchor.
            // Higher scores drift up, lower scores drift down. No absolute Y target.
            const drift = strength * 0.3;
            for (const n of this.nodes) {
                n.vy += (0.5 - n.score) * drift;
            }
        } else {
            const cy = this.effectiveHeight / 2;
            for (const n of this.nodes) {
                n.vy += (cy + (0.5 - n.score) * this._verticalScale - n.y) * strength;
            }
        }
    }

    _applyLinkConstraints() {
        const strength = this._linkStrength * this._alpha;

        for (const link of this.links) {
            const source = link.source;
            const target = link.target;
            if (!source || !target) continue;

            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 0.001) continue;

            const diff = dist - link._targetDistance;
            const force = diff * strength;
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;

            source.vx += fx;
            source.vy += fy;
            target.vx -= fx;
            target.vy -= fy;
        }
    }

    _resolveCollisions() {
        const strength = this._collisionStrength * this._alpha;
        const arr = this.nodes;
        if (arr.length < 2) return;

        const cellSize = this._cellSize;
        if (cellSize <= 0) return;

        const hash = new Map();
        for (const n of arr) {
            const cx = Math.floor(n.x / cellSize);
            const cy = Math.floor(n.y / cellSize);
            const key = `${cx},${cy}`;
            if (!hash.has(key)) hash.set(key, []);
            hash.get(key).push(n);
        }

        for (const [key, cell] of hash) {
            const comma = key.indexOf(',');
            const cx = parseInt(key.substring(0, comma));
            const cy = parseInt(key.substring(comma + 1));

            for (let i = 0; i < cell.length; i++) {
                for (let j = i + 1; j < cell.length; j++) {
                    this._resolvePair(cell[i], cell[j], strength);
                }
            }

            for (let dcx = -1; dcx <= 1; dcx++) {
                for (let dcy = -1; dcy <= 1; dcy++) {
                    if (dcx === 0 && dcy === 0) continue;
                    const nkey = `${cx + dcx},${cy + dcy}`;
                    const ncell = hash.get(nkey);
                    if (!ncell) continue;
                    for (const a of cell) {
                        for (const b of ncell) {
                            this._resolvePair(a, b, strength);
                        }
                    }
                }
            }
        }
    }

    _resolvePair(a, b, strength) {
        const rA = a._radius || 8;
        const rB = b._radius || 8;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const minDist = rA + rB;

        if (dist < minDist && dist > 0.001) {
            const overlap = minDist - dist;
            const nx = dx / dist;
            const ny = dy / dist;
            const push = overlap * strength * 0.5;
            a.x -= nx * push;
            a.y -= ny * push;
            b.x += nx * push;
            b.y += ny * push;
            a.vx -= nx * push * 0.2;
            a.vy -= ny * push * 0.2;
            b.vx += nx * push * 0.2;
            b.vy += ny * push * 0.2;
        }
    }

    _applyComponentRepulsion() {
        if (this._compNodesMap.size < 2) return;
        if (this._compNodesMap.size > 200 || this.nodes.length > 5000) return;

        const centers = new Map();
        for (const [compId, compNodes] of this._compNodesMap) {
            if (!compNodes.length) continue;
            let cx = 0, cy = 0;
            for (const n of compNodes) {
                cx += n.x;
                cy += n.y;
            }
            cx /= compNodes.length;
            cy /= compNodes.length;
            centers.set(compId, { cx, cy });
        }

        const strength = this._compRepelStrength * this._alpha;
        const ids = Array.from(centers.keys());

        for (let i = 0; i < ids.length; i++) {
            for (let j = i + 1; j < ids.length; j++) {
                const a = centers.get(ids[i]);
                const b = centers.get(ids[j]);
                const dx = b.cx - a.cx;
                const dy = b.cy - a.cy;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < this._compRepelRange && dist > 1) {
                    const force = ((this._compRepelRange - dist) / this._compRepelRange) * strength;
                    const nx = dx / dist;
                    const ny = dy / dist;

                    const ca = this._compNodesMap.get(ids[i]);
                    const cb = this._compNodesMap.get(ids[j]);
                    for (const n of ca) {
                        n.vx -= nx * force;
                        n.vy -= ny * force;
                    }
                    for (const n of cb) {
                        n.vx += nx * force;
                        n.vy += ny * force;
                    }
                }
            }
        }
    }

    _applyBoundaryBounce() {
        if (!this._bouncyBounds) return;

        const pad = this._boundaryPadding;
        const minX = pad;
        const maxX = this.effectiveWidth - pad;
        const minY = pad;
        const maxY = this.effectiveHeight - pad;
        const rest = this._restitution;

        for (const n of this.nodes) {
            const r = n._radius || 8;

            if (n.x - r < minX) {
                n.x = minX + r;
                n.vx = Math.abs(n.vx) * rest;
            } else if (n.x + r > maxX) {
                n.x = maxX - r;
                n.vx = -Math.abs(n.vx) * rest;
            }

            if (n.y - r < minY) {
                n.y = minY + r;
                n.vy = Math.abs(n.vy) * rest;
            } else if (n.y + r > maxY) {
                n.y = maxY - r;
                n.vy = -Math.abs(n.vy) * rest;
            }
        }
    }

    _applyMotion() {
        for (const n of this.nodes) {
            n.vx *= this._velocityDecay;
            n.vy *= this._velocityDecay;



            n.x += n.vx;
            n.y += n.vy;
        }
    }

    // ── Control ─────────────────────────────────────────────────

    stop() {
        this._running = false;
        if (this._timer) {
            cancelAnimationFrame(this._timer);
            this._timer = null;
        }
        this._phase = 2;
        this.isPaused = true;
    }

    pause() {
        this.isPaused = true;
    }

    play() {
        if (!this.isPaused) return;
        this.isPaused = false;
        if (!this._running || this._alpha < this._alphaMin) {
            this._running = true;
            this._alpha = this._rampTotal > 0 ? 0 : 0.5;
            if (this._rampTotal > 0) this._simStart = performance.now();
            this._phase = 1;
        }
        this._tick();
    }

    dragLinkedNodesWhilePaused(subject, dx, dy) {
        if (!this.isPaused) return;
        subject.x += dx;
        subject.y += dy;
    }

    // ── Utility ─────────────────────────────────────────────────

    getNodeId(nodeOrId) {
        return typeof nodeOrId === "object" ? nodeOrId.id : nodeOrId;
    }

    getSimNode(id) {
        return this.nodeById.get(id);
    }

    clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }
}
