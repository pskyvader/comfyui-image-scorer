/**
 * Chain Simulation Logic (Physics Engine)
 */

class ChainSimulation {
    constructor(nodes, links, components, options = {}) {
        this.nodes = nodes;
        this.links = links;
        this.components = Array.from(components);
        this._tickCount = 0;
        this._simStart = performance.now();

        this.config = { ...PHYSICS.link, ...PHYSICS.node, ...PHYSICS.world, ...RENDER.link, ...CONTROLS, ...options };

        this.simulation = null;
        this.linkForce = null;
        this.onTick = options.onTick;
        this.onEnd = options.onEnd;
        this.isPaused = false;

        this.nodeById = new Map();
        this.nodeDegree = new Map();
        this.componentNodes = new Map();
        this.runtime = this.resolveRuntimeProfile();

        this._activeNodeCount = 0;
        this._activeNodes = [];
        this._activeLinks = [];
        this._pendingNodeQueue = [...this.nodes];

        this.batchSize = Math.max(PERFORMANCE.batch.nodeBatchMin, Math.ceil(this.nodes.length / PERFORMANCE.batch.nodeBatchDivisor));

        this.simulationLinks = this.links;
        this.effectiveWidth = this.config.minWorldSize;
        this.effectiveHeight = this.config.minWorldSize;
        this.constraintPhase = 0;
        this.lastRenderAt = 0;
        this.currentLinkStrengthFactor = this.config.linkStrengthFinalFactor;
    }

    setVisibleCount(count) {
        const current = this._activeNodeCount;
        if (count <= current) return;
        const added = this._addNodes(count - current);
        this._syncSimulation();
        return added;
    }

    _addNodes(count) {
        if (this._pendingNodeQueue.length === 0 || count <= 0) return 0;
        const batch = this._pendingNodeQueue.splice(0, count);
        this._positionNodes(batch);
        this._activeNodes.push(...batch);
        this._activeNodeCount = this._activeNodes.length;
        return batch.length;
    }

    _syncSimulation(nudge) {
        this._rebuildActiveLinks();
        this.simulation.nodes(this._activeNodes);
        this.linkForce.links(this._activeLinks);
        if (nudge) {
            for (let i = 0; i < this._activeNodes.length; i++) {
                this._activeNodes[i].vx += (Math.random() - 0.5) * nudge;
                this._activeNodes[i].vy += (Math.random() - 0.5) * nudge;
            }
        }
        this.simulation.alpha(this.config.batchAlpha).restart();
    }

    addNodeBatch(count) {
        if (this._pendingNodeQueue.length === 0 || count <= 0) return 0;
        const added = this._addNodes(count);
        this._syncSimulation(this.config.centerSpread * 4);
        return added;
    }

    _rebuildActiveLinks() {
        const activeIds = new Set();
        for (let i = 0; i < this._activeNodes.length; i++) {
            activeIds.add(this._activeNodes[i].id);
        }
        this._activeLinks = this.links.filter(function (l) {
            const sId = typeof l.source === "object" ? l.source.id : l.source;
            const tId = typeof l.target === "object" ? l.target.id : l.target;
            return activeIds.has(sId) && activeIds.has(tId);
        });
    }

    _componentCenters() {
        const padding = this.runtime.boundaryPadding;
        const componentIds = this.components.map(c => String(c));
        const compCount = Math.max(1, componentIds.length);
        const columns = Math.max(1, Math.ceil(Math.sqrt(compCount)));
        const rows = Math.max(1, Math.ceil(compCount / columns));
        const usableW = Math.max(1, this.effectiveWidth - padding * 2);
        const usableH = Math.max(1, this.effectiveHeight - padding * 2);
        const centers = new Map();

        componentIds.forEach((id, i) => {
            const col = i % columns;
            const row = Math.floor(i / columns);
            centers.set(id, {
                x: padding + ((col + 0.5) / columns) * usableW,
                y: padding + ((row + 0.5) / rows) * usableH,
            });
        });
        return centers;
    }

    _positionNodes(nodes) {
        const centers = this._componentCenters();
        for (const node of nodes) {
            const center = centers.get(String(node.component));
            if (!center) {
                showError("ChainSim: node \"" + node.id + "\" has unknown component \"" + node.component + "\"");
                throw new Error("ChainSim: unknown component");
            }
            const scoreBias = this.getScoreBias(this.assertFinite(node.score, "node.score"));

            node.x = center.x + (Math.random() - 0.5) * this.config.centerSpread;
            node.y = center.y + (Math.random() - 0.5) * this.config.centerSpread;
            node.vx = 0;
            node.vy = 0;
            const degree = this.nodeDegree.get(node.id);
            node._charge = this.config.chargeStrength * (degree === 0 ? this.config.isolatedMultiplier : 1);
            node._scoreBias = scoreBias;
            node._scoreForce = scoreBias * this.runtime.gravityStrength;
        }
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
            if (mult !== undefined) return src[key] * mult;
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
            statusLabel: largeGraph ? "Canvas / Reduced Physics" : (mediumGraph ? "Canvas / Scaled" : "Canvas"),
        };
    }

    initialize() {
        this.indexNodes();
        this.groupNodesByComponent();
        this.calculateNodeDegree();
        this.prepareWorldBounds();
        this.prepareNodes();
        this.prepareLinks();
        this._activeNodes = [];
        this._activeLinks = [];
        this.setupSimulation();
        this.addNodeBatch(this.batchSize);
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
            this.componentNodes.get(componentId)
                .push(node);
        }
        if (!this.components.length) {
            this.components = Array.from(this.componentNodes.keys());
        }
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
        const centers = this._componentCenters();

        for (const node of this.nodes) {
            const center = centers.get(String(node.component));
            if (!center) {
                showError("ChainSim: node \"" + node.id + "\" has unknown component \"" + node.component + "\"");
                throw new Error("ChainSim: unknown component");
            }
            const scoreBias = this.getScoreBias(this.assertFinite(node.score, "node.score"));

            node.x = center.x + (Math.random() - 0.5) * this.config.centerSpread;
            node.y = center.y + (Math.random() - 0.5) * this.config.centerSpread;
            node.vx = 0;
            node.vy = 0;
            const degree = this.nodeDegree.get(node.id);
            node._charge = this.config.chargeStrength * (degree === 0 ? this.config.isolatedMultiplier : 1);
            node._scoreBias = scoreBias;
            node._scoreForce = scoreBias * this.runtime.gravityStrength;
        }
    }

    prepareLinks() {
        const baseDistance = this.config.linkDistance * this.runtime.linkDistanceScale;
        const scoreMultiplier = this.config.scoreDistanceMultiplier * this.runtime.scoreDistanceMultiplierScale;
        const countMultiplier = this.config.countDistanceMultiplier * this.runtime.countDistanceMultiplierScale;
        const maxLinkDistance = this.runtime.maxLinkDistance;

        for (const link of this.links) {
            const source = this.nodeById.get(this.getNodeId(link.source));
            const target = this.nodeById.get(this.getNodeId(link.target));
            if (!source || !target) {
                continue;
            }

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
            // link.distance = this.config.linkDistance;
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
        this.currentLinkStrengthFactor = this.config.linkStrengthFinalFactor;

        this.linkForce = d3.forceLink(this._activeLinks)
            .id(node => node.id)
            .distance(link => link.distance)
            .strength(link => link._baseStrength * this.currentLinkStrengthFactor);

        this.simulation = d3.forceSimulation(this._activeNodes)
            .force("link", this.linkForce)
            .force("charge", d3.forceManyBody()
                .strength(node => node._charge)
                .distanceMax(this.runtime.chargeDistanceMax))
            .force("scoreY", this.createScoreForce())
            .velocityDecay(this.runtime.velocityDecay)
            .alphaDecay(this.runtime.alphaDecay)
            .alphaTarget(0.1);

        if (this.runtime.enableCollision) {
            this.simulation.force("collide", d3.forceCollide((node) => {
                const count = this.assertFinite(node.comparison_count, "node.comparison_count");
                return this.config.collisionRadius + Math.log1p(count);
            }));
        }

        this.simulation.on("tick", () => this.handleTick());
        this.simulation.on("end", () => {
            console.log("SIM END | tick", this._tickCount, "| alpha", this.simulation.alpha());
            this.onTick(this.nodes, this.links, this.runtime);
            this.onEnd(this.runtime);
        });
    }

    handleTick() {
        this._tickCount++;
        const t = performance.now();
        if (this._tickCount % 40 === 0) {
            const a = this.simulation.alpha();
            const n0 = this._activeNodes[0];
            console.log("TICK", this._tickCount, "| alpha", a !== undefined ? a.toFixed(6) : "?", "| active", this._activeNodeCount, "| n0 x,y,vx,vy:", n0 ? n0.x.toFixed(1) : "N/A", n0 ? n0.y.toFixed(1) : "N/A", n0 ? n0.vx.toFixed(3) : "N/A", n0 ? n0.vy.toFixed(3) : "N/A");
        }

        if (this._pendingNodeQueue.length > 0 && this._tickCount % PERFORMANCE.batch.batchTickDivider === 0) {
            this.addNodeBatch(this.batchSize);
        }

        const now = performance.now();
        this.applyLinkConstraints();

        for (let i = 0; i < this._activeNodes.length; i++) {
            this.applyBoundaryBehavior(this._activeNodes[i]);
        }

        if (!this.onTick) {
            return;
        }

        if (now - this.lastRenderAt >= this.runtime.renderIntervalMs) {
            this.lastRenderAt = now;
            this.onTick(this.nodes, this.links, this.runtime);
        }
    }

    createScoreForce() {
        let nodes = [];
        const force = (alpha) => {
            for (let i = 0; i < nodes.length; i++) {
                const node = nodes[i];
                if (!node._scoreForce) {
                    continue;
                }
                node.vy -= node._scoreForce * alpha;
            }
        };
        force.initialize = (allNodes) => {
            nodes = allNodes;
        };
        return force;
    }

    applyLinkConstraints() {
        if (!this._activeLinks.length) {
            return;
        }

        const phases = this.runtime.constraintPhases;
        const phaseOffset = this.constraintPhase;
        const pull = this.runtime.constraintPull;

        if (pull <= 0 || this.config.linkConstraintPull <= 0) return;

        for (let index = phaseOffset; index < this._activeLinks.length; index += phases) {
            const link = this._activeLinks[index];
            const source = link.source;
            const target = link.target;
            if (!source || !target || typeof source !== "object" || typeof target !== "object") {
                continue;
            }
            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const distanceSq = (dx * dx) + (dy * dy);
            if (!distanceSq) {
                continue;
            }

            const distance = Math.sqrt(distanceSq);
            const stretchTolerance = PHYSICS.link.stretchTolerance;
            const limit = link.distance * stretchTolerance;
            if (distance <= limit) {
                continue;
            }

            const correction = ((distance - limit) / distance) * pull;
            const offsetX = dx * correction;
            const offsetY = dy * correction;
            const sourceFree = source.fx == null;
            const targetFree = target.fx == null;

            if (sourceFree && targetFree) {
                source.x += offsetX * 0.5;
                source.y += offsetY * 0.5;
                target.x -= offsetX * 0.5;
                target.y -= offsetY * 0.5;
            } else if (sourceFree) {
                source.x += offsetX;
                source.y += offsetY;
            } else if (targetFree) {
                target.x -= offsetX;
                target.y -= offsetY;
            }
        }
        this.constraintPhase = (this.constraintPhase + 1) % phases;
    }

    applyBoundaryBehavior(node) {
        const minX = this.runtime.boundaryPadding;
        const maxX = this.effectiveWidth - this.runtime.boundaryPadding;
        const minY = this.runtime.boundaryPadding;
        const maxY = this.effectiveHeight - this.runtime.boundaryPadding;

        if (node.fx != null) {
            node.x = this.clamp(node.fx, minX, maxX);
            node.vx = 0;
        }
        if (node.fy != null) {
            node.y = this.clamp(node.fy, minY, maxY);
            node.vy = 0;
        }

        if (!this.config.useBouncyBounds) {
            node.x = this.clamp(node.x, minX, maxX);
            node.y = this.clamp(node.y, minY, maxY);
            return;
        }

        const bounce = this.clamp(this.config.boundaryBounce, 0, 1);
        if (node.x < minX) {
            node.x = minX + ((minX - node.x) * bounce);
            if (node.vx < 0) node.vx = -node.vx * bounce;
        }
        if (node.x > maxX) {
            node.x = maxX - ((node.x - maxX) * bounce);
            if (node.vx > 0) node.vx = -node.vx * bounce;
        }
        if (node.y < minY) {
            node.y = minY + ((minY - node.y) * bounce);
            if (node.vy < 0) node.vy = -node.vy * bounce;
        }
        if (node.y > maxY) {
            node.y = maxY - ((node.y - maxY) * bounce);
            if (node.vy > 0) node.vy = -node.vy * bounce;
        }

        node.x = this.clamp(node.x, minX, maxX);
        node.y = this.clamp(node.y, minY, maxY);
    }

    dragLinkedNodesWhilePaused(subject, dx, dy) {
        if (!this.isPaused) {
            return;
        }
        const offsetX = this.assertFinite(dx, "dx") * this.config.pausedLinkedDragInfluence;
        const offsetY = this.assertFinite(dy, "dy") * this.config.pausedLinkedDragInfluence;
        if (!offsetX && !offsetY) {
            return;
        }

        const componentId = String(subject.component);
        const members = this.componentNodes.get(componentId);
        if (!members || !members.length) {
            return;
        }

        const minX = this.runtime.boundaryPadding;
        const maxX = this.effectiveWidth - this.runtime.boundaryPadding;
        const minY = this.runtime.boundaryPadding;
        const maxY = this.effectiveHeight - this.runtime.boundaryPadding;

        for (const node of members) {
            if (node === subject) {
                continue;
            }
            node.x = this.clamp(node.x + offsetX, minX, maxX);
            node.y = this.clamp(node.y + offsetY, minY, maxY);
            node.vx = 0;
            node.vy = 0;
        }
        this.onTick(this.nodes, this.links, this.runtime);
    }

    getScoreBias(score) {
        const centeredScore = (this.assertFinite(score, "score") - 0.5) * 2;
        const magnitude = Math.abs(centeredScore);
        const deadZone = this.config.neutralDeadZone;
        if (magnitude <= deadZone) {
            return 0;
        }
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

    updateConfig(newOptions) {
        this.config = { ...this.config, ...newOptions };
        this.runtime = this.resolveRuntimeProfile();
        if (this.simulation) {
            this.simulation.alpha(0.5)
                .restart();
        }
    }

    play() {
        if (!this.simulation) {
            return;
        }
        this._simStart = performance.now();
        console.log("SIM PLAY | initial alpha:", this.simulation.alpha().toFixed(6));
        this.simulation.alpha(Math.max(this.simulation.alpha(), 0.35))
            .restart();
        this.isPaused = false;
    }

    pause() {
        if (!this.simulation) {
            return;
        }
        this.simulation.stop();
        this.isPaused = true;
    }

    stop() {
        console.log("SIM STOP called at tick", this._tickCount, "alpha", this.simulation.alpha());
        this.simulation.stop();
    }

    restart() {
        if (!this.simulation) {
            return;
        }
        this.simulation.nodes(this.nodes);
        this.simulation.alpha(1)
            .restart();
    }

    clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }
}
