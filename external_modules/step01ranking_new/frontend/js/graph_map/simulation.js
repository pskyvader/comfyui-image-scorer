/**
 * Chain Simulation Logic (Physics Engine)
 */

class ChainSimulation {
    constructor(nodes, links, components, options = {}) {
        this.nodes = nodes;
        this.links = links;
        this.components = Array.from(components || []);
        
        // Use global split physics defaults and scaling multipliers
        const simLinks = typeof SIMULATION_LINKS !== 'undefined' ? SIMULATION_LINKS : {};
        const simNodes = typeof SIMULATION_NODES !== 'undefined' ? SIMULATION_NODES : {};
        const simWorld = typeof SIMULATION_WORLD !== 'undefined' ? SIMULATION_WORLD : {};

        this.config = { ...simLinks, ...simNodes, ...simWorld, ...options };

        this.config.scaling = {
            small: {
                ...(simLinks.scaling?.small || {}),
                ...(simNodes.scaling?.small || {}),
                ...(simWorld.scaling?.small || {}),
                ...(options.scaling?.small || {}),
            },
            medium: {
                ...(simLinks.scaling?.medium || {}),
                ...(simNodes.scaling?.medium || {}),
                ...(simWorld.scaling?.medium || {}),
                ...(options.scaling?.medium || {}),
            },
            large: {
                ...(simLinks.scaling?.large || {}),
                ...(simNodes.scaling?.large || {}),
                ...(simWorld.scaling?.large || {}),
                ...(options.scaling?.large || {}),
            },
        };

        this.simulation = null;
        this.linkForce = null;
        this.onTick = options.onTick || null;
        this.onEnd = options.onEnd || null;
        this.isPaused = false;

        this.nodeById = new Map();
        this.nodeDegree = new Map();
        this.componentNodes = new Map();
        this.runtime = this.resolveRuntimeProfile();
        this.simulationLinks = this.links;
        this.effectiveWidth = this.config.minWorldSize;
        this.effectiveHeight = this.config.minWorldSize;
        this.constraintPhase = 0;
        this.lastRenderAt = 0;
        this.simulationStartAt = 0;
        this.currentLinkStrengthFactor = this.config.linkStrengthInitialFactor;
    }

    resolveRuntimeProfile() {
        const nodeCount = this.nodes.length;
        const linkCount = this.links.length;
        const workUnits = nodeCount + (linkCount * 1.5);
        const mediumGraph = workUnits >= 9000;
        const largeGraph = workUnits >= 30000;
        const sizeKey = largeGraph ? "large" : (mediumGraph ? "medium" : "small");
        const scale = this.config.scaling[sizeKey] || this.config.scaling.small;

        return {
            canvasMode: true,
            enableCollision: !mediumGraph && nodeCount <= 1800,
            enableMarkers: false,
            enableLabels: nodeCount < 900,
            enableDrag: true,
            enablePausedGroupDrag: true,
            enableTooltips: nodeCount < 12000,
            enablePointerHits: nodeCount < 50000,
            drawLinks: linkCount < 180000,
            progressiveReveal: mediumGraph || nodeCount > 500,
            usePerLinkOpacity: !largeGraph && linkCount <= 12000,
            linkGlobalOpacity: largeGraph ? 0.12 : 0.18,
            linkVisibilityZoomThreshold: largeGraph ? 0.08 : (mediumGraph ? 0.03 : 0),
            renderIntervalMs: largeGraph ? 48 : (mediumGraph ? 32 : this.config.renderIntervalMs),
            chargeDistanceMax: this.scalePositive(this.config.chargeDistanceMax, scale.chargeDistanceMax),
            gravityStrength: this.config.gravityStrength * this.safeMultiplier(scale.gravityStrength),
            velocityDecay: this.scalePositive(this.config.velocityDecay, scale.velocityDecay),
            alphaDecay: this.scalePositive(this.config.alphaDecay, scale.alphaDecay),
            linkDistanceScale: this.safeMultiplier(scale.linkDistance),
            linkStrengthScale: this.safeMultiplier(scale.linkStrength),
            scoreDistanceMultiplierScale: this.safeMultiplier(scale.scoreDistanceMultiplier),
            countDistanceMultiplierScale: this.safeMultiplier(scale.countDistanceMultiplier),
            maxLinkDistance: this.scalePositive(this.config.maxLinkDistance, scale.maxLinkDistance),
            worldScale: this.scalePositive(this.config.worldScale, scale.worldScale),
            nodeJitter: this.scalePositive(this.config.nodeJitter, scale.nodeJitter),
            boundaryPadding: this.scalePositive(this.config.boundaryPadding, scale.boundaryPadding),
            simulationLinkStride: largeGraph
                ? Math.max(1, Math.ceil(linkCount / 60000))
                : (mediumGraph ? Math.max(1, Math.ceil(linkCount / 45000)) : 1),
            constraintPhases: largeGraph
                ? Math.max(1, Math.ceil(linkCount / 50000))
                : (mediumGraph ? Math.max(1, Math.ceil(linkCount / 90000)) : 1),
            constraintSlack: this.scalePositive(this.config.linkConstraintSlack, scale.constraintSlack),
            constraintPull: this.scalePositive(this.config.linkConstraintPull, scale.constraintPull),
            linkStiffnessRampMs: this.scalePositive(this.config.linkStrengthRampMs, scale.linkStiffnessRampMs),
            startPaused: workUnits >= 110000,
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
        this.setupSimulation();
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
            const componentId = String(node.component ?? "");
            if (!this.componentNodes.has(componentId)) {
                this.componentNodes.set(componentId, []);
            }
            this.componentNodes.get(componentId).push(node);
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
            this.nodeDegree.set(sourceId, (this.nodeDegree.get(sourceId) || 0) + 1);
            this.nodeDegree.set(targetId, (this.nodeDegree.get(targetId) || 0) + 1);
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
        const padding = this.runtime.boundaryPadding;
        const componentIds = this.components.map(component => String(component));
        const count = Math.max(1, componentIds.length);
        const columns = Math.max(1, Math.ceil(Math.sqrt(count)));
        const rows = Math.max(1, Math.ceil(count / columns));
        const usableWidth = Math.max(1, this.effectiveWidth - (padding * 2));
        const usableHeight = Math.max(1, this.effectiveHeight - (padding * 2));
        const centers = new Map();

        componentIds.forEach((componentId, index) => {
            const column = index % columns;
            const row = Math.floor(index / columns);
            centers.set(componentId, {
                x: padding + (((column + 0.5) / columns) * usableWidth),
                y: padding + (((row + 0.5) / rows) * usableHeight),
            });
        });

        for (const [componentId, members] of this.componentNodes.entries()) {
            const center = centers.get(componentId) || {
                x: this.effectiveWidth / 2,
                y: this.effectiveHeight / 2,
            };

            for (const node of members) {
                const score = this.getFiniteNumber(node.score, 0.5);
                const degree = this.nodeDegree.get(node.id) || 0;
                const isolatedMultiplier = degree === 0 ? 0.85 : 1;
                const scoreBias = this.getScoreBias(score);
                const jitter = this.runtime.nodeJitter;

                node.x = center.x + ((Math.random() - 0.5) * jitter);
                node.y = this.clamp(
                    center.y + ((Math.random() - 0.5) * jitter),
                    padding,
                    this.effectiveHeight - padding,
                );
                node.vx = 0;
                node.vy = 0;
                node._charge = this.config.chargeStrength * isolatedMultiplier;
                node._scoreBias = scoreBias;
                node._scoreForce = scoreBias * this.runtime.gravityStrength;
            }
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
            if (!source || !target) continue;

            const sourceDegree = this.nodeDegree.get(source.id) || 1;
            const targetDegree = this.nodeDegree.get(target.id) || 1;
            const scoreGap = Math.abs(
                this.getFiniteNumber(source.score, 0.5) - this.getFiniteNumber(target.score, 0.5),
            );
            const countWeight = Math.log1p(Math.max(
                this.getFiniteNumber(source.comparison_count, 0),
                this.getFiniteNumber(target.comparison_count, 0),
            ));

            link.distance = Math.min(
                maxLinkDistance,
                baseDistance
                + (baseDistance * scoreGap * scoreMultiplier)
                + (baseDistance * countWeight * countMultiplier),
            );
            link._baseStrength = Math.min(
                1,
                this.config.linkStrength
                * this.runtime.linkStrengthScale
                * (1 / Math.sqrt(Math.max(1, Math.min(sourceDegree, targetDegree)))),
            );
            link._opacity = this.runtime.usePerLinkOpacity
                ? Math.max(0.08, 1 - (link.distance / (maxLinkDistance * 1.15)))
                : this.runtime.linkGlobalOpacity;
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
        const initialFactor = this.clamp(this.config.linkStrengthInitialFactor, 0, this.config.linkStrengthFinalFactor);
        this.currentLinkStrengthFactor = initialFactor;
        this.simulationStartAt = performance.now();

        this.linkForce = d3.forceLink(this.simulationLinks)
            .id(node => node.id)
            .distance(link => link.distance)
            .strength(link => link._baseStrength * this.currentLinkStrengthFactor);

        this.simulation = d3.forceSimulation(this.nodes)
            .force("link", this.linkForce)
            .force("charge", d3.forceManyBody()
                .strength(node => node._charge)
                .distanceMax(this.runtime.chargeDistanceMax))
            .force("scoreY", this.createScoreForce())
            .velocityDecay(this.runtime.velocityDecay)
            .alphaDecay(this.runtime.alphaDecay);

        if (this.runtime.enableCollision) {
            this.simulation.force("collide", d3.forceCollide(node => {
                const count = this.getFiniteNumber(node.comparison_count, 0);
                return this.config.collisionRadius + Math.log1p(count);
            }));
        }

        this.simulation.on("tick", () => this.handleTick());
        this.simulation.on("end", () => {
            this.onTick?.(this.nodes, this.links, this.runtime);
            this.onEnd?.(this.runtime);
        });
    }

    handleTick() {
        const now = performance.now();
        this.updateLinkStiffness(now);
        this.applyLinkConstraints();

        for (const node of this.nodes) {
            this.applyBoundaryBehavior(node);
        }

        if (!this.onTick) return;

        if (now - this.lastRenderAt >= this.runtime.renderIntervalMs) {
            this.lastRenderAt = now;
            this.onTick(this.nodes, this.links, this.runtime);
        }
    }

    createScoreForce() {
        let nodes = [];
        const force = alpha => {
            for (let i = 0; i < nodes.length; i++) {
                const node = nodes[i];
                if (!node._scoreForce) continue;
                node.vy -= node._scoreForce * alpha;
            }
        };
        force.initialize = allNodes => { nodes = allNodes; };
        return force;
    }

    updateLinkStiffness(now) {
        if (!this.linkForce) return;

        const initial = this.clamp(this.config.linkStrengthInitialFactor, 0, this.config.linkStrengthFinalFactor);
        const final = Math.max(initial, this.config.linkStrengthFinalFactor);
        const duration = Math.max(1, this.runtime.linkStiffnessRampMs);
        const progress = this.clamp((now - this.simulationStartAt) / duration, 0, 1);
        const eased = Math.pow(progress, Math.max(0.01, this.config.linkStrengthRampCurve));
        const factor = initial + ((final - initial) * eased);

        if (progress >= 1 && this.currentLinkStrengthFactor === final && this._annealingComplete) {
            return;
        }

        this.currentLinkStrengthFactor = factor;
        if (progress >= 1) this._annealingComplete = true;

        this.linkForce.strength(link => link._baseStrength * this.currentLinkStrengthFactor);

        const distFactor = 3.0 - (2.0 * eased);
        this.linkForce.distance(link => (link.distance || 50) * distFactor);
    }

    applyLinkConstraints() {
        if (!this.links.length) return;

        const phases = this.runtime.constraintPhases;
        const phaseOffset = this.constraintPhase;
        const initial = this.clamp(this.config.linkStrengthInitialFactor, 0, this.config.linkStrengthFinalFactor);
        const final = Math.max(initial, this.config.linkStrengthFinalFactor);
        const normalizedStiffness = final === initial
            ? 1
            : this.clamp((this.currentLinkStrengthFactor - initial) / (final - initial), 0, 1);

        const relaxedSlack = this.runtime.constraintSlack + (this.config.linkConstraintInitialSlackExtra || 0);
        const slack = this.runtime.constraintSlack + ((relaxedSlack - this.runtime.constraintSlack) * (1 - normalizedStiffness));
        const pull = this.runtime.constraintPull * (0.15 + (0.85 * normalizedStiffness));

        for (let index = phaseOffset; index < this.links.length; index += phases) {
            const link = this.links[index];
            const source = link.source;
            const target = link.target;
            if (!source || !target || typeof source !== "object" || typeof target !== "object") continue;

            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const distanceSq = (dx * dx) + (dy * dy);
            if (!distanceSq) continue;

            const distance = Math.sqrt(distanceSq);
            const limit = link.distance * slack;
            if (distance <= limit) continue;

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
        } else if (node.x > maxX) {
            node.x = maxX - ((node.x - maxX) * bounce);
            if (node.vx > 0) node.vx = -node.vx * bounce;
        }

        if (node.y < minY) {
            node.y = minY + ((minY - node.y) * bounce);
            if (node.vy < 0) node.vy = -node.vy * bounce;
        } else if (node.y > maxY) {
            node.y = maxY - ((node.y - maxY) * bounce);
            if (node.vy > 0) node.vy = -node.vy * bounce;
        }
    }

    dragLinkedNodesWhilePaused(subject, dx, dy) {
        if (!this.isPaused) return;
        const offsetX = this.getFiniteNumber(dx, 0) * (this.config.pausedLinkedDragInfluence || 1);
        const offsetY = this.getFiniteNumber(dy, 0) * (this.config.pausedLinkedDragInfluence || 1);
        if (!offsetX && !offsetY) return;

        const componentId = String(subject.component ?? "");
        const members = this.componentNodes.get(componentId);
        if (!members || !members.length) return;

        const minX = this.runtime.boundaryPadding;
        const maxX = this.effectiveWidth - this.runtime.boundaryPadding;
        const minY = this.runtime.boundaryPadding;
        const maxY = this.effectiveHeight - this.runtime.boundaryPadding;

        for (const node of members) {
            if (node === subject) continue;
            node.x = this.clamp(node.x + offsetX, minX, maxX);
            node.y = this.clamp(node.y + offsetY, minY, maxY);
            node.vx = 0;
            node.vy = 0;
        }
        this.onTick?.(this.nodes, this.links, this.runtime);
    }

    getScoreBias(score) {
        const centeredScore = (this.getFiniteNumber(score, 0.5) - 0.5) * 2;
        const magnitude = Math.abs(centeredScore);
        const deadZone = this.config.neutralDeadZone || 0;
        if (magnitude <= deadZone) return 0;
        const normalized = (magnitude - deadZone) / (1 - deadZone);
        return Math.sign(centeredScore) * Math.pow(normalized, this.config.scoreExponent || 2);
    }

    getFiniteNumber(value, fallback = 0) {
        const number = Number(value);
        return Number.isFinite(number) ? number : fallback;
    }

    getNodeId(nodeOrId) {
        return typeof nodeOrId === "object" ? nodeOrId.id : nodeOrId;
    }

    getSimNode(id) {
        return this.nodeById.get(id) || null;
    }

    updateConfig(newOptions) {
        const scaling = newOptions.scaling
            ? {
                small: { ...this.config.scaling.small, ...(newOptions.scaling.small || {}) },
                medium: { ...this.config.scaling.medium, ...(newOptions.scaling.medium || {}) },
                large: { ...this.config.scaling.large, ...(newOptions.scaling.large || {}) },
            }
            : this.config.scaling;

        this.config = { ...this.config, ...newOptions, scaling };
        this.runtime = this.resolveRuntimeProfile();
        if (this.simulation) this.simulation.alpha(0.5).restart();
    }

    play() {
        if (!this.simulation) return;
        for (const node of this.nodes) { node.vx = 0; node.vy = 0; }
        this.simulation.alpha(Math.max(this.simulation.alpha(), 0.35)).restart();
        this.isPaused = false;
    }

    pause() {
        if (!this.simulation) return;
        this.simulation.stop();
        this.isPaused = true;
    }

    stop() { this.simulation?.stop(); }

    restart() {
        if (!this.simulation) return;
        this.simulation.nodes(this.nodes);
        this.simulation.alpha(1).restart();
    }

    safeMultiplier(value) {
        const number = Number(value);
        return Number.isFinite(number) ? number : 1;
    }

    scalePositive(base, multiplier) {
        const result = Number(base) * this.safeMultiplier(multiplier);
        return Number.isFinite(result) ? Math.max(0.000001, result) : base;
    }

    clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }
}
