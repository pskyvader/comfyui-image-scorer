class ChainSimulation {
    static config = {
        linkDistance: 100,
        linkDistanceMin: 50,
        linkDistanceMax: 2000,
        useDynamicDistance: true,
        linkStrength: 0.1,
        scoreSimilarityStrength: 0.9,
        chargeStrength: -300,
        disconnectedChargeMultiplier: 2.0,
        chargeDistanceMax: 1000,
        componentRepulsion: 0.3,
        collisionRadius: 25,
        scorePositionRange: 1500,
        forceXStrength: 0,
        forceYStrength: 0.05,
        velocityDecay: 0.1,
        alphaDecay: 0.02,
        yMargin: 0.01,
        scoreSpread: 5.0
    };

    constructor(nodes, links, options = {}) {
        this.nodes = nodes;
        this.links = links;
        this.width = options.width || 800;
        this.height = options.height || 600;
        this.simulation = null;
        this.isPaused = false;
        this.onTick = options.onTick || null;
        this.onEnd = options.onEnd || null;
        this.nodeDegree = {};
        this.componentAvgScore = {};
        this.componentNodes = {};
    }

    initialize() {
        this.calculateNodeDegree();
        this.calculateComponentData();
        this.setupSimulation();
    }

    calculateNodeDegree() {
        this.nodeDegree = {};
        this.links.forEach(l => {
            const sourceId = l.source.id || l.source;
            const targetId = l.target.id || l.target;
            this.nodeDegree[sourceId] = (this.nodeDegree[sourceId] || 0) + 1;
            this.nodeDegree[targetId] = (this.nodeDegree[targetId] || 0) + 1;
        });
    }

    calculateComponentData() {
        this.componentNodes = {};
        this.nodes.forEach(n => {
            const c = n.component;
            if (!this.componentNodes[c]) this.componentNodes[c] = [];
            this.componentNodes[c].push(n);
        });

        const components = Object.values(this.componentNodes);
        this.componentAvgScore = {};
        components.forEach(comp => {
            const avg = comp.reduce((sum, n) => sum + n.score, 0) / comp.length;
            this.componentAvgScore[comp[0].component] = avg;
        });
    }

    setInitialPositions() {
        const componentCenters = {};
        const padding = 80;
        const usableWidth = this.width - padding * 2;
        const usableHeight = this.height - padding * 2;

        const components = Object.values(this.componentNodes);
        components.forEach(compNodes => {
            componentCenters[compNodes[0].component] = {
                x: padding + Math.random() * usableWidth,
                y: padding + Math.random() * usableHeight
            };
        });

        this.nodes.forEach(n => {
            const compCenter = componentCenters[n.component] || { x: this.width / 2, y: this.height / 2 };
            const deg = this.nodeDegree[n.id] || 0;
            const clusterRadius = deg > 0 ? 200 : 300;

            n.x = compCenter.x + (Math.random() - 0.5) * clusterRadius * 2;
            n.y = compCenter.y + (Math.random() - 0.5) * clusterRadius * 2;
        });
    }

    setupSimulation() {
        this.setInitialPositions();

        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.links)
                .id(d => d.id)
                .distance(d => this.getLinkDistance(d))
                .strength(d => this.getLinkStrength(d)))
            .force('charge', d3.forceManyBody()
                .strength(d => this.getChargeStrength(d))
                .distanceMax(ChainSimulation.config.chargeDistanceMax))
            .force('componentY', this.getComponentYForce())
            .velocityDecay(ChainSimulation.config.velocityDecay)
            .alphaDecay(ChainSimulation.config.alphaDecay);

        if (this.nodes.length < 5000) {
            this.simulation.force('collision', d3.forceCollide().radius(ChainSimulation.config.collisionRadius));
            this.simulation.force('component', this.getComponentForce());
        }

        this.simulation.on('tick', () => {
            if (this.onTick) {
                this.onTick(this.nodes, this.links);
            }
        });

        this.simulation.on('end', () => {
            if (this.onEnd) {
                this.onEnd();
            }
        });
    }

    getLinkDistance(d) {
        if (!ChainSimulation.config.useDynamicDistance) return ChainSimulation.config.linkDistance;

        const degA = this.nodeDegree[d.source.id || d.source] || 1;
        const degB = this.nodeDegree[d.target.id || d.target] || 1;
        const avgDeg = (degA + degB) / 2;
        const maxDeg = 10;
        const factor = Math.min(1, (avgDeg - 1) / (maxDeg - 1));

        return ChainSimulation.config.linkDistanceMin + (factor * (ChainSimulation.config.linkDistanceMax - ChainSimulation.config.linkDistanceMin));
    }

    getLinkStrength(d) {
        const base = ChainSimulation.config.linkStrength;
        const bonus = ChainSimulation.config.scoreSimilarityStrength;
        if (bonus <= 0) return base;

        const scoreDiff = Math.abs((d.source.score || 0) - (d.target.score || 0));
        const similarity = Math.max(0, 1 - scoreDiff);
        return base + (similarity * bonus);
    }

    getChargeStrength(d) {
        const base = ChainSimulation.config.chargeStrength;
        const deg = this.nodeDegree[d.id] || 0;
        return deg === 0 ? base * ChainSimulation.config.disconnectedChargeMultiplier : base;
    }

    getComponentYForce() {
        let nodesRef = this.nodes;
        const componentForce = (alpha) => {
            const range = ChainSimulation.config.scorePositionRange;
            const centerY = this.height / 2;

            nodesRef.forEach(n => {
                const avgScore = this.componentAvgScore[n.component] || 0.5;
                const targetY = centerY + (avgScore - 0.5) * 2 * range;
                const yStrength = ChainSimulation.config.forceYStrength * alpha;
                n.vy += (targetY - n.y) * yStrength;
            });
        };
        componentForce.initialize = (allNodes) => { nodesRef = allNodes; };
        return componentForce;
    }

    getComponentForce() {
        let nodes;
        const force = (alpha) => {
            if (ChainSimulation.config.componentRepulsion <= 0) return;

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
                    const strength = alpha * ChainSimulation.config.componentRepulsion * 20000;
                    const f = strength / (distSq + 100);
                    const fx = (dx / dist) * f;
                    const fy = (dy / dist) * f;

                    componentForces[id1].x += fx;
                    componentForces[id1].y += fy;
                    componentForces[id2].x -= fx;
                    componentForces[id2].y -= fy;
                }
            }

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

    getSimNode(nodeId) {
        return this.nodes.find(n => n.id === nodeId);
    }

    play() {
        if (!this.simulation) return;
        this.isPaused = false;
        this.simulation.alpha(0.3).restart();
    }

    pause() {
        if (!this.simulation) return;
        this.isPaused = true;
        this.simulation.stop();
    }

    stop() {
        if (this.simulation) this.simulation.stop();
    }

    restart() {
        if (this.simulation) {
            this.simulation.nodes(this.nodes);
            this.simulation.alpha(1).restart();
        }
    }
}