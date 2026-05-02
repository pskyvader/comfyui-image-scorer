class ChainSimulation {
    static defaults = {
        // BUOYANCY: Increased significantly. 
        // 1.0 score floats UP, 0.0 score sinks DOWN.
        gravityStrength: 2.5,

        // REPULSION: Only force moving nodes horizontally.
        chargeStrength: -100,
        chargeDistanceMax: 300,

        // TETHERS: Links have a natural length and a hard "rope" limit.
        linkDistance: 10,
        maxLinkDistance: 150,
        linkStrength: 0.1,

        // PHYSICS: 
        // Lower velocityDecay (0.1 - 0.2) makes them feel more fluid/slippery.
        velocityDecay: 0.1,
        alphaDecay: 0.0005, // Slower decay gives them time to reach the top/bottom
        collisionRadius: 20,

        simWidth: 5000,
        simHeight: 5000
    };

    constructor(nodes, links, options = {}) {
        this.nodes = nodes;
        this.links = links;
        this.config = { ...ChainSimulation.defaults, ...options };

        this.simulation = null;
        this.onTick = options.onTick || null;
        this.onEnd = options.onEnd || null;
        this.nodeDegree = {};
        this.isPaused = false;
    }

    initialize() {
        this.calculateNodeDegree();
        this.setupSimulation();
    }

    calculateNodeDegree() {
        this.nodeDegree = {};
        this.links.forEach(l => {
            const s = l.source.id || l.source;
            const t = l.target.id || l.target;
            this.nodeDegree[s] = (this.nodeDegree[s] || 0) + 1;
            this.nodeDegree[t] = (this.nodeDegree[t] || 0) + 1;
        });
    }

    /**
     * INDIVIDUAL BUOYANCY FORCE
     * Every node acts independently.
     */
    getBuoyancyForce() {
        let nodes;
        const force = (alpha) => {
            // We use alpha to gradually stabilize, but keep the strength high
            const strength = this.config.gravityStrength * alpha * 10;

            for (let i = 0, n = nodes.length; i < n; ++i) {
                const node = nodes[i];
                const score = Number(node.score ?? 0.5);

                // Score 1.0 (Balloon) -> -0.5 direction (UP)
                // Score 0.0 (Weight)  -> +0.5 direction (DOWN)
                const direction = 0.5 - score;

                node.vy += direction * strength;
            }
        };
        force.initialize = (_) => nodes = _;
        return force;
    }

    /**
     * TETHER CONSTRAINT
     * This acts like a physical string that cannot stretch past maxLinkDistance.
     */
    applyTetherConstraints() {
        const maxDist = this.config.maxLinkDistance;
        for (let i = 0; i < this.links.length; i++) {
            const link = this.links[i];
            const s = link.source;
            const t = link.target;

            if (typeof s === 'object' && typeof t === 'object') {
                const dx = t.x - s.x;
                const dy = t.y - s.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance > maxDist) {
                    const diff = (distance - maxDist) / distance;
                    const offsetX = dx * diff * 0.5;
                    const offsetY = dy * diff * 0.5;

                    s.x += offsetX;
                    s.y += offsetY;
                    t.x -= offsetX;
                    t.y -= offsetY;
                }
            }
        }
    }

    setupSimulation() {
        // Randomize positions across the whole universe
        this.nodes.forEach(n => {
            if (n.x === undefined) n.x = Math.random() * this.config.simWidth;
            if (n.y === undefined) n.y = Math.random() * this.config.simHeight;
        });

        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.links)
                .id(d => d.id)
                .distance(this.config.linkDistance)
                .strength(this.config.linkStrength))
            .force('charge', d3.forceManyBody()
                .strength(this.config.chargeStrength)
                .distanceMax(this.config.chargeDistanceMax))
            .force('buoyancy', this.getBuoyancyForce())
            .velocityDecay(this.config.velocityDecay)
            .alphaDecay(this.config.alphaDecay);

        if (this.nodes.length < 2500) {
            this.simulation.force('collide', d3.forceCollide(this.config.collisionRadius).iterations(2));
        }

        this.simulation.on('tick', () => {
            // Apply tether constraints before finalizing coordinates
            this.applyTetherConstraints();

            this.nodes.forEach(n => {
                // Keep within universe boundaries
                n.x = Math.max(25, Math.min(this.config.simWidth - 25, n.x));
                n.y = Math.max(25, Math.min(this.config.simHeight - 25, n.y));
            });

            if (this.onTick) this.onTick(this.nodes, this.links);
        });

        this.simulation.on('end', () => {
            if (this.onEnd) this.onEnd();
        });
    }

    // --- Control API ---

    getSimNode(id) {
        return this.nodes.find(n => (n.id === id || n === id));
    }

    updateConfig(newOpts) {
        this.config = { ...this.config, ...newOpts };
        if (this.simulation) {
            this.simulation.alpha(0.5).restart();
        }
    }

    play() {
        if (this.simulation) {
            this.simulation.alpha(0.3).restart(); this.isPaused = false;
        }
    }

    pause() {
        if (this.simulation) { this.simulation.stop(); this.isPaused = true; }
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