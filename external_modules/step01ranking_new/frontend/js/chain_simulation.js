class ChainSimulation {
    static defaults = {
        // BUOYANCY: Increased significantly.
        // 1.0 score floats UP, 0.0 score sinks DOWN.
        gravityStrength: 0.5,

        // REPULSION: Only force moving nodes horizontally.
        chargeStrength: -100,
        chargeDistanceMax: 70,

        // TETHERS: Links have a natural length and a hard "rope" limit.
        linkDistance: 3,
        // maxLinkDistance: 1000,
        linkStrength: 1,
        scoreDistanceMultiplier: 100,
        countDistanceMultiplier: 10,

        // PHYSICS:
        // Lower velocityDecay (0.1 - 0.2) makes them feel more fluid/slippery.
        velocityDecay: 0.05,
        alphaDecay: 0.0005, // Slower decay gives them time to reach the top/bottom
        collisionRadius: 67,

        simWidth: 300,
        simHeight: 300,
    };

    constructor(nodes, links, components, options = {}) {
        this.nodes = nodes;
        this.links = links;
        this.components = components;
        this.config = { ...ChainSimulation.defaults, ...options };

        this.simulation = null;
        this.onTick = options.onTick || null;
        this.onEnd = options.onEnd || null;
        this.nodeDegree = {};
        this.isPaused = false;
        // effective size based on node count
        const scale = Math.sqrt(this.nodes.length);
        this.effectiveWidth = this.config.simWidth * scale;
        this.effectiveHeight = this.config.simHeight * scale;
        console.log(`Effective simulation size: ${this.effectiveWidth} x ${this.effectiveHeight}`);
    }

    // make a list of

    initialize() {
        this.calculateNodeDegree();
        this.setupSimulation();
    }

    calculateNodeDegree() {
        this.nodeDegree = {};
        this.links.forEach((l) => {
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
                const direction = 0.5 - (score);
                const sign = direction < 0 ? -1 : 1;

                node.vy += sign * (direction * direction) * strength;
            }
        };
        force.initialize = _ => nodes = _;
        return force;
    }

    /**
     * TETHER CONSTRAINT
     * This acts like a physical string that cannot stretch past maxLinkDistance.
     */

    defineRopeLength() {
        const distance = this.config.linkDistance;
        const scoreDistanceMultiplier = this.config.scoreDistanceMultiplier;
        const countDistanceMultiplier = this.config.countDistanceMultiplier;
        for (let i = 0; i < this.links.length; i++) {
            const link = this.links[i];
            const s = this.nodes.find(n => n.id === link.source);
            const t = this.nodes.find(n => n.id === link.target);
            const count = Math.max(s.comparison_count, t.comparison_count);
            const score_distance = Math.abs(s.score - t.score);
            let relativeDistance = distance;
            relativeDistance += (distance * score_distance * scoreDistanceMultiplier);
            relativeDistance += distance * count * countDistanceMultiplier;

            link.distance = relativeDistance;
        }
    }

    applyTetherConstraints() {
        // const maxDist = this.config.maxLinkDistance;
        for (let i = 0; i < this.links.length; i++) {
            const link = this.links[i];
            const s = link.source;
            const t = link.target;
            const linkDistance = link.distance;

            // const confidence = Math.max(s.confidence, t.confidence);
            // const maxRelative = maxDist + (maxDist * confidence * 1);
            const maxRelative = linkDistance;
            // if (typeof s === 'object' && typeof t === 'object') {
            const dx = t.x - s.x;
            const dy = t.y - s.y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance > maxRelative) {
                const diff = (distance - maxRelative) / distance;
                const offsetX = dx * diff * 0.5;
                const offsetY = dy * diff * 0.5;

                s.x += offsetX;
                s.y += offsetY;
                t.x -= offsetX;
                t.y -= offsetY;
            }
            // }
        }
    }

    defineInitialPositions() {
        // 1. Map component strings to pre-defined coordinates
        const componentCoords = {};
        // let i = 0;
        // const totalComponents = this.components.size;

        this.components.forEach((comp) => {
            const cid = String(comp);
            componentCoords[cid] = {
                x: Math.floor(Math.random() * (this.effectiveWidth)),
                y: Math.floor(Math.random() * (this.effectiveHeight)),
                // y: Math.floor((i / totalComponents) * this.config.simHeight)
            };
            // i++;
        });

        // 2. Single pass through nodes to assign positions and reset velocity
        this.nodes.forEach((n) => {
            const coords = componentCoords[String(n.component)];
            if (coords) {
                n.x = coords.x + Math.random() * 100;
                n.y = coords.y + Math.random() * 100;
                n.vx = 0;
                n.vy = 0;
                // console.log(n)
            }
        });
    }

    setupSimulation() {
        this.defineInitialPositions();

        this.defineRopeLength();

        this.simulation = d3.forceSimulation(this.nodes)
            .force("link", d3.forceLink(this.links)
                .id(d => d.id)
                // .distance(this.config.linkDistance)
                .distance(d => d.distance)
                .strength(d => this.config.linkStrength / d.distance))
            .force("charge", d3.forceManyBody()
                .strength(this.config.chargeStrength)
                .distanceMax(this.config.chargeDistanceMax))
            .force("buoyancy", this.getBuoyancyForce())
            .velocityDecay(this.config.velocityDecay)
            .alphaDecay(this.config.alphaDecay);

        // if (this.nodes.length < 2500) {
        //     this.simulation.force("collide",
        //         d3.forceCollide(
        //             this.config.collisionRadius,
        //         )
        //             .iterations(2));
        // }

        this.simulation.on("tick", () => {
            // Apply tether constraints before finalizing coordinates
            this.applyTetherConstraints();

            this.nodes.forEach((n) => {
                // Keep within universe boundaries
                n.x = Math.max(25, Math.min(this.effectiveWidth - 25, n.x));
                n.y = Math.max(25, Math.min(this.effectiveHeight - 25, n.y));
            });

            if (this.onTick) {
                this.onTick(this.nodes, this.links);
            }
        });

        this.simulation.on("end", () => {
            if (this.onEnd) {
                this.onEnd();
            }
        });
    }

    // --- Control API ---

    getSimNode(id) {
        return this.nodes.find(n => (n.id === id || n === id));
    }

    updateConfig(newOpts) {
        this.config = { ...this.config, ...newOpts };
        if (this.simulation) {
            this.simulation.alpha(0.5)
                .restart();
        }
    }

    play() {
        if (this.simulation) {
            this.nodes.forEach((n) => {
                n.vx = 0;
                n.vy = 0;
            });
            this.simulation
                .alpha(0.3)
                .restart();
            this.isPaused = false;
        }
    }

    pause() {
        if (this.simulation) {
            this.simulation.stop();
            this.isPaused = true;
        }
    }

    stop() {
        if (this.simulation) {
            this.simulation.stop();
        }
    }

    restart() {
        if (this.simulation) {
            this.simulation.nodes(this.nodes);
            this.simulation.alpha(1)
                .restart();
        }
    }
}
