globalThis.ChainMapUI = class {
    constructor() {
        this.rawData = null;
        this.renderer = null;
        this.selectedNodes = [];
        this._showMain = true;
        this._showSecondary = true;
        this._showRegular = false;
        this._showLabels = true;
        this._mainLinkPhysics = true;
        this._secondaryChainPhysics = true;
        this._buoyancyEnabled = true;
        this._repulsionEnabled = true;
        this._dampingEnabled = true;
        this._collisionsEnabled = true;
        this._activeForces = [];
        this._simNodes = [];
        this._simLinks = [];
        this._paused = true;
        this._simAlpha = 1;
        this._tickCount = 0;
        this._maxNodes = 0;
        this._nodePower = 0.5;
        this._physicsCfg = {
            baseLinkLength: globalThis.RENDER.physics.defaultBaseLinkLength,
            linkScoreMultiplier: 2,
            linkStrength: globalThis.RENDER.physics.defaultLinkStrength,
            secondaryLinkStrength: globalThis.RENDER.physics.defaultLinkStrength,
            buoyancyStrength: globalThis.RENDER.physics.defaultBuoyancyStrength,
            repulsionStrength: globalThis.RENDER.physics.defaultRepulsionStrength,
            repulsionRange: globalThis.RENDER.physics.defaultRepulsionRange,
            velocityDecay: globalThis.RENDER.physics.defaultVelocityDecay,
            nodeBaseSize: globalThis.RENDER.physics.defaultNodeBaseSize,
            alphaDecay: 0.0005,
            alphaMin: 0.02,
            minAreaPerNode: 40000,
            maxVelocity: 100,
        };
        this._initActiveForces();
    }

    async init() {
        this.ce();
        if (!this.container) {
            return;
        }
        this.loadFiltersFromStorage();
        this._initActiveForces();
        this.setupRenderer();
        if (!this.renderer || !this.renderer.canvas) {
            globalThis.showError("WebGL unavailable");
            if (this.loader) {
                this.loader.classList.add("hidden");
            }
            return;
        }
        this.renderer.resize();
        window.addEventListener("resize", () => {
            if (this.renderer) {
                this.renderer.resize();
            }
        });
        document.addEventListener("visibilitychange", () => {
            if (document.hidden) {
                this._stopSim();
            } else if (!this._paused && this._simNodes.length && this._simAlpha > 0.001) {
                this._startSim();
            }
        });
        this._listen();
        await this.loadData();
    }

    _startSim() {
        if (!this._simNodes.length) {
            return;
        }
        this._simAlpha = 1;
        this.renderer.onSimulationEnd = () => {
            const pi = document.getElementById("pause-icon");
            const pl = document.getElementById("play-icon");
            if (pi) {
                pi.classList.add("hidden");
            }
            if (pl) {
                pl.classList.remove("hidden");
            }
            this._paused = true;
        };
        this.renderer.startLoop(() => this._tick());
    }

    _stopSim() {
        this.renderer.stopLoop();
    }

    _restartSim() {
        const wasPaused = this._paused;
        this._stopSim();
        this.applyFilters();
        if (!wasPaused && this._simNodes.length) {
            this._startSim();
        }
    }

    _tick() {
        try {
            const nodes = this._simNodes;
            if (!nodes.length) {
                return false;
            }

            for (const n of nodes) {
                n.fx = 0;
                n.fy = 0;
            }

            for (const fnName of this._activeForces) {
                this[fnName]();
            }

            this._simAlpha *= (1 - this._physicsCfg.alphaDecay);
            if (this._simAlpha < this._physicsCfg.alphaMin) {
                this._simAlpha = this._physicsCfg.alphaMin;
            }

            this._tickCount++;
            if (this.statPhysics) {
                this.statPhysics.textContent = this._tickCount;
            }

            return true;
        } catch (e) {
            globalThis.showError("Physics tick error: " + e.message);
            return true;
        }
    }

    _sync() {
        const h = (onId, offId, show) => {
            const on = document.getElementById(onId);
            const off = document.getElementById(offId);
            if (on) {
                on.classList.toggle("hidden", !show);
            }
            if (off) {
                off.classList.toggle("hidden", show);
            }
        };
        h("tag-icon-on", "tag-icon-off", this._showLabels);
        if (this.renderer) {
            this.renderer.labelsVisible = this._showLabels;
        }
        h("main-chains-icon-on", "main-chains-icon-off", this._showMain);
        h("secondary-chains-icon-on", "secondary-chains-icon-off", this._showSecondary);
        h("regular-links-icon-on", "regular-links-icon-off", this._showRegular);
    }

    render(filteredNodes, filteredEdges, stats) {
        if (!filteredNodes.length) {
            this._setStats(0, 0, 0, 0);
            if (this.renderer) {
                this.renderer.render({
                    nodes: [],
                    links: [],
                    selectedIds: [],
                    world: { x: 0, y: 0, width: 800, height: 600 },
                    nodeBaseSize: this._physicsCfg.nodeBaseSize,
                });
            }
            if (this.loader) {
                this.loader.classList.add("hidden");
            }
            return;
        }

        const comps = new Set(filteredNodes.map(n => String(n.component)));
        this._setStats(filteredNodes.length, filteredEdges.length, comps.size, stats.total_chains || 0);

        const cScale = globalThis.d3.scaleLinear()
            .domain(globalThis.RENDER.node.colorDomain)
            .range(globalThis.RENDER.node.colorRange);
        const compSize = {};
        if (this.rawData.components) {
            for (const [id, m] of Object.entries(this.rawData.components)) {
                compSize[id] = m.length;
            }
        }

        const simN = filteredNodes.map(d => ({
            ...d,
            _fill: cScale(d.score),
            _compSize: compSize[String(d.component)],
            _chainPrev: null,
            _chainNext: null,
            _chainId: null,
            _allChains: null,
            vx: 0,
            vy: 0,
            fx: 0,
            fy: 0,
        }));

        const nM = new Map(simN.map(n => [n.id, n]));
        const mcE = new Set();
        const chI = new Map();

        if (this.rawData.chains) {
            for (const ch of this.rawData.chains) {
                for (let i = 0; i < ch.nodes.length; i++) {
                    const id = ch.nodes[i];
                    const ex = chI.get(id);
                    if (!ex) {
                        chI.set(id, {
                            chainId: ch.id,
                            prev: i > 0 ? ch.nodes[i - 1] : null,
                            next: i < ch.nodes.length - 1 ? ch.nodes[i + 1] : null,
                            chainLen: ch.nodes.length,
                            allChains: [ch.id],
                        });
                    } else {
                        ex.allChains.push(ch.id);
                        if (ex.chainLen < ch.nodes.length) {
                            ex.chainId = ch.id;
                            ex.prev = i > 0 ? ch.nodes[i - 1] : null;
                            ex.next = i < ch.nodes.length - 1 ? ch.nodes[i + 1] : null;
                            ex.chainLen = ch.nodes.length;
                        }
                    }
                    if (i < ch.nodes.length - 1) {
                        mcE.add(ch.nodes[i] + "|" + ch.nodes[i + 1]);
                        mcE.add(ch.nodes[i + 1] + "|" + ch.nodes[i]);
                    }
                }
            }
        }
        for (const n of simN) {
            const i = chI.get(n.id);
            if (i) {
                n._chainPrev = i.prev;
                n._chainNext = i.next;
                n._chainId = i.chainId;
                n._allChains = i.allChains;
            }
        }

        const simL = [];
        const exL = new Set();
        for (const d of filteredEdges) {
            const s = nM.get(d.source);
            const t = nM.get(d.target);
            if (s && t) {
                const mc = mcE.has(d.source + "|" + d.target);
                simL.push({ source: s, target: t, isMainChain: mc });
                exL.add(d.source + "|" + d.target);
                exL.add(d.target + "|" + d.source);
            }
        }
        if (this.rawData.chains) {
            for (const ch of this.rawData.chains) {
                for (let i = 0; i < ch.nodes.length - 1; i++) {
                    const a = ch.nodes[i];
                    const b = ch.nodes[i + 1];
                    const s = nM.get(a);
                    const t = nM.get(b);
                    if (s && t && !exL.has(a + "|" + b)) {
                        simL.push({ source: s, target: t, isMainChain: true });
                        exL.add(a + "|" + b);
                        exL.add(b + "|" + a);
                    }
                }
            }
        }

        const deg = new Map();
        for (const l of simL) {
            deg.set(l.source.id, (deg.get(l.source.id) || 0) + 1);
            deg.set(l.target.id, (deg.get(l.target.id) || 0) + 1);
        }
        const np = this._nodePower;
        for (const n of simN) {
            const d = deg.get(n.id) || 0;
            n._deg = d;
            n._radius = this._physicsCfg.nodeBaseSize + (np === 0 ? 0 : Math.pow(d, np));
        }

        const oldPM = new Map((this._simNodes || []).map(n => [n.id, n]));
        this._simNodes = simN;
        this._simLinks = simL;

        const mapSize = Math.sqrt(simN.length * this._physicsCfg.minAreaPerNode);
        for (const n of simN) {
            const old = oldPM.get(n.id);
            if (old) {
                n.x = old.x;
                n.y = old.y;
                n.vx = old.vx || 0;
                n.vy = old.vy || 0;
            } else {
                n.x = (Math.random() - 0.5) * mapSize;
                n.y = (Math.random() - 0.5) * mapSize;
                n.vx = 0;
                n.vy = 0;
            }
        }
        this._mapHalf = mapSize / 2;

        const world = {
            x: -this._mapHalf - globalThis.RENDER.border.padding,
            y: -this._mapHalf - globalThis.RENDER.border.padding,
            width: mapSize + globalThis.RENDER.border.padding * 2,
            height: mapSize + globalThis.RENDER.border.padding * 2,
        };

        if (this.renderer) {
            this.renderer._showMainLinks = this._showMain;
            this.renderer._showRegularLinks = this._showRegular;
        }

        this.renderer.render({
            nodes: simN,
            links: simL,
            selectedIds: this.selectedNodes,
            world,
            nodeBaseSize: this._physicsCfg.nodeBaseSize,
        });
        if (this.loader) {
            this.loader.classList.add("hidden");
        }

        this._paused = true;
        this._simAlpha = 1;
        this._tickCount = 0;
        this._sync();

        const pi = document.getElementById("pause-icon");
        const pl = document.getElementById("play-icon");
        if (pi) {
            pi.classList.add("hidden");
        }
        if (pl) {
            pl.classList.remove("hidden");
        }
    }

    _setStats(n, e, c, ch) {
        if (this.statNodes) {
            this.statNodes.textContent = n.toLocaleString();
        }
        if (this.statComparisons) {
            this.statComparisons.textContent = e.toLocaleString();
        }
        if (this.statComponents) {
            this.statComponents.textContent = c.toLocaleString();
        }
        if (this.statChains) {
            this.statChains.textContent = ch.toLocaleString();
        }
    }

    cleanup() {
        if (this.renderer) {
            this._stopSim();
            this.renderer.destroy();
            this.renderer = null;
        }
        this.rawData = null;
        this.selectedNodes = [];
        this._simNodes = [];
        this._simLinks = [];
    }
};

window.chainMapUI = new globalThis.ChainMapUI();
window.Sections = window.Sections || {};
window.Sections.chains = globalThis.ChainMapUI;
