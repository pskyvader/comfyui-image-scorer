globalThis.ChainMapUI.prototype.applyFilters = function () {
    if (!this.rawData) {
        return;
    }
    if (this.loader) {
        this.loader.classList.remove("hidden");
    }
    const minComp = globalThis.sliderToValue(parseInt(this.minCompFilter.value), globalThis.SLIDER.comp);
    const maxComp = globalThis.sliderToValue(parseInt(this.maxCompFilter.value), globalThis.SLIDER.comp);
    const minChain = globalThis.sliderToValue(parseInt(this.minChainFilter.value), globalThis.SLIDER.chain);
    const maxChain = globalThis.sliderToValue(parseInt(this.maxChainFilter.value), globalThis.SLIDER.chain);
    const minCC = this.minCompCountFilter ? globalThis.sliderToValue(parseInt(this.minCompCountFilter.value), globalThis.SLIDER.compCount) : 0;
    const maxCC = this.maxCompCountFilter ? globalThis.sliderToValue(parseInt(this.maxCompCountFilter.value), globalThis.SLIDER.compCount) : globalThis.SLIDER.compCount.max;
    this.saveFilters();
    const uMinC = minComp > globalThis.SLIDER.comp.min;
    const uMaxC = maxComp < globalThis.SLIDER.comp.max;
    const uMinH = minChain > globalThis.SLIDER.chain.min;
    const uMaxH = maxChain < globalThis.SLIDER.chain.max;
    const uMinCC = minCC > globalThis.SLIDER.compCount.min;
    const uMaxCC = maxCC < globalThis.SLIDER.compCount.max;
    const nodeMap = new Map(this.rawData.nodes.map(n => [n.id, n]));
    const valid = new Set();
    Object.entries(this.rawData.components)
        .forEach(([id, members]) => {
            const size = members.length;
            let maxH = 0;
            members.forEach((nid) => {
                const nodeEntry = nodeMap.get(nid);
                if (nodeEntry && nodeEntry.height > maxH) {
                    maxH = nodeEntry.height;
                }
            });
            let ccOk = true;
            if (uMinCC || uMaxCC) {
                ccOk = false;
                for (const nid of members) {
                    const nodeEntry = nodeMap.get(nid);
                    if (nodeEntry) {
                        const meetsMin = !uMinCC || nodeEntry.comparison_count >= minCC;
                        const meetsMax = !uMaxCC || nodeEntry.comparison_count <= maxCC;
                        if (meetsMin && meetsMax) {
                            ccOk = true;
                            break;
                        }
                    }
                }
            }
            if ((!uMinC || size >= minComp) && (!uMaxC || size <= maxComp) && (!uMinH || maxH >= minChain) && (!uMaxH || maxH <= maxChain) && ccOk) {
                valid.add(id.toString());
            }
        });
    let fn = this.rawData.nodes.filter(n => valid.has(String(n.component)));
    if (this._maxNodes > 0 && fn.length > this._maxNodes) {
        const pool = fn.slice();
        fn = [];
        const poolLength = pool.length;
        for (let i = 0; i < this._maxNodes; i++) {
            const idx = Math.floor(Math.random() * (poolLength - i));
            fn.push(pool[idx]);
            pool[idx] = pool[poolLength - i - 1];
        }
    }
    const ids = new Set(fn.map(n => n.id));
    const filteredEdges = this.rawData.edges.filter(e => ids.has(e.source) && ids.has(e.target));
    this.render(fn, filteredEdges, this.rawData.stats);
};

globalThis.ChainMapUI.prototype.resetView = function () {
    const sub = this.renderer;
    if (!sub) {
        return;
    }
    if (sub.worldBounds) {
        sub._fitToWorld(sub.worldBounds);
    }
};
