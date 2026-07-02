globalThis.ChainMapUI.prototype._initActiveForces = function () {
    this._activeForces = [];
    this._forceIndex = 0;
    this._tickSkipAccum = 0;
    this._forceStats = {};
    this._statsCursor = 0;
    this._chartHistory = {};
    const all = {
        _applyDirectLinkSprings: "Main Springs",
        _applySecondaryLinkSprings: "Sec Springs",
        _applyBuoyancy: "Buoyancy",
        _applyRepulsion: "Repulsion",
        _applyCollisions: "Collisions",
        _tick: "Tick Total",
    };
    for (const [fn, label] of Object.entries(all)) {
        this._forceStats[fn] = { label, samples: [], live: 0, avg10s: 0, avg60s: 0 };
        this._chartHistory[fn] = [];
    }
    if (this._mainLinkPhysics) {
        this._activeForces.push("_applyDirectLinkSprings");
    }
    if (this._secondaryChainPhysics) {
        this._activeForces.push("_applySecondaryLinkSprings");
    }
    if (this._buoyancyEnabled) {
        this._activeForces.push("_applyBuoyancy");
    }
    if (this._repulsionEnabled) {
        this._activeForces.push("_applyRepulsion");
    }
    if (this._collisionsEnabled) {
        this._activeForces.push("_applyCollisions");
    }
};

globalThis.ChainMapUI.prototype._recordStat = function (fnName, duration) {
    const stats = this._forceStats[fnName];
    if (!stats) {
        return;
    }
    const now = performance.now();
    stats.samples.push({ t: now, d: duration });
    const cutoff = now - 70000;
    while (stats.samples.length > 0 && stats.samples[0].t < cutoff) {
        stats.samples.shift();
    }
    const ch = this._chartHistory[fnName];
    if (ch) {
        ch.push(duration);
        if (ch.length > 100) {
            ch.splice(0, ch.length - 100);
        }
    }
};

globalThis.ChainMapUI.prototype._updateStats = function () {
    const fnNames = Object.keys(this._forceStats);
    if (!fnNames.length) {
        return;
    }
    const now = performance.now();
    const idx = this._statsCursor % (fnNames.length * 3);
    const fnIdx = Math.floor(idx / 3);
    const windowType = idx % 3;
    const fnName = fnNames[fnIdx];
    const stats = this._forceStats[fnName];
    if (!stats || !stats.samples.length) {
        this._statsCursor++;
        return;
    }
    if (windowType === 0) {
        const recent = stats.samples.slice(-20);
        stats.live = recent.length ? recent.reduce((s, r) => s + r.d, 0) / recent.length : 0;
    } else if (windowType === 1) {
        const cutoff = now - 10000;
        const recent = stats.samples.filter(s => s.t >= cutoff);
        stats.avg10s = recent.length ? recent.reduce((s, r) => s + r.d, 0) / recent.length : 0;
    } else {
        const cutoff = now - 60000;
        const recent = stats.samples.filter(s => s.t >= cutoff);
        stats.avg60s = recent.length ? recent.reduce((s, r) => s + r.d, 0) / recent.length : 0;
    }
    this._statsCursor++;
    this._updateStatsView();
};

globalThis.ChainMapUI.prototype._applyDirectLinkSprings = function () {
    const cfg = this._physicsCfg;
    const links = this._simLinks;
    const alpha = this._simAlpha;
    const linkK = Math.min(cfg.linkStrength * alpha, 1);
    for (let i = 0; i < links.length; i++) {
        const l = links[i];
        if (!l.isMainChain) {
            continue;
        }
        const isDirect = l.source._chainNext === l.target.id || l.source._chainPrev === l.target.id || l.target._chainNext === l.source.id || l.target._chainPrev === l.source.id;
        if (!isDirect) {
            continue;
        }
        const s = l.source;
        const t = l.target;
        const dx = t.x - s.x;
        const dy = t.y - s.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const targetDist = cfg.baseLinkLength + cfg.baseLinkLength * Math.abs(s.score - t.score) * cfg.linkScoreMultiplier + (s._radius || 0) + (t._radius || 0);
        const error = (dist - targetDist) / dist;
        const halfCorr = error * linkK * 0.5;
        const fx = dx * halfCorr;
        const fy = dy * halfCorr;
        s.x += fx;
        s.y += fy;
        t.x -= fx;
        t.y -= fy;
    }
};

globalThis.ChainMapUI.prototype._applySecondaryLinkSprings = function () {
    const cfg = this._physicsCfg;
    const links = this._simLinks;
    const alpha = this._simAlpha;
    const linkK = Math.min(cfg.secondaryLinkStrength * alpha, 1);
    for (let i = 0; i < links.length; i++) {
        const l = links[i];
        if (!l.isMainChain) {
            continue;
        }
        const isDirect = l.source._chainNext === l.target.id || l.source._chainPrev === l.target.id || l.target._chainNext === l.source.id || l.target._chainPrev === l.source.id;
        if (isDirect) {
            continue;
        }
        const s = l.source;
        const t = l.target;
        const dx = t.x - s.x;
        const dy = t.y - s.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const targetDist = cfg.baseLinkLength + cfg.baseLinkLength * Math.abs(s.score - t.score) * cfg.linkScoreMultiplier + (s._radius || 0) + (t._radius || 0);
        const error = (dist - targetDist) / dist;
        const halfCorr = error * linkK * 0.5;
        const fx = dx * halfCorr;
        const fy = dy * halfCorr;
        s.x += fx;
        s.y += fy;
        t.x -= fx;
        t.y -= fy;
    }
};

globalThis.ChainMapUI.prototype._applyBuoyancy = function () {
    const cfg = this._physicsCfg;
    const nodes = this._simNodes;
    const alpha = this._simAlpha;
    const p = cfg.buoyancyStrength;
    for (const n of nodes) {
        const diff = n.score - 0.5;
        n.fy += Math.sign(diff) * Math.pow(Math.abs(diff) * 100, p) / 100 * alpha;
    }
};

globalThis.ChainMapUI.prototype._applyRepulsion = function () {
    const cfg = this._physicsCfg;
    const nodes = this._simNodes;
    const alpha = this._simAlpha;
    const repStr = cfg.repulsionStrength * alpha;
    const repRng = cfg.repulsionRange;
    const repRng2 = repRng * repRng;
    const cellSize = repRng;
    const grid = new Map();
    for (const n of nodes) {
        const cx = Math.floor(n.x / cellSize);
        const cy = Math.floor(n.y / cellSize);
        const key = ((cx + 32768) << 16) | (cy + 32768);
        let bucket = grid.get(key);
        if (!bucket) {
            bucket = [];
            grid.set(key, bucket);
        }
        bucket.push(n);
    }
    for (const n of nodes) {
        const cx = Math.floor(n.x / cellSize);
        const cy = Math.floor(n.y / cellSize);
        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                const key = ((cx + dx + 32768) << 16) | (cy + dy + 32768);
                const cells = grid.get(key);
                if (!cells) {
                    continue;
                }
                for (const other of cells) {
                    if (other === n) {
                        continue;
                    }
                    const rx = n.x - other.x;
                    const ry = n.y - other.y;
                    const dist2 = rx * rx + ry * ry;
                    if (dist2 < repRng2 && dist2 > 0) {
                        const dist = Math.sqrt(dist2);
                        const force = repStr * (repRng - dist) / repRng / dist;
                        n.fx += rx * force;
                        n.fy += ry * force;
                    }
                }
            }
        }
    }
    const half = this._mapHalf || 0;
    const boundaryPad = 60;
    if (half > 0) {
        for (const n of nodes) {
            const dL = n.x + half;
            if (dL < boundaryPad) {
                n.fx += cfg.repulsionStrength * (boundaryPad - dL) / boundaryPad;
            }
            const dR = half - n.x;
            if (dR < boundaryPad) {
                n.fx -= cfg.repulsionStrength * (boundaryPad - dR) / boundaryPad;
            }
            const dB = n.y + half;
            if (dB < boundaryPad) {
                n.fy += cfg.repulsionStrength * (boundaryPad - dB) / boundaryPad;
            }
            const dT = half - n.y;
            if (dT < boundaryPad) {
                n.fy -= cfg.repulsionStrength * (boundaryPad - dT) / boundaryPad;
            }
        }
    }
};

globalThis.ChainMapUI.prototype._applyCollisions = function () {
    const nodes = this._simNodes;
    const alpha = this._simAlpha;
    const strength = 1 * alpha;
    const cellSize = 50;
    const grid = new Map();
    for (const n of nodes) {
        const cx = Math.floor(n.x / cellSize);
        const cy = Math.floor(n.y / cellSize);
        const key = cx + "," + cy;
        if (!grid.has(key)) {
            grid.set(key, []);
        }
        grid.get(key)
            .push(n);
    }
    for (const n of nodes) {
        const cx = Math.floor(n.x / cellSize);
        const cy = Math.floor(n.y / cellSize);
        const rn = Math.max(n._radius || 3, 3);
        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                const cells = grid.get((cx + dx) + "," + (cy + dy));
                if (!cells) {
                    continue;
                }
                for (const other of cells) {
                    if (other === n) {
                        continue;
                    }
                    const ro = Math.max(other._radius || 3, 3);
                    const minDist = rn + ro + 1;
                    const rx = other.x - n.x;
                    const ry = other.y - n.y;
                    const dist2 = rx * rx + ry * ry;
                    if (dist2 < minDist * minDist && dist2 > 0.01) {
                        const dist = Math.sqrt(dist2);
                        const overlap = minDist - dist;
                        const force = overlap / dist * strength * 0.5;
                        n.x -= rx * force;
                        n.y -= ry * force;
                        other.x += rx * force;
                        other.y += ry * force;
                    }
                }
            }
        }
    }
};
