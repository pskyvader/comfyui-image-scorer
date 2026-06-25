globalThis.ChainMapUI.prototype._initActiveForces = function () {
    this._activeForces = [];
    if (this._mainLinkPhysics) {
        this._activeForces.push("_applyLinkSprings");
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
    this._activeForces.push("_applyVelocity");
};

globalThis.ChainMapUI.prototype._applyLinkSprings = function () {
    const cfg = this._physicsCfg;
    const links = this._simLinks;
    if (!this._mainLinkPhysics) {
        return;
    }
    const alpha = this._simAlpha;
    const linkK = Math.min(cfg.linkStrength * alpha, 1);
    const secondaryLinkK = Math.min(cfg.secondaryLinkStrength * alpha, 1);
    for (let i = 0; i < links.length; i++) {
        const l = links[i];
        if (!l.isMainChain) {
            continue;
        }
        const isDirect = l.source._chainNext === l.target.id || l.source._chainPrev === l.target.id || l.target._chainNext === l.source.id || l.target._chainPrev === l.source.id;
        if (!isDirect && !this._secondaryChainPhysics) {
            continue;
        }
        const effLinkK = isDirect ? linkK : secondaryLinkK;
        const s = l.source;
        const t = l.target;
        const dx = t.x - s.x;
        const dy = t.y - s.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const targetDist = cfg.baseLinkLength + cfg.baseLinkLength * Math.abs(s.score - t.score) * cfg.linkScoreMultiplier;
        const error = (dist - targetDist) / dist;
        const halfCorr = error * effLinkK * 0.5;
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
    const cellSize = repRng;
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
                    const rx = n.x - other.x;
                    const ry = n.y - other.y;
                    const dist = Math.sqrt(rx * rx + ry * ry);
                    if (dist < repRng && dist > 0) {
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

globalThis.ChainMapUI.prototype._applyVelocity = function () {
    const cfg = this._physicsCfg;
    const nodes = this._simNodes;
    const half = this._mapHalf || 0;
    const decayFactor = this._dampingEnabled ? (1 - cfg.velocityDecay) : 1;
    const maxV = this._dampingEnabled ? cfg.maxVelocity : Infinity;
    for (const n of nodes) {
        n.vx = (n.vx || 0) * decayFactor + n.fx;
        n.vy = (n.vy || 0) * decayFactor + n.fy;
        if (this._dampingEnabled) {
            if (n.vx > maxV) {
                n.vx = maxV;
            } else if (n.vx < -maxV) {
                n.vx = -maxV;
            }
            if (n.vy > maxV) {
                n.vy = maxV;
            } else if (n.vy < -maxV) {
                n.vy = -maxV;
            }
        }
        n.x += n.vx;
        n.y += n.vy;
        if (half > 0) {
            if (n.x > half) {
                n.x = half;
                if (n.vx > 0) {
                    n.vx = 0;
                }
            } else if (n.x < -half) {
                n.x = -half;
                if (n.vx < 0) {
                    n.vx = 0;
                }
            }
            if (n.y > half) {
                n.y = half;
                if (n.vy > 0) {
                    n.vy = 0;
                }
            } else if (n.y < -half) {
                n.y = -half;
                if (n.vy < 0) {
                    n.vy = 0;
                }
            }
        }
    }
};
