globalThis.WebGLPhysics = class {
    constructor() {
        this.gl = null;
        this._initialized = false;
    }

    init(gl, nodes, links) {
        this._lastError = null;
        this.gl = gl;
        if (!gl) {
            this._lastError = "WebGL 2.0 not available";
            return false;
        }

        const ext = gl.getExtension("EXT_color_buffer_float");
        if (!ext) {
            this._lastError = "EXT_color_buffer_float not supported (device may not support float textures)";
            return false;
        }

        const maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
        if (maxSize < 256) {
            this._lastError = "MAX_TEXTURE_SIZE too small: " + maxSize;
            return false;
        }

        this._nodeCount = nodes.length;
        if (this._nodeCount === 0) {
            this._lastError = "No nodes to simulate";
            return false;
        }
        const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
        const maxStateW = Math.min(2048, maxTex);
        this._stateTexW = Math.min(maxStateW, Math.max(256, 1 << Math.ceil(Math.log2(Math.sqrt(nodes.length)))));
        this._stateTexH = Math.ceil(nodes.length / this._stateTexW);
        this._stateTexSize = this._stateTexW * this._stateTexH;

        this._buildLinkData(nodes, links);

        const prog = this._createProgram();
        if (!prog) {
            return false;
        }
        this._prog = prog;
        this._uLoc = {};

        const unis = ["u_stateTexW", "u_linkTexW", "u_linkRangeTexW",
            "u_nodeCount", "u_alpha",
            "u_baseLinkLength", "u_linkScoreMultiplier", "u_linkStrength", "u_secondaryLinkStrength",
            "u_buoyancyStrength", "u_repulsionStrength", "u_repulsionRange",
            "u_velocityDecay", "u_maxVelocity", "u_mapHalf",
            "u_enableMainSprings", "u_enableSecondarySprings", "u_enableBuoyancy", "u_enableRepulsion", "u_enableCollisions", "u_enableDamping",
            "u_nodeState", "u_nodeStatic", "u_linkData", "u_nodeLinkRange"];
        for (const n of unis) {
            this._uLoc[n] = gl.getUniformLocation(prog, n);
        }

        this._setupQuad();
        if (!this._vao) {
            return false;
        }

        this._texState = [this._createTex2D(this._stateTexW, this._stateTexH), this._createTex2D(this._stateTexW, this._stateTexH)];
        this._texStatic = this._createTex2D(this._stateTexW, this._stateTexH);
        this._texLinkData = this._createTex2D(this._linkTexW, this._linkTexH);
        this._texLinkRange = this._createTex2D(this._stateTexW, this._stateTexH);
        this._fbo = [gl.createFramebuffer(), gl.createFramebuffer()];
        if (!this._texState[0] || !this._texState[1] || !this._texStatic || !this._texLinkData || !this._texLinkRange || !this._fbo[0] || !this._fbo[1]) {
            this._lastError = "Failed to create framebuffer objects";
            return false;
        }

        this._uploadNodeData(nodes);
        this._uploadLinkData();

        this._ping = 0;
        this._readbackBuf = new Float32Array(this._nodeCount * 4);
        this._tempCanvas = document.createElement("canvas");
        this._tempCtx = this._tempCanvas.getContext("2d");
        this._tempCanvas.width = 1;
        this._tempCanvas.height = 1;
        this._initialized = true;
        return true;
    }

    _buildLinkData(nodes, links) {
        const nodeCount = nodes.length;
        const idxMap = new Map();
        for (let i = 0; i < nodeCount; i++) {
            idxMap.set(nodes[i].id, i);
        }

        const adj = new Array(nodeCount);
        for (let i = 0; i < nodeCount; i++) {
            adj[i] = [];
        }

        for (const l of links) {
            const si = idxMap.get(l.source.id);
            const ti = idxMap.get(l.target.id);
            if (si === undefined || ti === undefined || si === ti) {
                continue;
            }
            const isDirect = l.source._chainNext === l.target.id || l.source._chainPrev === l.target.id || l.target._chainNext === l.source.id || l.target._chainPrev === l.source.id;
            let type = 2;
            if (l.isMainChain) {
                type = isDirect ? 0 : 1;
            }
            adj[si].push({ other: ti, type });
            adj[ti].push({ other: si, type });
        }

        this._nodeLinkRanges = new Float32Array(nodeCount * 4);
        const allData = [];
        let offset = 0;
        for (let i = 0; i < nodeCount; i++) {
            const entries = adj[i];
            const cnt = entries.length;
            this._nodeLinkRanges[i * 4] = offset;
            this._nodeLinkRanges[i * 4 + 1] = cnt;
            for (const e of entries) {
                allData.push(e.other, e.type, 0, 0);
            }
            offset += cnt;
        }
        this._linkDataArray = new Float32Array(allData);
        this._linkEntryCount = allData.length / 4;
        const maxTex = this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE);
        this._linkTexW = Math.min(4096, maxTex, Math.max(256, 1 << Math.ceil(Math.log2(Math.sqrt(this._linkEntryCount)))));
        this._linkTexH = Math.ceil(this._linkEntryCount / this._linkTexW);
        this._linkRangeTexW = this._stateTexW;
    }

    _createTex2D(w, h) {
        const gl = this.gl;
        const tex = gl.createTexture();
        if (!tex) {
            this._lastError = "Failed to create texture (" + w + "×" + h + ")";
            return null;
        }
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, w, h);
        const status = gl.getError();
        if (status !== gl.NO_ERROR) {
            this._lastError = "texStorage2D failed (size " + w + "×" + h + "): GL error " + status;
            gl.deleteTexture(tex);
            return null;
        }
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return tex;
    }

    _uploadNodeData(nodes) {
        const gl = this.gl;
        const data = new Float32Array(this._stateTexSize * 4);
        const statData = new Float32Array(this._stateTexSize * 4);
        for (let i = 0; i < nodes.length; i++) {
            const n = nodes[i];
            const idx = i * 4;
            data[idx] = n.x || 0;
            data[idx + 1] = n.y || 0;
            data[idx + 2] = n.vx || 0;
            data[idx + 3] = n.vy || 0;
            statData[idx] = n.score || 0.5;
            statData[idx + 1] = n._radius || 3;
            statData[idx + 2] = 0;
            statData[idx + 3] = 0;
        }
        gl.bindTexture(gl.TEXTURE_2D, this._texState[0]);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this._stateTexW, this._stateTexH, gl.RGBA, gl.FLOAT, data);
        gl.bindTexture(gl.TEXTURE_2D, this._texState[1]);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this._stateTexW, this._stateTexH, gl.RGBA, gl.FLOAT, data);
        gl.bindTexture(gl.TEXTURE_2D, this._texStatic);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this._stateTexW, this._stateTexH, gl.RGBA, gl.FLOAT, statData);
    }

    _uploadLinkData() {
        const gl = this.gl;
        const w = this._linkTexW;
        const h = this._linkTexH;
        const padded = new Float32Array(w * h * 4);
        const n = this._linkDataArray.length;
        for (let i = 0; i < n; i++) {
            padded[i] = this._linkDataArray[i];
        }
        gl.bindTexture(gl.TEXTURE_2D, this._texLinkData);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, gl.FLOAT, padded);

        const rangeData = new Float32Array(this._stateTexSize * 4);
        for (let i = 0; i < this._nodeCount; i++) {
            rangeData[i * 4] = this._nodeLinkRanges[i * 4];
            rangeData[i * 4 + 1] = this._nodeLinkRanges[i * 4 + 1];
        }
        gl.bindTexture(gl.TEXTURE_2D, this._texLinkRange);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this._stateTexW, this._stateTexH, gl.RGBA, gl.FLOAT, rangeData);
    }

    _setupQuad() {
        const gl = this.gl;
        this._vao = gl.createVertexArray();
        if (!this._vao) {
            this._lastError = "Failed to create VAO";
            return;
        }
        gl.bindVertexArray(this._vao);
        const buf = gl.createBuffer();
        if (!buf) {
            this._lastError = "Failed to create vertex buffer";
            gl.bindVertexArray(null);
            return;
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        gl.bindVertexArray(null);
    }

    _createProgram() {
        const gl = this.gl;
        const vsCode = `#version 300 es
layout(location = 0) in vec2 a_pos;
out vec2 v_uv;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_pos * 0.5 + 0.5;
}`;
        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, vsCode);
        gl.compileShader(vs);
        if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
            this._lastError = "Vertex shader error: " + gl.getShaderInfoLog(vs);
            return null;
        }
        const unis = [];
        const fsCode = this._getFragShaderSource(unis);
        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, fsCode);
        gl.compileShader(fs);
        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
            this._lastError = "Fragment shader error: " + gl.getShaderInfoLog(fs);
            return null;
        }
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            this._lastError = "Program link error: " + gl.getProgramInfoLog(prog);
            return null;
        }
        gl.deleteShader(vs);
        gl.deleteShader(fs);
        return prog;
    }

    _getFragShaderSource(unis) {
        return `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outState;
uniform sampler2D u_nodeState;
uniform sampler2D u_nodeStatic;
uniform sampler2D u_linkData;
uniform sampler2D u_nodeLinkRange;
uniform float u_stateTexW;
uniform float u_linkTexW;
uniform float u_linkRangeTexW;
uniform float u_nodeCount;
uniform float u_alpha;
uniform float u_baseLinkLength;
uniform float u_linkScoreMultiplier;
uniform float u_linkStrength;
uniform float u_secondaryLinkStrength;
uniform float u_buoyancyStrength;
uniform float u_repulsionStrength;
uniform float u_repulsionRange;
uniform float u_velocityDecay;
uniform float u_maxVelocity;
uniform float u_mapHalf;
uniform float u_enableMainSprings;
uniform float u_enableSecondarySprings;
uniform float u_enableBuoyancy;
uniform float u_enableRepulsion;
uniform float u_enableCollisions;
uniform float u_enableDamping;
void main() {
    int stexW = int(u_stateTexW + 0.5);
    int sx = int(v_uv.x * float(stexW));
    int sy = int(v_uv.y * float(textureSize(u_nodeState, 0).y));
    int idx = sy * stexW + sx;
    int nodeCount = int(u_nodeCount + 0.5);
    if (idx >= nodeCount) { outState = vec4(0.0); return; }
    vec4 state = texelFetch(u_nodeState, ivec2(sx, sy), 0);
    vec4 stat = texelFetch(u_nodeStatic, ivec2(sx, sy), 0);
    float x = state.x, y = state.y, vx = state.z, vy = state.w;
    float score = stat.x, radius = stat.y;
    float fx = 0.0, fy = 0.0;
    vec4 range = texelFetch(u_nodeLinkRange, ivec2(sx, sy), 0);
    int linkStart = int(range.x + 0.5);
    int linkCount = int(range.y + 0.5);
    int linkTW = int(u_linkTexW + 0.5);
    float alpha = u_alpha;
    float mainK = min(u_linkStrength * alpha, 1.0);
    float secK = min(u_secondaryLinkStrength * alpha, 1.0);
    float bLinkLen = u_baseLinkLength;
    float lScoreMult = u_linkScoreMultiplier;
    if ((u_enableMainSprings > 0.0 || u_enableSecondarySprings > 0.0) && linkCount > 0) {
        for (int i = 0; i < 4096; i++) {
            if (i >= linkCount) break;
            int li = linkStart + i;
            int ltx = li % linkTW;
            int lty = li / linkTW;
            vec4 entry = texelFetch(u_linkData, ivec2(ltx, lty), 0);
            int oIdx = int(entry.x + 0.5);
            float ltype = entry.y;
            int ox = oIdx % stexW;
            int oy = oIdx / stexW;
            vec4 oState = texelFetch(u_nodeState, ivec2(ox, oy), 0);
            vec4 oStat = texelFetch(u_nodeStatic, ivec2(ox, oy), 0);
            float odx = oState.x - x;
            float ody = oState.y - y;
            float dist = sqrt(odx * odx + ody * ody);
            if (dist < 0.001) dist = 1.0;
            float oRadius = oStat.y;
            float targetDist = bLinkLen + bLinkLen * abs(score - oStat.x) * lScoreMult + radius + oRadius;
            float error = (dist - targetDist) / dist;
            float k = ltype < 0.5 ? mainK : secK;
            if ((ltype < 0.5 && u_enableMainSprings > 0.0) || (ltype >= 0.5 && u_enableSecondarySprings > 0.0)) {
                float halfCorr = error * k * 0.5;
                fx += odx * halfCorr;
                fy += ody * halfCorr;
            }
        }
    }
    if (u_enableBuoyancy > 0.0) {
        float diff = score - 0.5;
        float signDiff = diff >= 0.0 ? 1.0 : -1.0;
        float f = signDiff * pow(abs(diff) * 100.0, u_buoyancyStrength) / 100.0 * alpha;
        fy += f;
    }
    float repRng = u_repulsionRange;
    float repRng2 = repRng * repRng;
    float repStr = u_repulsionStrength * alpha;
    for (int j = 0; j < 65536; j++) {
        if (j >= nodeCount) break;
        if (j == idx) continue;
        int nx = j % stexW;
        int ny = j / stexW;
        vec4 ns = texelFetch(u_nodeState, ivec2(nx, ny), 0);
        float rdx = x - ns.x;
        float rdy = y - ns.y;
        float d2 = rdx * rdx + rdy * rdy;
        if (d2 < repRng2 && d2 > 0.001 && u_enableRepulsion > 0.0) {
            float d = sqrt(d2);
            float f = repStr * (repRng - d) / repRng / d;
            fx += rdx * f;
            fy += rdy * f;
        }
        if (u_enableCollisions > 0.0 && d2 > 0.01) {
            vec4 nStat = texelFetch(u_nodeStatic, ivec2(nx, ny), 0);
            float oRadius = nStat.y;
            float minDist = radius + oRadius + 1.0;
            if (d2 < minDist * minDist) {
                float d = sqrt(d2);
                float overlap = minDist - d;
                float f = overlap / d * alpha;
                fx += rdx * f;
                fy += rdy * f;
            }
        }
    }
    float decay = u_enableDamping > 0.0 ? (1.0 - u_velocityDecay) : 1.0;
    float maxV = u_enableDamping > 0.0 ? u_maxVelocity : 1e10;
    vx = vx * decay + fx;
    vy = vy * decay + fy;
    if (vx > maxV) vx = maxV;
    if (vx < -maxV) vx = -maxV;
    if (vy > maxV) vy = maxV;
    if (vy < -maxV) vy = -maxV;
    x += vx;
    y += vy;
    if (!(x > -1e15 && x < 1e15)) x = 0.0;
    if (!(y > -1e15 && y < 1e15)) y = 0.0;
    if (u_mapHalf > 0.0) {
        float hm = u_mapHalf;
        if (x > hm) { x = hm; if (vx > 0.0) vx = 0.0; }
        if (x < -hm) { x = -hm; if (vx < 0.0) vx = 0.0; }
        if (y > hm) { y = hm; if (vy > 0.0) vy = 0.0; }
        if (y < -hm) { y = -hm; if (vy < 0.0) vy = 0.0; }
    }
    outState = vec4(x, y, vx, vy);
}`;
    }

    updateNodes(nodes) {
        const gl = this.gl;
        const data = new Float32Array(this._stateTexSize * 4);
        for (let i = 0; i < nodes.length && i < this._nodeCount; i++) {
            const n = nodes[i];
            data[i * 4] = n.x || 0;
            data[i * 4 + 1] = n.y || 0;
            data[i * 4 + 2] = n.vx || 0;
            data[i * 4 + 3] = n.vy || 0;
        }
        gl.bindTexture(gl.TEXTURE_2D, this._texState[this._ping]);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this._stateTexW, this._stateTexH, gl.RGBA, gl.FLOAT, data);
        gl.bindTexture(gl.TEXTURE_2D, this._texState[1 - this._ping]);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this._stateTexW, this._stateTexH, gl.RGBA, gl.FLOAT, data);
    }

    tick(alpha, cfg, activeForces, mapHalf, dampingEnabled) {
        const gl = this.gl;
        if (!this._initialized || !this._prog) {
            return;
        }

        const readTex = this._texState[1 - this._ping];
        const writeFbo = this._fbo[this._ping];

        gl.bindFramebuffer(gl.FRAMEBUFFER, writeFbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this._texState[this._ping], 0);

        gl.viewport(0, 0, this._stateTexW, this._stateTexH);
        gl.useProgram(this._prog);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, readTex);
        gl.uniform1i(this._uLoc.u_nodeState, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this._texStatic);
        gl.uniform1i(this._uLoc.u_nodeStatic, 1);

        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, this._texLinkData);
        gl.uniform1i(this._uLoc.u_linkData, 2);

        gl.activeTexture(gl.TEXTURE3);
        gl.bindTexture(gl.TEXTURE_2D, this._texLinkRange);
        gl.uniform1i(this._uLoc.u_nodeLinkRange, 3);

        gl.uniform1f(this._uLoc.u_stateTexW, this._stateTexW);
        gl.uniform1f(this._uLoc.u_linkTexW, this._linkTexW);
        gl.uniform1f(this._uLoc.u_linkRangeTexW, this._linkRangeTexW);
        gl.uniform1f(this._uLoc.u_nodeCount, this._nodeCount);
        gl.uniform1f(this._uLoc.u_alpha, alpha);
        gl.uniform1f(this._uLoc.u_baseLinkLength, cfg.baseLinkLength);
        gl.uniform1f(this._uLoc.u_linkScoreMultiplier, cfg.linkScoreMultiplier);
        gl.uniform1f(this._uLoc.u_linkStrength, cfg.linkStrength);
        gl.uniform1f(this._uLoc.u_secondaryLinkStrength, cfg.secondaryLinkStrength);
        gl.uniform1f(this._uLoc.u_buoyancyStrength, cfg.buoyancyStrength);
        gl.uniform1f(this._uLoc.u_repulsionStrength, cfg.repulsionStrength);
        gl.uniform1f(this._uLoc.u_repulsionRange, cfg.repulsionRange);
        gl.uniform1f(this._uLoc.u_velocityDecay, cfg.velocityDecay);
        gl.uniform1f(this._uLoc.u_maxVelocity, cfg.maxVelocity);
        gl.uniform1f(this._uLoc.u_mapHalf, mapHalf || 0);
        gl.uniform1f(this._uLoc.u_enableMainSprings, activeForces.includes("_applyDirectLinkSprings") ? 1 : 0);
        gl.uniform1f(this._uLoc.u_enableSecondarySprings, activeForces.includes("_applySecondaryLinkSprings") ? 1 : 0);
        gl.uniform1f(this._uLoc.u_enableBuoyancy, activeForces.includes("_applyBuoyancy") ? 1 : 0);
        gl.uniform1f(this._uLoc.u_enableRepulsion, activeForces.includes("_applyRepulsion") ? 1 : 0);
        gl.uniform1f(this._uLoc.u_enableCollisions, activeForces.includes("_applyCollisions") ? 1 : 0);
        gl.uniform1f(this._uLoc.u_enableDamping, dampingEnabled ? 1 : 0);

        gl.bindVertexArray(this._vao);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.bindVertexArray(null);

        this._ping = 1 - this._ping;
    }

    readback(destNodes) {
        const gl = this.gl;
        if (!this._initialized) {
            return;
        }

        const fbo = this._fbo[1 - this._ping];
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this._texState[1 - this._ping], 0);

        const pixelsPerRow = Math.min(this._stateTexW, 4096);
        const rowsPerRead = Math.ceil(this._nodeCount / pixelsPerRow);
        const buf = new Float32Array(pixelsPerRow * 4);

        for (let row = 0; row < rowsPerRead; row++) {
            const y = row;
            const count = Math.min(pixelsPerRow, this._nodeCount - row * pixelsPerRow);
            gl.readPixels(0, y, count, 1, gl.RGBA, gl.FLOAT, buf);
            for (let i = 0; i < count; i++) {
                const di = (row * pixelsPerRow + i);
                if (di < destNodes.length) {
                    const rx = buf[i * 4], ry = buf[i * 4 + 1];
                    const rvx = buf[i * 4 + 2], rvy = buf[i * 4 + 3];
                    if (isFinite(rx) && isFinite(ry)) {
                        destNodes[di].x = rx;
                        destNodes[di].y = ry;
                        destNodes[di].vx = isFinite(rvx) ? rvx : 0;
                        destNodes[di].vy = isFinite(rvy) ? rvy : 0;
                    }
                }
            }
        }

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    destroy() {
        const gl = this.gl;
        if (!gl) {
            return;
        }
        if (this._vao) {
            gl.deleteVertexArray(this._vao);
        }
        if (this._prog) {
            gl.deleteProgram(this._prog);
        }
        for (const t of this._texState || []) {
            if (t) {
                gl.deleteTexture(t);
            }
        }
        if (this._texStatic) {
            gl.deleteTexture(this._texStatic);
        }
        if (this._texLinkData) {
            gl.deleteTexture(this._texLinkData);
        }
        if (this._texLinkRange) {
            gl.deleteTexture(this._texLinkRange);
        }
        for (const f of this._fbo || []) {
            if (f) {
                gl.deleteFramebuffer(f);
            }
        }
        this._initialized = false;
        this.gl = null;
    }
};
