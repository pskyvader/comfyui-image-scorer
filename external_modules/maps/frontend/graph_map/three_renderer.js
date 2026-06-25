class ThreeGraphRenderer {
    constructor({ container }) {
        this.container = container;
        this._ready = false;
        this._ctx = null;

        const w = Math.max(container.clientWidth || 800, 1);
        const h = Math.max(container.clientHeight || 600, 1);
        const dpr = Math.min(window.devicePixelRatio, 2);

        this.nodesData = [];
        this.linksData = [];
        this.selectedIds = new Set();
        this.worldBounds = null;
        this.detailLevel = { showLabels: true, labelCap: 500, showLinks: true };
        this._showMainChains = true;
        this._showRegularLinks = true;
        this.highlightedChainId = null;
        this.onViewportChange = null;
        this._resizeObserver = new ResizeObserver(() => this.resize());
        this._resizeObserver.observe(container);

        const canvas = document.createElement('canvas');
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
        canvas.style.position = "absolute";
        canvas.style.top = "0";
        canvas.style.left = "0";
        canvas.style.display = "block";
        canvas.style.pointerEvents = "auto";
        canvas.style.touchAction = "none";

        canvas.classList.add('graph-map-canvas');
        container.insertBefore(canvas, container.firstChild);

        // Test WebGL availability on a throwaway canvas to avoid polluting the real one
        let webglOk = false;
        try {
            const testCanvas = document.createElement('canvas');
            testCanvas.width = 2;
            testCanvas.height = 2;
            webglOk = !!(testCanvas.getContext('webgl2', { alpha: true, antialias: true }) ||
                         testCanvas.getContext('webgl', { alpha: true, antialias: true }));
        } catch (_) {}

        if (webglOk) {
            try {
                this.renderer = new THREE.WebGLRenderer({
                    canvas: canvas,
                    alpha: true,
                    antialias: true,
                });
            } catch (_) {
                try {
                    this.renderer = new THREE.WebGLRenderer({
                        canvas: canvas,
                        alpha: true,
                        antialias: false,
                    });
                } catch (_) {
                    try {
                        this.renderer = new THREE.WebGLRenderer({
                            canvas: canvas,
                            alpha: false,
                            antialias: false,
                        });
                    } catch (_) {
                        webglOk = false;
                    }
                }
            }
        }

        if (!webglOk) {
            if (typeof showError === 'function') showError("WebGL not available — using Canvas 2D fallback.");
            this.renderer = null;
            this._ctx = canvas.getContext('2d');
            this._fallbackCanvas = canvas;
            this._w = w;
            this._h = h;
            this._dpr = dpr;
            this._fallback = true;
            this._panX = 0;
            this._panY = 0;
            this._zoomLevel = 1;
            this._isDragging = false;
            this._dragStartX = 0;
            this._dragStartY = 0;
            this._dragPanX = 0;
            this._dragPanY = 0;
            this._dragDx = 0;
            this._dragDy = 0;
            this._didDrag = false;
            this._snapCanvas = null;

            this._zoomPending = false;
            this._handleWheel = (ev) => {
                ev.preventDefault();
                // Allow zoom during chunked render — cancel chunks and re-render at new zoom
                if (this._chunkTimer) this._cancelChunked();
                const delta = ev.deltaY > 0 ? 1 / 1.2 : 1.2;
                const newZoom = this._zoomLevel * delta;
                this._zoomLevel = Math.max(0.05, Math.min(newZoom, 50));
                if (!this._zoomPending && this.nodesData.length) {
                    this._zoomPending = true;
                    requestAnimationFrame(() => {
                        this._zoomPending = false;
                        this._renderFallback();
                    });
                }
            };
            canvas.addEventListener('wheel', this._handleWheel, { passive: false });

            this._handleMouseDown = (ev) => {
                if (ev.button !== 0) return;
                // Cancel chunked render on drag start so user can pan immediately
                if (this._chunkTimer) this._cancelChunked();
                this._isDragging = true;
                this._didDrag = false;
                this._dragStartX = ev.clientX;
                this._dragStartY = ev.clientY;
                this._dragPanX = this._panX;
                this._dragPanY = this._panY;
                this._dragDx = 0;
                this._dragDy = 0;
                canvas.style.cursor = 'grabbing';
                this._snapCanvas = document.createElement('canvas');
                this._snapCanvas.width = this._fallbackCanvas.width;
                this._snapCanvas.height = this._fallbackCanvas.height;
                this._snapCanvas.getContext('2d').drawImage(this._fallbackCanvas, 0, 0);
            };
            canvas.addEventListener('mousedown', this._handleMouseDown);

            this._handleMouseMove = (ev) => {
                if (!this._isDragging) return;
                const dx = ev.clientX - this._dragStartX;
                const dy = ev.clientY - this._dragStartY;
                if (Math.abs(dx) > 3 || Math.abs(dy) > 3) this._didDrag = true;
                this._dragDx = dx;
                this._dragDy = dy;
                if (this._ctx && this._snapCanvas) {
                    this._ctx.save();
                    this._ctx.setTransform(this._dpr, 0, 0, this._dpr, 0, 0);
                    this._ctx.clearRect(0, 0, this._w, this._h);
                    this._ctx.drawImage(this._snapCanvas, dx, dy);
                    this._ctx.restore();
                }
            };
            window.addEventListener('mousemove', this._handleMouseMove);

            this._handleMouseUp = () => {
                if (!this._isDragging) return;
                this._isDragging = false;
                canvas.style.cursor = 'default';
                this._panX = this._dragPanX + this._dragDx;
                this._panY = this._dragPanY + this._dragDy;
                if (this._didDrag && this.nodesData.length) this._renderFallback();
                this._snapCanvas = null;
            };
            window.addEventListener('mouseup', this._handleMouseUp);

            this._ready = true;
            return;
        }

        this.renderer.setPixelRatio(dpr);
        this.renderer.setClearColor(0x000000, 0);
        this.renderer.setSize(w, h);
        container.insertBefore(this.renderer.domElement, container.firstChild);

        this.scene = new THREE.Scene();

        const aspect = w / h;
        const viewSize = Math.max(w, h);
        this.camera = new THREE.OrthographicCamera(
            -viewSize / 2, viewSize / 2,
            viewSize / 2, -viewSize / 2,
            -10000, 10000
        );
        this.camera.position.z = 1000;
        this._referenceZoom = 1;

        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableRotate = false;
        this.controls.screenSpacePanning = true;
        this.controls.zoomSpeed = 1.5;
        // Left-click pans (natural for 2D); right-click also pans; scroll zooms
        this.controls.mouseButtons = {
            LEFT: THREE.MOUSE.PAN,
            MIDDLE: THREE.MOUSE.DOLLY,
            RIGHT: THREE.MOUSE.PAN,
        };
        // Touch: two-finger pan, pinch zoom
        this.controls.touches = {
            ONE: THREE.TOUCH.PAN,
            TWO: THREE.TOUCH.DOLLY_PAN,
        };
        this.controls.update();

        this.nodePoints = null;
        this.selectedNodePoints = null;
        this.linkLines = null;
        this.mainChainLines = null;

        this.raycaster = new THREE.Raycaster();
        this._pointer = new THREE.Vector2();
        this._tempVec3 = new THREE.Vector3();

        this._ready = true;
        // Render on demand via controls change event (no continuous loop for static graph)
        this._onControlsChange = () => {
            if (!this._ready || this._fallback) return;
            this.renderer.render(this.scene, this.camera);
            if (typeof this.onViewportChange === 'function') {
                this.onViewportChange({ k: this.camera.zoom / this._referenceZoom });
            }
        };
        this.controls.addEventListener('change', this._onControlsChange);

        // Initial render
        requestAnimationFrame(() => {
            if (this._ready && !this._fallback) {
                this.renderer.render(this.scene, this.camera);
            }
        });
    }

    get canvas() {
        if (!this._ready) return null;
        if (this._fallback) return this._fallbackCanvas;
        return this.renderer.domElement;
    }

    get transform() {
        if (!this._ready || this._fallback) return { k: 1, x: 0, y: 0 };
        const zoom = this.camera.zoom / this._referenceZoom;
        return { k: zoom, x: 0, y: 0 };
    }

    render({ nodes, links, profile, selectedIds, world }, onComplete) {
        if (!this._ready) return;
        this._cancelChunked();

        this.resize();
        this.nodesData = nodes || [];
        this.linksData = links || [];
        this.selectedIds = new Set(selectedIds || []);
        this.worldBounds = world ? { x: world.x, y: world.y, width: world.width, height: world.height } : null;

        if (this._fallback) {
            this._panX = 0;
            this._panY = 0;
            this._zoomLevel = 1;
            this._renderFallback(onComplete);
            return;
        }
        this._buildScene();
        if (this.worldBounds) this._fitToWorld(this.worldBounds);
    }

    _updateTransformCache() {
        if (!this.worldBounds || !this.worldBounds.width || !this.worldBounds.height) {
            this._tc = { scale: 1, cx: 0, cy: 0, ox: 0, oy: 0 };
            return;
        }
        const diag = Math.sqrt(
            this.worldBounds.width * this.worldBounds.width +
            this.worldBounds.height * this.worldBounds.height
        );
        const baseScale = Math.min(this._w, this._h) / Math.max(diag, 1) * 0.9;
        const cx = this.worldBounds.x + this.worldBounds.width / 2;
        const cy = this.worldBounds.y + this.worldBounds.height / 2;
        const s = baseScale * this._zoomLevel;
        this._tc = {
            scale: s, cx, cy,
            ox: this._w / 2 + this._panX,
            oy: this._h / 2 + this._panY,
        };
    }

    _w2sX(wx) { return (wx - this._tc.cx) * this._tc.scale + this._tc.ox; }
    _w2sY(wy) { return -(wy - this._tc.cy) * this._tc.scale + this._tc.oy; }
    _s2wX(px) { return (px - this._tc.ox) / this._tc.scale + this._tc.cx; }
    _s2wY(py) { return -(py - this._tc.oy) / this._tc.scale + this._tc.cy; }

    _effectivePanX() { return this._isDragging ? this._dragPanX + this._dragDx : this._panX; }
    _effectivePanY() { return this._isDragging ? this._dragPanY + this._dragDy : this._panY; }

    _screenToFallbackWorld(px, py) {
        if (!this.worldBounds || !this.worldBounds.width || !this.worldBounds.height) {
            return { x: 0, y: 0 };
        }
        const diag = Math.sqrt(
            this.worldBounds.width * this.worldBounds.width +
            this.worldBounds.height * this.worldBounds.height
        );
        const baseScale = Math.min(this._w, this._h) / Math.max(diag, 1) * 0.9;
        const s = baseScale * this._zoomLevel;
        const cx = this.worldBounds.x + this.worldBounds.width / 2;
        const cy = this.worldBounds.y + this.worldBounds.height / 2;
        return {
            x: (px - this._w / 2 - this._effectivePanX()) / s + cx,
            y: -(py - this._h / 2 - this._effectivePanY()) / s + cy,
        };
    }

    // --- Chunked async render (keeps main thread responsive) ---

    _cancelChunked() {
        if (this._chunkTimer) {
            clearTimeout(this._chunkTimer);
            this._chunkTimer = null;
        }
        this._chunkOnComplete = null;
    }

    _renderFallback(onComplete) {
        this._cancelChunked();

        const ctx = this._ctx;
        if (!ctx) return;
        const w = this._w;
        const h = this._h;
        const dpr = this._dpr;

        ctx.save();
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, w, h);

        if (this.nodesData.length === 0) {
            ctx.fillStyle = '#888';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('WebGL not available — graph data loaded', w / 2, h / 2);
            ctx.restore();
            if (onComplete) onComplete();
            return;
        }

        this._updateTransformCache();

        // Draw nodes synchronously (fast) so user sees them immediately
        this._drawNodesNow(ctx);

        ctx.restore();

        // Start chunked link rendering (links overlay on top of nodes)
        this._chunkOnComplete = onComplete;
        this._chunkQueue = [];
        for (let i = 0; i < this.linksData.length; i++) {
            const link = this.linksData[i];
            if (link.isMainChain && !this._showMainChains) continue;
            if (!link.isMainChain && !this._showRegularLinks) continue;
            const sx = link.source.x, sy = link.source.y;
            const tx = link.target.x, ty = link.target.y;
            if (!isFinite(sx) || !isFinite(sy) || !isFinite(tx) || !isFinite(ty)) continue;
            this._chunkQueue.push(link.isMainChain ? 1 : 0, sx, sy, tx, ty);
        }

        this._chunkPos = 0;
        this._renderLinkChunk();
    }

    _drawNodesNow(ctx) {
        const diag = this.worldBounds ? Math.sqrt(
            this.worldBounds.width * this.worldBounds.width +
            this.worldBounds.height * this.worldBounds.height
        ) : 1;
        const nodeSize = Math.max(2, diag * 0.004) * Math.min(1, this._zoomLevel);
        const tc = this._tc;
        if (!tc) return;

        const useSelectedColor = this.selectedIds && this.selectedIds.size > 0;

        if (nodeSize <= 3) {
            for (let i = 0; i < this.nodesData.length; i++) {
                const node = this.nodesData[i];
                if (!isFinite(node.x) || !isFinite(node.y)) continue;
                const sx = (node.x - tc.cx) * tc.scale + tc.ox;
                const sy = -(node.y - tc.cy) * tc.scale + tc.oy;
                ctx.fillStyle = (useSelectedColor && this.selectedIds.has(node.id)) ? '#f59e0b' : (node._fill || '#888');
                ctx.fillRect(Math.round(sx) - 1, Math.round(sy) - 1, 3, 3);
            }
        } else {
            for (let i = 0; i < this.nodesData.length; i++) {
                const node = this.nodesData[i];
                if (!isFinite(node.x) || !isFinite(node.y)) continue;
                const sx = (node.x - tc.cx) * tc.scale + tc.ox;
                const sy = -(node.y - tc.cy) * tc.scale + tc.oy;
                ctx.beginPath();
                ctx.arc(sx, sy, nodeSize, 0, Math.PI * 2);
                ctx.fillStyle = (useSelectedColor && this.selectedIds.has(node.id)) ? '#f59e0b' : (node._fill || '#888');
                ctx.fill();
            }
        }
    }

    _renderLinkChunk() {
        const ctx = this._ctx;
        if (!ctx) return;
        const q = this._chunkQueue;
        const pos = this._chunkPos;
        if (pos >= q.length) {
            this._chunkQueue = null;
            const cb = this._chunkOnComplete;
            this._chunkOnComplete = null;
            if (cb) cb();
            return;
        }

        const dpr = this._dpr;
        const tc = this._tc;
        const chunkEnd = Math.min(pos + 60000, q.length); // 30k links per chunk
        const regColor = 'rgba(107,114,128,0.15)';
        const mainColor = 'rgba(255,255,255,0.25)';
        const regWidth = Math.max(0.5, 0.5 * Math.min(1, this._zoomLevel));
        const mainWidth = Math.max(0.5, 1.5 * Math.min(1, this._zoomLevel));

        // First pass: build paths for regular and main chain
        let regStarted = false, mainStarted = false;
        ctx.save();
        ctx.scale(dpr, dpr);

        for (let i = pos; i < chunkEnd; i += 5) {
            const isMain = q[i];
            const sx = q[i + 1], sy = q[i + 2];
            const tx = q[i + 3], ty = q[i + 4];
            const ax = (sx - tc.cx) * tc.scale + tc.ox;
            const ay = -(sy - tc.cy) * tc.scale + tc.oy;
            const bx = (tx - tc.cx) * tc.scale + tc.ox;
            const by = -(ty - tc.cy) * tc.scale + tc.oy;

            if (isMain) {
                if (!mainStarted) { ctx.beginPath(); mainStarted = true; }
                ctx.moveTo(ax, ay);
                ctx.lineTo(bx, by);
            } else {
                if (!regStarted) { ctx.beginPath(); regStarted = true; }
                ctx.moveTo(ax, ay);
                ctx.lineTo(bx, by);
            }
        }

        if (regStarted) { ctx.strokeStyle = regColor; ctx.lineWidth = regWidth; ctx.stroke(); }
        if (mainStarted) { ctx.strokeStyle = mainColor; ctx.lineWidth = mainWidth; ctx.stroke(); }
        ctx.restore();

        this._chunkPos = chunkEnd;
        this._chunkTimer = setTimeout(() => this._renderLinkChunk(), 0);
    }

    _buildScene() {
        if (!this._ready) return;
        if (this.nodePoints) {
            this.scene.remove(this.nodePoints);
            this.nodePoints.geometry.dispose();
            this.nodePoints.material.dispose();
            this.nodePoints = null;
        }
        if (this.linkLines) {
            this.scene.remove(this.linkLines);
            this.linkLines.geometry.dispose();
            this.linkLines.material.dispose();
            this.linkLines = null;
        }
        if (this.mainChainLines) {
            this.scene.remove(this.mainChainLines);
            this.mainChainLines.geometry.dispose();
            this.mainChainLines.material.dispose();
            this.mainChainLines = null;
        }

        let nodeSize = 6;
        if (this.worldBounds && this.worldBounds.width && this.worldBounds.height) {
            const diag = Math.sqrt(
                this.worldBounds.width * this.worldBounds.width +
                this.worldBounds.height * this.worldBounds.height
            );
            nodeSize = Math.max(2, diag * 0.004);
        }

        this._buildLinks();
        this._buildNodes(nodeSize);
    }

    _buildLinks() {
        const regularPositions = [];
        const mainPositions = [];

        for (const link of this.linksData) {
            const sx = link.source.x, sy = link.source.y;
            const tx = link.target.x, ty = link.target.y;
            if (!isFinite(sx) || !isFinite(sy) || !isFinite(tx) || !isFinite(ty)) continue;
            if (link.isMainChain) {
                mainPositions.push(sx, sy, 0, tx, ty, 0);
            } else {
                regularPositions.push(sx, sy, 0, tx, ty, 0);
            }
        }

        if (regularPositions.length > 0) {
            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.Float32BufferAttribute(regularPositions, 3));
            const mat = new THREE.LineBasicMaterial({ color: 0x6b7280, transparent: true, opacity: 0.35 });
            this.linkLines = new THREE.LineSegments(geo, mat);
            this.linkLines.visible = this._showRegularLinks;
            this.scene.add(this.linkLines);
        }

        if (mainPositions.length > 0) {
            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.Float32BufferAttribute(mainPositions, 3));
            const mat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 });
            this.mainChainLines = new THREE.LineSegments(geo, mat);
            this.mainChainLines.visible = this._showMainChains;
            this.scene.add(this.mainChainLines);
        }
    }

    _buildNodes(nodeSize) {
        const positions = [];
        const colors = [];
        const tempColor = new THREE.Color();

        for (const node of this.nodesData) {
            if (!isFinite(node.x) || !isFinite(node.y)) continue;
            positions.push(node.x, node.y, 0);
            tempColor.set(node._fill || '#888888');
            colors.push(tempColor.r, tempColor.g, tempColor.b);
        }

        if (positions.length === 0) return;

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const mat = new THREE.PointsMaterial({
            size: nodeSize,
            vertexColors: true,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.9,
        });

        this.nodePoints = new THREE.Points(geo, mat);
        this.scene.add(this.nodePoints);
    }

    _fitToWorld(world) {
        if (!this._ready || !world || !world.width || !world.height) return;
        if (this._fallback) {
            this._panX = 0;
            this._panY = 0;
            this._zoomLevel = 1;
            if (this.nodesData.length) this._renderFallback();
            return;
        }
        const aspect = this.camera.aspect || 1;
        let viewW = world.width * 1.2;
        let viewH = world.height * 1.2;
        if (viewW / aspect > viewH) {
            viewH = viewW / aspect;
        } else {
            viewW = viewH * aspect;
        }

        this.camera.left = -viewW / 2;
        this.camera.right = viewW / 2;
        this.camera.top = viewH / 2;
        this.camera.bottom = -viewH / 2;
        this.camera.zoom = 1;
        this._referenceZoom = 1;
        this.camera.updateProjectionMatrix();

        const cx = world.x + world.width / 2;
        const cy = world.y + world.height / 2;
        this.controls.target.set(cx, cy, 0);
        this.controls.update();
    }

    resize() {
        if (!this._ready) return;
        const w = Math.max(this.container.clientWidth || 800, 1);
        const h = Math.max(this.container.clientHeight || 600, 1);
        const dpr = Math.min(window.devicePixelRatio, 2);

        if (this._fallback) {
            this._w = w;
            this._h = h;
            this._dpr = dpr;
            this._fallbackCanvas.width = Math.round(w * dpr);
            this._fallbackCanvas.height = Math.round(h * dpr);
            this._fallbackCanvas.style.width = w + 'px';
            this._fallbackCanvas.style.height = h + 'px';
            // Draw nodes immediately (synchronous, fast), then link chunks in background
            if (this.nodesData.length) this._renderFallback();
            return;
        }

        this.renderer.setSize(w, h);

        const aspect = w / h;
        const viewH = (this.camera.top - this.camera.bottom) / Math.max(this.camera.zoom, 0.001);
        const viewW = viewH * aspect;
        this.camera.left = -viewW / 2;
        this.camera.right = viewW / 2;
        this.camera.top = viewH / 2;
        this.camera.bottom = -viewH / 2;
        this.camera.aspect = aspect;
        this.camera.updateProjectionMatrix();

        // Re-render immediately (no animation loop for static graph)
        this.renderer.render(this.scene, this.camera);
    }

    hitTest(event) {
        if (!this._ready || !this.nodesData.length) return null;
        if (this._fallback) return this._hitTestFallback(event);

        if (!this.nodePoints) return null;
        const rect = this.renderer.domElement.getBoundingClientRect();
        const cx = (event.clientX !== undefined ? event.clientX : event.x) - rect.left;
        const cy = (event.clientY !== undefined ? event.clientY : event.y) - rect.top;
        this._pointer.x = (cx / rect.width) * 2 - 1;
        this._pointer.y = -(cy / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this._pointer, this.camera);
        const intersects = this.raycaster.intersectObject(this.nodePoints);
        if (intersects.length > 0) {
            const idx = intersects[0].index;
            return this.nodesData[idx] || null;
        }
        return null;
    }

    _hitTestFallback(event) {
        if (!this.worldBounds || !this.worldBounds.width) return null;
        const rect = this._fallbackCanvas.getBoundingClientRect();
        const px = (event.clientX || event.x) - rect.left;
        const py = (event.clientY || event.y) - rect.top;

        const world = this._screenToFallbackWorld(px, py);
        const diag = Math.sqrt(
            this.worldBounds.width * this.worldBounds.width +
            this.worldBounds.height * this.worldBounds.height
        );
        const baseScale = Math.min(this._w, this._h) / Math.max(diag, 1) * 0.9;
        const s = baseScale * this._zoomLevel;
        const hitRadius = 10 / Math.max(s, 0.001);

        let closest = null;
        let closestDist = hitRadius;

        for (const node of this.nodesData) {
            if (!isFinite(node.x) || !isFinite(node.y)) continue;
            const dx = node.x - world.x;
            const dy = node.y - world.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < closestDist) {
                closestDist = dist;
                closest = node;
            }
        }

        return closest;
    }

    hitTestLink(event) {
        if (!this._ready) return null;
        if (this._fallback) return this._hitTestLinkFallback(event);

        const targetLines = this.mainChainLines && this.mainChainLines.visible ? this.mainChainLines : this.linkLines;
        if (!targetLines) return null;

        const rect = this.renderer.domElement.getBoundingClientRect();
        const cx = (event.clientX !== undefined ? event.clientX : event.x) - rect.left;
        const cy = (event.clientY !== undefined ? event.clientY : event.y) - rect.top;
        this._pointer.x = (cx / rect.width) * 2 - 1;
        this._pointer.y = -(cy / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this._pointer, this.camera);
        const intersects = this.raycaster.intersectObject(targetLines);
        if (intersects.length > 0) {
            const faceIdx = intersects[0].faceIndex;
            if (faceIdx !== undefined) {
                const linkIdx = Math.floor(faceIdx / 2);
                return this.linksData[linkIdx] || null;
            }
        }
        return null;
    }

    _hitTestLinkFallback(event) {
        if (!this.linksData.length || !this.worldBounds || !this.worldBounds.width) return null;
        const rect = this._fallbackCanvas.getBoundingClientRect();
        const px = (event.clientX || event.x) - rect.left;
        const py = (event.clientY || event.y) - rect.top;

        const world = this._screenToFallbackWorld(px, py);
        const diag = Math.sqrt(
            this.worldBounds.width * this.worldBounds.width +
            this.worldBounds.height * this.worldBounds.height
        );
        const baseScale = Math.min(this._w, this._h) / Math.max(diag, 1) * 0.9;
        const s = baseScale * this._zoomLevel;
        const hitDist = 5 / Math.max(s, 0.001);

        const distToSegment = (ax, ay, bx, by, px, py) => {
            const dx = bx - ax, dy = by - ay;
            const lenSq = dx * dx + dy * dy;
            if (lenSq === 0) return Math.sqrt((px - ax) ** 2 + (py - ay) ** 2);
            let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
            t = Math.max(0, Math.min(1, t));
            return Math.sqrt((px - (ax + t * dx)) ** 2 + (py - (ay + t * dy)) ** 2);
        };

        for (const link of this.linksData) {
            const sx = link.source.x, sy = link.source.y;
            const tx = link.target.x, ty = link.target.y;
            if (!isFinite(sx) || !isFinite(sy) || !isFinite(tx) || !isFinite(ty)) continue;
            if (distToSegment(sx, sy, tx, ty, world.x, world.y) < hitDist) return link;
        }
        return null;
    }

    worldToScreen(worldX, worldY) {
        if (!this._ready) return { x: 0, y: 0 };
        if (this._fallback) {
            this._updateTransformCache();
            return {
                x: (worldX - this._tc.cx) * this._tc.scale + this._tc.ox,
                y: -(worldY - this._tc.cy) * this._tc.scale + this._tc.oy,
            };
        }
        const vec = this._tempVec3.set(worldX, worldY, 0);
        vec.project(this.camera);
        const rect = this.renderer.domElement.getBoundingClientRect();
        return {
            x: (vec.x * 0.5 + 0.5) * rect.width,
            y: (-vec.y * 0.5 + 0.5) * rect.height,
        };
    }

    setDetailLevel(detailLevel) {
        this.detailLevel = { ...this.detailLevel, ...detailLevel };
    }

    setTransform() {}

    setLinkVisibility(showMain, showRegular) {
        this._showMainChains = showMain;
        this._showRegularLinks = showRegular;
        if (this._fallback) {
            if (this.nodesData.length) this._renderFallback();
            return;
        }
        if (this.mainChainLines) this.mainChainLines.visible = showMain;
        if (this.linkLines) this.linkLines.visible = showRegular;
    }

    setHighlightedChain(chainId) {
        this.highlightedChainId = chainId;
    }

    updateSelection(selectedIds) {
        this.selectedIds = new Set(selectedIds);
    }

    get didDrag() {
        return this._didDrag || false;
    }

    requestRender() {}

    destroy() {
        this._ready = false;
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }
        if (this._fallback) {
            if (this._handleWheel) this._fallbackCanvas.removeEventListener('wheel', this._handleWheel);
            if (this._handleMouseDown) this._fallbackCanvas.removeEventListener('mousedown', this._handleMouseDown);
            if (this._handleMouseMove) window.removeEventListener('mousemove', this._handleMouseMove);
            if (this._handleMouseUp) window.removeEventListener('mouseup', this._handleMouseUp);
            if (this._fallbackCanvas && this._fallbackCanvas.parentNode) {
                this._fallbackCanvas.remove();
            }
            this._ctx = null;
            this._fallbackCanvas = null;
            return;
        }
        if (this.controls) this.controls.dispose();
        if (this.renderer) {
            this.renderer.dispose();
            if (this.renderer.domElement.parentNode) {
                this.renderer.domElement.remove();
            }
        }
    }
}
