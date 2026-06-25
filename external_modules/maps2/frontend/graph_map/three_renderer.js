globalThis.ThreeGraphRenderer = class {
    constructor({ container }) {
        this.container = container;
        this._ready = false;

        this.nodesData = [];
        this.linksData = [];
        this.selectedIds = new Set();
        this.worldBounds = null;
        this._showMainLinks = true;
        this._showRegularLinks = true;
        this._isLoopActive = false;
        this._animationFrameId = null;

        const containerWidth = Math.max(container.clientWidth || 800, 1);
        const containerHeight = Math.max(container.clientHeight || 600, 1);
        const dpr = Math.min(window.devicePixelRatio, 2);

        this._resizeObserver = new ResizeObserver(() => this.resize());
        this._resizeObserver.observe(container);

        const canvas = document.createElement("canvas");
        canvas.width = Math.round(containerWidth * dpr);
        canvas.height = Math.round(containerHeight * dpr);
        canvas.style.cssText = "position:absolute;top:0;left:0;display:block;width:" + containerWidth + "px;height:" + containerHeight + "px;pointer-events:auto;touch-action:none";
        canvas.classList.add("graph-map-canvas");
        container.insertBefore(canvas, container.firstChild);

        const testCanvas = document.createElement("canvas");
        testCanvas.width = testCanvas.height = 2;
        const webglOk = !!(testCanvas.getContext("webgl2", { alpha: true, antialias: true }) || testCanvas.getContext("webgl", { alpha: true, antialias: true }));

        if (!webglOk) {
            const errorElement = document.createElement("div");
            errorElement.className = "webgl-error";
            errorElement.textContent = "WebGL is required for graph rendering.";
            container.appendChild(errorElement);
            this._errorElement = errorElement;
            return;
        }

        try {
            this.renderer = new globalThis.THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
        } catch {
            globalThis.showError("WebGL initialization failed.");
            this._errorElement = true;
            return;
        }

        this.renderer.setPixelRatio(dpr);
        this.renderer.setClearColor(0x000000, 0);
        this.renderer.setSize(containerWidth, containerHeight);
        container.insertBefore(this.renderer.domElement, container.firstChild);

        this._pixelRatio = dpr;
        this.labelsVisible = false;
        this._highlightedChainId = null;
        this._labelCanvas = document.createElement("canvas");
        this._labelCanvas.style.cssText = "position:absolute;top:0;left:0;display:block;width:" + containerWidth + "px;height:" + containerHeight + "px;pointer-events:none";
        this._labelCtx = this._labelCanvas.getContext("2d");
        this._labelCanvas.width = Math.round(containerWidth * dpr);
        this._labelCanvas.height = Math.round(containerHeight * dpr);
        container.appendChild(this._labelCanvas);
        this._projectVec = new globalThis.THREE.Vector3();

        this.scene = new globalThis.THREE.Scene();
        const viewSize = Math.max(containerWidth, containerHeight);
        this.camera = new globalThis.THREE.OrthographicCamera(-viewSize / 2, viewSize / 2, viewSize / 2, -viewSize / 2, -10000, 10000);
        this.camera.position.z = 1000;
        this._referenceZoom = 1;

        this.controls = new globalThis.THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableRotate = false;
        this.controls.enableKeys = false;
        this._rotationEnabled = false;
        this.controls.screenSpacePanning = true;
        this.controls.zoomSpeed = 1.5;
        this.controls.mouseButtons = { LEFT: globalThis.THREE.MOUSE.PAN, MIDDLE: globalThis.THREE.MOUSE.DOLLY, RIGHT: globalThis.THREE.MOUSE.PAN };
        this.controls.touches = { ONE: globalThis.THREE.TOUCH.PAN, TWO: globalThis.THREE.TOUCH.DOLLY_PAN };
        this.controls.update();

        this.gridGroup = new globalThis.THREE.Group();
        this.scene.add(this.gridGroup);
        this.nodePoints = null;
        this.regLinks = null;
        this.secondaryLinks = null;
        this.primaryLinks = null;
        this.borderLines = null;
        this._sizeAttribute = null;
        this._nodeRadii = null;

        this._squareTexture = (() => {
            const textureCanvas = document.createElement("canvas");
            textureCanvas.width = textureCanvas.height = 64;
            const context = textureCanvas.getContext("2d");
            context.fillStyle = "#fff";
            context.fillRect(6, 6, 52, 52);
            return new globalThis.THREE.CanvasTexture(textureCanvas);
        })();

        this.raycaster = new globalThis.THREE.Raycaster();
        this.raycaster.params.Points.threshold = 500;
        this._pointer = new globalThis.THREE.Vector2();
        this._tempVector = new globalThis.THREE.Vector3();
        this._onControlsChange = () => {
            if (!this._ready) {
                return;
            }
            this._updateNodeSizes();
            this._updateGrids();
            this._drawLabels();
            if (!this._isLoopActive) {
                this.renderer.render(this.scene, this.camera);
            }
        };
        this.controls.addEventListener("change", this._onControlsChange);

        this._updateRotationMode = (enabled) => {
            this.controls.enableRotate = enabled;
            if (enabled) {
                this.controls.mouseButtons = { LEFT: globalThis.THREE.MOUSE.ROTATE, MIDDLE: globalThis.THREE.MOUSE.DOLLY, RIGHT: globalThis.THREE.MOUSE.PAN };
                this.controls.touches = { ONE: globalThis.THREE.TOUCH.PAN, TWO: globalThis.THREE.TOUCH.DOLLY_ROTATE };
            } else {
                this.controls.mouseButtons = { LEFT: globalThis.THREE.MOUSE.PAN, MIDDLE: globalThis.THREE.MOUSE.DOLLY, RIGHT: globalThis.THREE.MOUSE.PAN };
                this.controls.touches = { ONE: globalThis.THREE.TOUCH.PAN, TWO: globalThis.THREE.TOUCH.DOLLY_PAN };
            }
        };
        document.addEventListener("keydown", (event) => {
            if (event.key === "r" && event.target.tagName !== "INPUT") {
                this._rotationEnabled = !this._rotationEnabled;
                this._updateRotationMode(this._rotationEnabled);
                if (!this._rotationEnabled) {
                    const target = this.controls.target;
                    this.camera.position.set(target.x, target.y, 1000);
                    this.controls.update();
                    this._updateNodeSizes();
                    this._updateGrids();
                    this.renderer.render(this.scene, this.camera);
                }
            }
        });

        this._ready = true;
    }

    get canvas() {
        return this._errorElement ? null : this.renderer.domElement;
    }

    render({ nodes, links, selectedIds, world }) {
        if (!this._ready || this._errorElement) {
            return;
        }
        this.resize();
        this.nodesData = nodes || [];
        this.linksData = links || [];
        this.selectedIds = new Set(selectedIds || []);
        this.worldBounds = world ? { x: world.x, y: world.y, width: world.width, height: world.height } : null;
        this._rebuildScene();
        if (this.worldBounds) {
            this._fitToWorld(this.worldBounds);
        }
        this._drawLabels();
    }

    renderOne() {
        if (!this._ready || this._errorElement) {
            return;
        }
        this._updateNodeSizes();
        this._updateGrids();
        this._updateLinkOpacity();
        this._drawLabels();
        this.renderer.render(this.scene, this.camera);
    }

    _drawLabels() {
        const ctx = this._labelCtx;
        const dpr = this._pixelRatio;
        const w = this._labelCanvas.width;
        const h = this._labelCanvas.height;
        ctx.clearRect(0, 0, w, h);
        if (!this.labelsVisible || !this.nodesData.length) {
            return;
        }
        const rect = this.renderer.domElement.getBoundingClientRect();
        const cap = globalThis.RENDER.label.cap || 500;
        const maxLabels = Math.min(this.nodesData.length, cap);
        const v = this._projectVec;
        ctx.font = "500 " + (11 * dpr) + "px Inter, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        for (let i = 0; i < maxLabels; i++) {
            const node = this.nodesData[i];
            if (!isFinite(node.x) || !isFinite(node.y)) {
                continue;
            }
            v.set(node.x, node.y, 0);
            v.project(this.camera);
            const sx = (v.x * 0.5 + 0.5) * rect.width;
            const sy = (-v.y * 0.5 + 0.5) * rect.height;
            if (sx < -100 || sx > rect.width + 100 || sy < -100 || sy > rect.height + 100) {
                continue;
            }
            const label = node.id.split("/")
                .pop();
            ctx.fillStyle = globalThis.RENDER.label.color;
            ctx.fillText(label, sx * dpr, (sy - 5) * dpr);
        }
    }

    startLoop(tickFn) {
        if (this._isLoopActive) {
            return;
        }
        this._isLoopActive = true;
        const loop = () => {
            if (!this._isLoopActive) {
                return;
            }
            let alive;
            try {
                alive = tickFn();
            } catch (error) {
                globalThis.showError("Tick error: " + error.message);
                alive = true;
            }
            if (alive) {
                this._updatePositions();
                this._updateNodeSizes();
                this._updateGrids();
                this._updateLinkOpacity();
                this._drawLabels();
                this.renderer.render(this.scene, this.camera);
                this._animationFrameId = requestAnimationFrame(loop);
            } else {
                this._isLoopActive = false;
                this._animationFrameId = null;
                this._updatePositions();
                this._updateNodeSizes();
                this._updateGrids();
                this._updateLinkOpacity();
                this._drawLabels();
                this.renderer.render(this.scene, this.camera);
                if (this.onSimulationEnd) {
                    this.onSimulationEnd();
                }
            }
        };
        this._animationFrameId = requestAnimationFrame(loop);
    }

    stopLoop() {
        this._isLoopActive = false;
        if (this._animationFrameId) {
            cancelAnimationFrame(this._animationFrameId);
            this._animationFrameId = null;
        }
    }

    _updatePositions() {
        if (!this.nodePoints) {
            return;
        }
        const nodeAttribute = this.nodePoints.geometry.attributes.position;
        const positions = nodeAttribute.array;
        for (let index = 0; index < this.nodesData.length; index++) {
            const node = this.nodesData[index];
            positions[index * 3] = node.x;
            positions[index * 3 + 1] = node.y;
        }
        nodeAttribute.needsUpdate = true;

        const isDirect = link => link.source._chainNext === link.target.id || link.source._chainPrev === link.target.id || link.target._chainNext === link.source.id || link.target._chainPrev === link.source.id;
        if (this.regLinks) {
            const regularAttribute = this.regLinks.geometry.attributes.position;
            const regularPositions = regularAttribute.array;
            let regularIndex = 0;
            for (const link of this.linksData) {
                if (isDirect(link) || link.isMainChain) {
                    continue;
                }
                regularPositions[regularIndex] = link.source.x;
                regularPositions[regularIndex + 1] = link.source.y;
                regularPositions[regularIndex + 3] = link.target.x;
                regularPositions[regularIndex + 4] = link.target.y;
                regularIndex += 6;
            }
            regularAttribute.needsUpdate = true;
        }
        if (this.secondaryLinks) {
            const secondaryAttribute = this.secondaryLinks.geometry.attributes.position;
            const secondaryPositions = secondaryAttribute.array;
            let secondaryIndex = 0;
            for (const link of this.linksData) {
                if (isDirect(link) || !link.isMainChain) {
                    continue;
                }
                secondaryPositions[secondaryIndex] = link.source.x;
                secondaryPositions[secondaryIndex + 1] = link.source.y;
                secondaryPositions[secondaryIndex + 3] = link.target.x;
                secondaryPositions[secondaryIndex + 4] = link.target.y;
                secondaryIndex += 6;
            }
            secondaryAttribute.needsUpdate = true;
        }
        if (this.primaryLinks) {
            const primaryAttribute = this.primaryLinks.geometry.attributes.position;
            const primaryPositions = primaryAttribute.array;
            let primaryIndex = 0;
            for (const link of this.linksData) {
                if (!isDirect(link)) {
                    continue;
                }
                primaryPositions[primaryIndex] = link.source.x;
                primaryPositions[primaryIndex + 1] = link.source.y;
                primaryPositions[primaryIndex + 3] = link.target.x;
                primaryPositions[primaryIndex + 4] = link.target.y;
                primaryIndex += 6;
            }
            primaryAttribute.needsUpdate = true;
        }
    }

    _updateLinkOpacity() {
        const factor = Math.min(1, this.camera.zoom);
        const dim = this._highlightedChainId ? 0.25 : 1;
        if (this.primaryLinks) {
            this.primaryLinks.material.opacity = 0.8 * factor;
        }
        if (this.secondaryLinks) {
            this.secondaryLinks.material.opacity = 0.3 * factor * dim;
        }
        if (this.regLinks) {
            this.regLinks.material.opacity = 0.1 * factor * dim;
        }
    }

    _rebuildScene() {
        this._clear();
        if (!this.nodesData.length) {
            return;
        }
        this._buildGrids();
        this._buildBorder();
        this._buildLinks();
        this._buildNodes();
    }

    _clear() {
        const disposeObject = (object) => {
            if (!object) {
                return;
            }
            this.scene.remove(object);
            if (object.geometry) {
                object.geometry.dispose();
            }
            if (object.material) {
                object.material.dispose();
            }
        };
        for (const child of [...this.gridGroup.children]) {
            this.gridGroup.remove(child);
            if (child.geometry) {
                child.geometry.dispose();
            }
            if (child.material) {
                child.material.dispose();
            }
        }
        disposeObject(this.nodePoints);
        disposeObject(this.regLinks);
        disposeObject(this.secondaryLinks);
        disposeObject(this.primaryLinks);
        disposeObject(this.borderLines);
        this.nodePoints = null;
        this.regLinks = null;
        this.secondaryLinks = null;
        this.primaryLinks = null;
        this.borderLines = null;
        this._gridLevels = [];
        this._sizeAttribute = null;
        this._nodeRadii = null;
    }

    _buildGrids() {
        if (!this.worldBounds) {
            return;
        }
        const left = this.worldBounds.x;
        const bottom = this.worldBounds.y;
        const right = left + this.worldBounds.width;
        const top = bottom + this.worldBounds.height;
        this._gridLevels = [];

        for (const step of globalThis.RENDER.grid.levels) {
            if (step > Math.max(this.worldBounds.width, this.worldBounds.height)) {
                break;
            }
            const positions = [];
            for (let coordX = Math.ceil(left / step) * step; coordX <= right; coordX += step) {
                positions.push(coordX, bottom, 0, coordX, top, 0);
            }
            for (let coordY = Math.ceil(bottom / step) * step; coordY <= top; coordY += step) {
                positions.push(left, coordY, 0, right, coordY, 0);
            }
            if (!positions.length) {
                continue;
            }
            const geometry = new globalThis.THREE.BufferGeometry();
            geometry.setAttribute("position", new globalThis.THREE.Float32BufferAttribute(positions, 3));
            const material = new globalThis.THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.08, depthTest: false });
            const lineSegments = new globalThis.THREE.LineSegments(geometry, material);
            lineSegments.userData.step = step;
            this.gridGroup.add(lineSegments);
            this._gridLevels.push(lineSegments);
        }
    }

    _updateGrids() {
        if (!this._gridLevels || !this._gridLevels.length) {
            return;
        }
        const zoom = this.camera.zoom / this._referenceZoom;
        const viewHeight = (this.camera.top - this.camera.bottom) / zoom;
        const screenPixelsPerUnit = this.container.clientHeight / viewHeight;

        let bestLevelIndex = 0;
        for (let index = 0; index < this._gridLevels.length; index++) {
            const stepScreenSize = this._gridLevels[index].userData.step * screenPixelsPerUnit;
            if (stepScreenSize > 5) {
                bestLevelIndex = index;
                break;
            }
            bestLevelIndex = index;
        }

        for (let index = 0; index < this._gridLevels.length; index++) {
            const step = this._gridLevels[index].userData.step;
            const stepScreenSize = step * screenPixelsPerUnit;
            let opacity = 0;
            const levelDistance = Math.abs(index - bestLevelIndex);
            if (levelDistance === 0) {
                opacity = 0.12;
            } else if (levelDistance === 1) {
                const neighborStepSize = this._gridLevels[bestLevelIndex].userData.step;
                const ratio = step / neighborStepSize;
                if (ratio > 1) {
                    opacity = Math.max(0, (15 - stepScreenSize) / 15) * 0.12;
                } else {
                    opacity = Math.min(1, (stepScreenSize - 2) / 8) * 0.06;
                }
            }
            this._gridLevels[index].material.opacity = Math.max(0.0, opacity);
        }
    }

    _buildBorder() {
        if (!this.worldBounds) {
            return;
        }
        const left = this.worldBounds.x;
        const bottom = this.worldBounds.y;
        const right = left + this.worldBounds.width;
        const top = bottom + this.worldBounds.height;
        const positions = [
            left, bottom, 0, right, bottom, 0,
            right, bottom, 0, right, top, 0,
            right, top, 0, left, top, 0,
            left, top, 0, left, bottom, 0,
        ];
        const geometry = new globalThis.THREE.BufferGeometry();
        geometry.setAttribute("position", new globalThis.THREE.Float32BufferAttribute(positions, 3));
        const material = new globalThis.THREE.LineBasicMaterial({
            color: new globalThis.THREE.Color(globalThis.RENDER.border.strokeColor),
            transparent: true,
            opacity: 0.5,
            depthTest: false,
        });
        this.borderLines = new globalThis.THREE.LineSegments(geometry, material);
        this.scene.add(this.borderLines);
    }

    _buildLinks() {
        const regularPositions = [];
        const secondaryPositions = [];
        const primaryPositions = [];
        const regularMap = [];
        const secondaryMap = [];
        const primaryMap = [];
        let linkIdx = 0;
        for (const link of this.linksData) {
            const sourceX = link.source.x;
            const sourceY = link.source.y;
            const targetX = link.target.x;
            const targetY = link.target.y;
            if (!isFinite(sourceX) || !isFinite(sourceY) || !isFinite(targetX) || !isFinite(targetY)) {
                linkIdx++;
                continue;
            }
            const isDirect = link.source._chainNext === link.target.id || link.source._chainPrev === link.target.id || link.target._chainNext === link.source.id || link.target._chainPrev === link.source.id;
            if (isDirect) {
                primaryPositions.push(sourceX, sourceY, 0, targetX, targetY, 0);
                primaryMap.push(linkIdx);
            } else if (link.isMainChain) {
                secondaryPositions.push(sourceX, sourceY, 0, targetX, targetY, 0);
                secondaryMap.push(linkIdx);
            } else {
                regularPositions.push(sourceX, sourceY, 0, targetX, targetY, 0);
                regularMap.push(linkIdx);
            }
            linkIdx++;
        }
        if (regularPositions.length) {
            const regularAttribute = new globalThis.THREE.Float32BufferAttribute(regularPositions, 3);
            regularAttribute.usage = globalThis.THREE.DynamicDrawUsage;
            const geometry = new globalThis.THREE.BufferGeometry();
            geometry.setAttribute("position", regularAttribute);
            const material = new globalThis.THREE.LineBasicMaterial({
                color: new globalThis.THREE.Color(globalThis.RENDER.link.regularColor),
                transparent: true,
                opacity: 0.1,
                depthTest: false,
            });
            this.regLinks = new globalThis.THREE.LineSegments(geometry, material);
            this.regLinks.userData.linkMap = regularMap;
            this.regLinks.visible = this._showRegularLinks;
            this.scene.add(this.regLinks);
        }
        if (secondaryPositions.length) {
            const secondaryAttribute = new globalThis.THREE.Float32BufferAttribute(secondaryPositions, 3);
            secondaryAttribute.usage = globalThis.THREE.DynamicDrawUsage;
            const geometry = new globalThis.THREE.BufferGeometry();
            geometry.setAttribute("position", secondaryAttribute);
            const material = new globalThis.THREE.LineBasicMaterial({
                color: new globalThis.THREE.Color(globalThis.RENDER.link.mainChainColor),
                transparent: true,
                opacity: 0.3,
                depthTest: false,
            });
            this.secondaryLinks = new globalThis.THREE.LineSegments(geometry, material);
            this.secondaryLinks.userData.linkMap = secondaryMap;
            this.secondaryLinks.visible = this._showMainLinks;
            this.scene.add(this.secondaryLinks);
        }
        if (primaryPositions.length) {
            const primaryAttribute = new globalThis.THREE.Float32BufferAttribute(primaryPositions, 3);
            primaryAttribute.usage = globalThis.THREE.DynamicDrawUsage;
            const geometry = new globalThis.THREE.BufferGeometry();
            geometry.setAttribute("position", primaryAttribute);
            const material = new globalThis.THREE.LineBasicMaterial({
                color: new globalThis.THREE.Color(globalThis.RENDER.link.mainChainColor),
                transparent: true,
                opacity: 0.8,
                depthTest: false,
            });
            this.primaryLinks = new globalThis.THREE.LineSegments(geometry, material);
            this.primaryLinks.userData.linkMap = primaryMap;
            this.primaryLinks.visible = this._showMainLinks;
            this.scene.add(this.primaryLinks);
        }
    }

    _buildNodes() {
        const positions = [];
        const colors = [];
        const sizes = [];
        const color = new globalThis.THREE.Color();
        const hasSelection = this.selectedIds.size > 0;
        for (const node of this.nodesData) {
            if (!isFinite(node.x) || !isFinite(node.y)) {
                continue;
            }
            positions.push(node.x, node.y, 0);
            let nodeColor;
            if (this._highlightedChainId && node._allChains && node._allChains.includes(this._highlightedChainId)) {
                nodeColor = globalThis.RENDER.node.highlightColor;
            } else if (hasSelection && this.selectedIds.has(node.id)) {
                nodeColor = globalThis.RENDER.node.selectedColor;
            } else {
                nodeColor = node._fill || "#888";
            }
            color.set(nodeColor);
            colors.push(color.r, color.g, color.b);
            sizes.push(node._radius || 1);
        }
        if (!positions.length) {
            return;
        }
        const geometry = new globalThis.THREE.BufferGeometry();
        const positionAttribute = new globalThis.THREE.Float32BufferAttribute(positions, 3);
        positionAttribute.usage = globalThis.THREE.DynamicDrawUsage;
        geometry.setAttribute("position", positionAttribute);
        geometry.setAttribute("color", new globalThis.THREE.Float32BufferAttribute(colors, 3));
        const sizeAttribute = new globalThis.THREE.Float32BufferAttribute(sizes, 1);
        sizeAttribute.usage = globalThis.THREE.DynamicDrawUsage;
        geometry.setAttribute("size", sizeAttribute);
        this._sizeAttribute = geometry.attributes.size;
        this._nodeRadii = sizes.slice();
        const material = new globalThis.THREE.PointsMaterial({
            size: 1,
            vertexColors: true,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.9,
            map: this._squareTexture,
            blending: globalThis.THREE.NormalBlending,
            depthWrite: false,
            depthTest: false,
        });
        material.onBeforeCompile = (shader) => {
            shader.vertexShader = shader.vertexShader.replace("uniform float size;", "attribute float size;");
        };
        this.nodePoints = new globalThis.THREE.Points(geometry, material);
        this.nodePoints.renderOrder = 1;
        this.scene.add(this.nodePoints);
    }

    _updateNodeSizes() {
        if (!this._sizeAttribute || !this._nodeRadii) {
            return;
        }
        const viewHeight = (this.camera.top - this.camera.bottom) / this.camera.zoom;
        const screenPixelsPerUnit = this.container.clientHeight / viewHeight;
        const sizeArray = this._sizeAttribute.array;
        for (let index = 0; index < sizeArray.length; index++) {
            sizeArray[index] = 2 * this._nodeRadii[index] * screenPixelsPerUnit;
        }
        this._sizeAttribute.needsUpdate = true;
    }

    _fitToWorld(world) {
        if (!this._ready || !world || !world.width || !world.height) {
            return;
        }
        const aspect = this.camera.aspect || 1;
        let viewWidth = world.width * 1.0;
        let viewHeight = world.height * 1.0;
        if (viewWidth / aspect > viewHeight) {
            viewHeight = viewWidth / aspect;
        } else {
            viewWidth = viewHeight * aspect;
        }
        this.camera.left = -viewWidth / 2;
        this.camera.right = viewWidth / 2;
        this.camera.top = viewHeight / 2;
        this.camera.bottom = -viewHeight / 2;
        const maxDimension = Math.max(world.width, world.height);
        this.camera.near = -maxDimension * 3;
        this.camera.far = maxDimension * 3;
        this.camera.zoom = 1;
        this._referenceZoom = 1;
        this.camera.minZoom = 0.1;
        this.camera.maxZoom = 50;
        this.camera.updateProjectionMatrix();
        const centerX = world.x + world.width / 2;
        const centerY = world.y + world.height / 2;
        this.camera.position.set(centerX, centerY, 1000);
        this.controls.target.set(centerX, centerY, 0);
        this.controls.update();
        this._updateNodeSizes();
    }

    resize() {
        if (!this._ready || this._errorElement) {
            return;
        }
        const width = Math.max(this.container.clientWidth || 800, 1);
        const height = Math.max(this.container.clientHeight || 600, 1);
        this.renderer.setSize(width, height);
        const dpr = this._pixelRatio;
        this._labelCanvas.width = Math.round(width * dpr);
        this._labelCanvas.height = Math.round(height * dpr);
        this._labelCanvas.style.width = width + "px";
        this._labelCanvas.style.height = height + "px";
        const aspect = width / height;
        const viewHeight = (this.camera.top - this.camera.bottom) / Math.max(this.camera.zoom, 0.001);
        const viewWidth = viewHeight * aspect;
        this.camera.left = -viewWidth / 2;
        this.camera.right = viewWidth / 2;
        this.camera.top = viewHeight / 2;
        this.camera.bottom = -viewHeight / 2;
        this.camera.aspect = aspect;
        this.camera.updateProjectionMatrix();
        this._updateNodeSizes();
        if (this.nodesData.length) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    hitTest(event) {
        if (!this._ready || this._errorElement || !this.nodePoints) {
            return null;
        }
        const rect = this.renderer.domElement.getBoundingClientRect();
        const clientX = event.clientX - rect.left;
        const clientY = event.clientY - rect.top;
        this._pointer.x = (clientX / rect.width) * 2 - 1;
        this._pointer.y = -(clientY / rect.height) * 2 + 1;
        const viewHeight = (this.camera.top - this.camera.bottom) / this.camera.zoom;
        const pixelsPerUnit = rect.height / viewHeight;
        this.raycaster.params.Points.threshold = Math.max(25 / pixelsPerUnit, 10);
        this.raycaster.setFromCamera(this._pointer, this.camera);
        const hits = this.raycaster.intersectObject(this.nodePoints);
        return hits.length ? this.nodesData[hits[0].index] || null : null;
    }

    hitTestLink(event) {
        if (!this._ready || this._errorElement || !this.linksData.length) {
            return null;
        }
        const groups = [this.primaryLinks, this.secondaryLinks, this.regLinks].filter(g => g && g.visible);
        if (!groups.length) {
            return null;
        }
        const rect = this.renderer.domElement.getBoundingClientRect();
        const clientX = event.clientX - rect.left;
        const clientY = event.clientY - rect.top;
        this._pointer.x = (clientX / rect.width) * 2 - 1;
        this._pointer.y = -(clientY / rect.height) * 2 + 1;
        const viewHeight = (this.camera.top - this.camera.bottom) / this.camera.zoom;
        const pixelsPerUnit = rect.height / viewHeight;
        this.raycaster.params.Line.threshold = Math.max(15 / pixelsPerUnit, 5);
        this.raycaster.setFromCamera(this._pointer, this.camera);
        for (const target of groups) {
            const hits = this.raycaster.intersectObject(target);
            if (hits.length && hits[0].faceIndex !== undefined) {
                const localIndex = Math.floor(hits[0].faceIndex / 2);
                const dataIndex = target.userData.linkMap[localIndex];
                return this.linksData[dataIndex] || null;
            }
        }
        return null;
    }

    worldToScreen(worldX, worldY) {
        if (!this._ready || this._errorElement) {
            return { x: 0, y: 0 };
        }
        const vector = this._tempVector.set(worldX, worldY, 0);
        vector.project(this.camera);
        const rect = this.renderer.domElement.getBoundingClientRect();
        return { x: (vector.x * 0.5 + 0.5) * rect.width, y: (-vector.y * 0.5 + 0.5) * rect.height };
    }

    setLinkVisibility(showMainLinks, showSecondaryLinks, showRegularLinks) {
        this._showMainLinks = showMainLinks;
        this._showRegularLinks = showRegularLinks;
        if (this.primaryLinks) {
            this.primaryLinks.visible = showMainLinks;
        }
        if (this.secondaryLinks) {
            this.secondaryLinks.visible = showSecondaryLinks;
        }
        if (this.regLinks) {
            this.regLinks.visible = showRegularLinks;
        }
    }

    setHighlightedChain(chainId) {
        this._highlightedChainId = chainId || null;
        if (!this._ready || this._errorElement) {
            return;
        }
        this._rebuildScene();
        this._drawLabels();
        this.renderer.render(this.scene, this.camera);
    }

    updateSelection(ids) {
        this.selectedIds = new Set(ids);
    }

    destroy() {
        this.stopLoop();
        this._ready = false;
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }
        if (this.controls) {
            this.controls.removeEventListener("change", this._onControlsChange);
            this.controls.dispose();
        }
        if (this.renderer) {
            this.renderer.dispose();
            if (this.renderer.domElement.parentNode) {
                this.renderer.domElement.remove();
            }
        }
        this._clear();
        this.nodesData = [];
        this.linksData = [];
        if (this._labelCanvas && this._labelCanvas.parentNode) {
            this._labelCanvas.remove();
        }
        this._labelCanvas = null;
        this._labelCtx = null;
    }
};
