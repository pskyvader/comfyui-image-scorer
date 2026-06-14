/**
 * Interaction Logic (Zoom, Drag, Click) for ChainMapUI
 */

ChainMapUI.prototype._reheatSimulation = function() {
    if (!this.chainSim) return;
    if (this.chainSim._alpha < this.chainSim._alphaMin || !this.chainSim._running) {
        this.chainSim._alpha = Math.max(this.chainSim._alpha, 0.1);
        this.chainSim._running = true;
        this.chainSim._tick();
    }
};

ChainMapUI.prototype._getCanvasCoords = function(event) {
    const e = event.sourceEvent || event;
    const rect = this.renderer.canvas.getBoundingClientRect();
    let clientX, clientY;
    if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    } else if (e.changedTouches && e.changedTouches.length > 0) {
        clientX = e.changedTouches[0].clientX;
        clientY = e.changedTouches[0].clientY;
    } else {
        clientX = e.clientX;
        clientY = e.clientY;
    }
    return {
        x: clientX - rect.left,
        y: clientY - rect.top
    };
};

ChainMapUI.prototype.setupSVGAndRenderer = function() {
    if (this.svg) this.svg.remove();

    this.svg = d3.select("#graph-container")
        .append("svg")
        .attr("width", "100%")
        .attr("height", "100%")
        .style("position", "absolute")
        .style("top", 0)
        .style("left", 0)
        .style("z-index", 5)
        .style("pointer-events", "none");

    this.g = this.svg.append("g");

    this.zoom = d3.zoom()
        .scaleExtent(CAMERA.zoomExtent)
        .filter((event) => {
            if (event.type === "mousedown" || event.type === "touchstart") {
                const p = this._getCanvasCoords(event);
                const hit = this.renderer.hitTest({ x: p.x, y: p.y });
                if (hit) return false; 
            }
            return !event.ctrlKey && !event.button;
        })
        .on("zoom", (event) => {
            this.g.attr("transform", event.transform);
            this.updateHUD(event.transform);

            const k = event.transform.k;
            this.renderer.setDetailLevel({
                showLabels: this.labelsOverridden ? this.renderer.detailLevel.showLabels : k > CAMERA.labelThreshold,
                showArrows: k > CAMERA.arrowThreshold,
                showNodeBorders: k > CAMERA.borderThreshold,
                showLinks: this.linksOverridden ? this.renderer.detailLevel.showLinks : k > CAMERA.linkThreshold,
                labelCap: RENDER.label.labelCap,
                arrowCap: RENDER.arrow.arrowCap,
                zoom: k
            });
            this.renderer.setTransform(event.transform);
        });

    this.renderer = new ChainMapRenderer({
        container: this.container,
        svg: this.svg,
        g: this.g,
        interaction: CONTROLS
    });

    const canvas = d3.select(this.renderer.canvas);

    d3.select(this.container).call(this.zoom).on("dblclick.zoom", null);
    this._zoomEnabled = true;
    const self = this;
    const origZoomFilter = this.zoom.filter();
    this.zoom.filter(function(event) {
        if (!self._zoomEnabled) return false;
        return origZoomFilter.call(this, event);
    });

    canvas.on("mousemove", (event) => {
        const p = this._getCanvasCoords(event);
        const node = this.renderer.hitTest({ x: p.x, y: p.y });
        if (node) {
            this.showTooltip(event, node);
            canvas.style("cursor", "pointer");
        } else {
            const link = this.renderer.hitTestLink({ x: p.x, y: p.y });
            if (link) {
                this.hideTooltip();
                canvas.style("cursor", "pointer");
            } else {
                this.hideTooltip();
                canvas.style("cursor", "default");
            }
        }
    });

    canvas.on("touchstart", (event) => {
        if (event.touches.length !== 1) return;
        const touch = event.touches[0];
        const p = this._getCanvasCoords(event);
        const node = this.renderer.hitTest({ x: p.x, y: p.y });
        if (node) {
            this.tooltip.classList.remove("hidden");
            this.tooltip.style.left = `${touch.pageX + 10}px`;
            this.tooltip.style.top = `${touch.pageY + 10}px`;
            this.tooltip.innerHTML = `
                <div class="font-bold mb-1" style="cursor:pointer" onclick="window.chainMapUI.showNodeDetailsById('${node.id.replace(/'/g, "\\'")}')">${node.id.split('/').pop()}</div>
                <div>Score: ${node.score.toFixed(3)}</div>
                <div>Component: ${node._component_size}</div>
                <div>Comparisons: ${node.comparison_count}</div>
            `;
        } else {
            this.hideTooltip();
        }
    });

    this._dragged = false;

    canvas.on("click", (event) => {
        if (event.defaultPrevented || this._dragged) {
            console.warn("[click] suppressed — defaultPrevented:", event.defaultPrevented, "_dragged:", this._dragged);
            return;
        }
        if (this.tooltip && event.target && (this.tooltip === event.target || this.tooltip.contains(event.target))) return;
        if (this.nodeDetails && event.target && (this.nodeDetails === event.target || this.nodeDetails.contains(event.target))) return;
        const p = this._getCanvasCoords(event);
        const node = this.renderer.hitTest({ x: p.x, y: p.y });
        if (node) {
            console.log("[click] hit node:", node.id);
            this.showNodeDetails(node);
            return;
        }
        const link = this.renderer.hitTestLink({ x: p.x, y: p.y });
        if (link) {
            console.log("[click] hit link:", link.source.id, "->", link.target.id);
            this.showLinkDetails(link);
            return;
        }
        console.warn("[click] no hit — coords:", p.x, p.y);
        this.nodeDetails.classList.add("hidden");
    });

    const DRAG_THRESHOLD = 5; // CSS pixels — movement below this is a click/tap

    canvas.on("mousedown.drag", (event) => {
        const p = this._getCanvasCoords(event);
        const node = this.renderer.hitTest({ x: p.x, y: p.y });
        if (!node) return;
        this._reheatSimulation();
        this._dragNode = node;
        this._dragStartNode = { x: node.x, y: node.y };
        this._dragStartPointer = [p.x, p.y];
        node.vx = 0;
        node.vy = 0;
    });

    canvas.on("mousemove.drag", (event) => {
        if (!this._dragNode) return;
        const p = this._getCanvasCoords(event);
        const screenDx = p.x - this._dragStartPointer[0];
        const screenDy = p.y - this._dragStartPointer[1];
        if (Math.abs(screenDx) < DRAG_THRESHOLD && Math.abs(screenDy) < DRAG_THRESHOLD) return;
        this._dragged = true;
        event.preventDefault();
        const worldDx = screenDx / this.renderer.subRenderer.transform.k;
        const worldDy = screenDy / this.renderer.subRenderer.transform.k;
        this._dragNode.x = this._dragStartNode.x + worldDx;
        this._dragNode.y = this._dragStartNode.y + worldDy;
        this._dragNode.vx = 0;
        this._dragNode.vy = 0;
        this.renderer.requestRender();
    });

    canvas.on("mouseup.drag", () => {
        if (this._dragNode && this._dragged) {
            this._reheatSimulation();
        }
        this._dragNode = null;
        setTimeout(() => { this._dragged = false; }, 0);
    });

    canvas.on("touchstart.drag", (event) => {
        if (event.touches.length !== 1) return;
        const p = this._getCanvasCoords(event);
        const node = this.renderer.hitTest({ x: p.x, y: p.y });
        if (!node) return;
        this._reheatSimulation();
        event.stopPropagation();
        event.preventDefault();
        this._dragNode = node;
        this._dragStartNode = { x: node.x, y: node.y };
        this._dragStartPointer = [p.x, p.y];
        node.vx = 0;
        node.vy = 0;
        this._dragged = false;
    });

    canvas.on("touchmove.drag", (event) => {
        if (!this._dragNode || event.touches.length !== 1) return;
        const p = this._getCanvasCoords(event);
        const screenDx = p.x - this._dragStartPointer[0];
        const screenDy = p.y - this._dragStartPointer[1];
        if (Math.abs(screenDx) < DRAG_THRESHOLD && Math.abs(screenDy) < DRAG_THRESHOLD) return;
        this._dragged = true;
        const worldDx = screenDx / this.renderer.subRenderer.transform.k;
        const worldDy = screenDy / this.renderer.subRenderer.transform.k;
        this._dragNode.x = this._dragStartNode.x + worldDx;
        this._dragNode.y = this._dragStartNode.y + worldDy;
        this._dragNode.vx = 0;
        this._dragNode.vy = 0;
        this.renderer.requestRender();
    });

    canvas.on("touchend.drag", (event) => {
        const node = this._dragNode;
        if (node && !this._dragged) {
            this.showNodeDetails(node);
        }
        if (this._dragNode && this._dragged) {
            this._reheatSimulation();
        }
        this._dragNode = null;
        setTimeout(() => { this._dragged = false; }, 0);
        event.preventDefault();
    });
};

ChainMapUI.prototype.focusNode = function(id) {
    const node = this.chainSim.nodes.find(n => n.id === id);
    if (!node) return;

    d3.select(this.container).transition().duration(CAMERA.focusDuration).call(
        this.zoom.transform,
        d3.zoomIdentity
            .translate(this.width / 2, this.height / 2)
            .scale(CAMERA.focusScale)
            .translate(-node.x, -node.y)
    );
};

ChainMapUI.prototype.toggleSelection = function(id) {
    const idx = this.selectedNodes.indexOf(id);
    if (idx > -1) this.selectedNodes.splice(idx, 1);
    else {
        if (this.selectedNodes.length >= 2) this.selectedNodes.shift();
        this.selectedNodes.push(id);
    }
    this.updateSelectionUI();
    this.renderer.updateSelection(this.selectedNodes);
};

ChainMapUI.prototype.compareSelected = function() {
    const [left, right] = this.selectedNodes;
    window.location.hash = `#compare?left=${encodeURIComponent(left)}&right=${encodeURIComponent(right)}`;
};
