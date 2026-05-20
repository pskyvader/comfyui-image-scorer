/**
 * Interaction Logic (Zoom, Drag, Click) for ChainMapUI
 */

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
                const p = d3.pointer(event, this.renderer.canvas);
                const hit = this.renderer.hitTest({ x: p[0], y: p[1] });
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
        const p = d3.pointer(event, this.renderer.canvas);
        const node = this.renderer.hitTest({ x: p[0], y: p[1] });
        if (node) {
            this.showTooltip(event, node);
            canvas.style("cursor", "pointer");
        } else {
            const link = this.renderer.hitTestLink({ x: p[0], y: p[1] });
            if (link) {
                this.hideTooltip();
                canvas.style("cursor", "pointer");
            } else {
                this.hideTooltip();
                canvas.style("cursor", "default");
            }
        }
    });

    canvas.on("click", (event) => {
        if (event.defaultPrevented) return;
        const p = d3.pointer(event, this.renderer.canvas);
        const node = this.renderer.hitTest({ x: p[0], y: p[1] });
        if (node) {
            this.showNodeDetails(node);
            return;
        }
        const link = this.renderer.hitTestLink({ x: p[0], y: p[1] });
        if (link) {
            this.showLinkDetails(link);
            return;
        }
        this.clearChainSelection();
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
