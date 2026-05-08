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
        .scaleExtent(MAP_INTERACTION.zoomExtent)
        .filter((event) => {
            if (event.type === "mousedown" || event.type === "touchstart") {
                const p = d3.pointer(event, this.renderer?.canvas);
                const hit = this.renderer?.hitTest({ x: p[0], y: p[1] });
                if (hit) return false; 
            }
            return !event.ctrlKey && !event.button;
        })
        .on("zoom", (event) => {
            this.g.attr("transform", event.transform);
            this.updateHUD(event.transform);

            const k = event.transform.k;
            this.renderer?.setDetailLevel({
                showLabels: this.labelsOverridden ? this.renderer.detailLevel.showLabels : k > 0.25,
                showArrows: k > 0.15,
                showNodeBorders: k > 0.2,
                showLinks: k > 0.05,
                labelCap: 500,
                arrowCap: 2000,
                zoom: k
            });
            this.renderer?.setTransform(event.transform);
        });

    this.renderer = new ChainMapRenderer({
        container: this.container,
        svg: this.svg,
        g: this.g,
        interaction: MAP_INTERACTION
    });

    const canvas = d3.select(this.renderer.canvas);

    d3.select(this.container).call(this.zoom).on("dblclick.zoom", null);
    canvas.call(this.zoom).on("dblclick.zoom", null);

    canvas.call(d3.drag()
        .container(this.renderer.canvas)
        .subject((event) => {
            const p = d3.pointer(event, this.renderer.canvas);
            return this.renderer?.hitTest({ x: p[0], y: p[1] });
        })
        .on("start", (event) => {
            if (this.chainSim?.isPaused) this.chainSim.play();
            if (!event.active) this.chainSim?.simulation?.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        })
        .on("drag", (event) => {
            const transform = d3.zoomTransform(this.renderer.canvas);
            event.subject.fx = (event.x - transform.x) / transform.k;
            event.subject.fy = (event.y - transform.y) / transform.k;
        })
        .on("end", (event) => {
            if (!event.active) this.chainSim?.simulation?.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        })
    );

    canvas.on("mousemove", (event) => {
        const p = d3.pointer(event, this.renderer.canvas);
        const node = this.renderer?.hitTest({ x: p[0], y: p[1] });
        if (node) {
            this.showTooltip(event, node);
            canvas.style("cursor", "pointer");
        } else {
            this.hideTooltip();
            canvas.style("cursor", "default");
        }
    });

    canvas.on("click", (event) => {
        if (event.defaultPrevented) return;
        const p = d3.pointer(event, this.renderer.canvas);
        const node = this.renderer?.hitTest({ x: p[0], y: p[1] });
        if (node) this.showNodeDetails(node);
    });
};

ChainMapUI.prototype.focusNode = function(id) {
    const node = this.chainSim.nodes.find(n => n.id === id);
    if (!node) return;

    this.svg.transition().duration(750).call(
        this.zoom.transform,
        d3.zoomIdentity
            .translate(this.width / 2, this.height / 2)
            .scale(1.5)
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
