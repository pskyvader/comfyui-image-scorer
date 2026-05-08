/**
 * High-level Graph Renderer Orchestrator
 */

class ChainMapRenderer {
    constructor({ container, svg, g, interaction }) {
        this.container = container;
        this.svg = svg;
        this.g = g;
        this.interaction = interaction;
        this.renderer = null;
        this.mode = "canvas";
        this.worldBounds = null;
        this.detailLevel = {
            showNodeBorders: true,
            showLabels: false,
            showArrows: false,
            labelCap: 0,
            arrowCap: 0,
            zoom: 1,
        };
        // CanvasGraphRenderer must be loaded before this
        this.subRenderer = new CanvasGraphRenderer({
            container: this.container,
            svg: this.svg,
        });
    }

    get canvas() {
        return this.subRenderer?.canvas;
    }

    render({ nodes, links, profile, selectedIds, dragBehavior, world }) {
        this.renderWorldBounds(world);
        this.mode = "canvas";
        this.subRenderer.render({ nodes, links, profile, selectedIds, world });
        this.subRenderer.setDetailLevel?.(this.detailLevel);
    }

    renderWorldBounds(world) {
        if (!world) return;
        if (this.worldBounds) this.worldBounds.remove();

        this.worldBounds = this.g.append("rect")
            .attr("class", "world-bounds")
            .attr("x", world.x)
            .attr("y", world.y)
            .attr("width", world.width)
            .attr("height", world.height)
            .attr("fill", "none")
            .attr("stroke", "rgba(139, 92, 246, 0.3)")
            .attr("stroke-dasharray", "10 5")
            .attr("vector-effect", "non-scaling-stroke");
    }

    update(nodes, links) {
        this.subRenderer?.update(nodes, links);
    }

    setTransform(transform) {
        this.subRenderer?.setTransform(transform);
    }

    setDetailLevel(detailLevel) {
        this.detailLevel = detailLevel || this.detailLevel;
        this.subRenderer?.setDetailLevel?.(this.detailLevel);
    }

    updateSelection(selectedIds) {
        this.subRenderer?.updateSelection?.(selectedIds);
    }

    resize() {
        this.subRenderer?.resize?.();
    }

    hitTest(event) {
        return this.subRenderer?.hitTest?.(event) ?? null;
    }

    isCanvasMode() {
        return true;
    }

    destroy() {
        this.g.selectAll("*").remove();
        this.worldBounds = null;
        this.subRenderer?.destroy?.();
        this.subRenderer = null;
        this.mode = "canvas";
    }
}
