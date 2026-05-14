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
        this.mode = "canvas";
        this.subRenderer.render({ nodes, links, profile, selectedIds, world });
        this.subRenderer.setDetailLevel?.(this.detailLevel);
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
        this.subRenderer?.destroy?.();
        this.subRenderer = null;
        this.mode = "canvas";
    }
}
