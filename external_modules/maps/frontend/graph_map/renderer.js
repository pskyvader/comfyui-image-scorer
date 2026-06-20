class ChainMapRenderer {
    constructor({ container }) {
        this.container = container;
        this.detailLevel = {
            showNodeBorders: false,
            showLabels: true,
            showArrows: false,
            showLinks: true,
            labelCap: 500,
            arrowCap: 0,
            zoom: 1,
        };
        this.subRenderer = new ThreeGraphRenderer({ container: this.container });
    }

    get canvas() {
        return this.subRenderer.canvas;
    }

    get transform() {
        return this.subRenderer.transform;
    }

    render({ nodes, links, profile, selectedIds, dragBehavior, world }) {
        this.subRenderer.render({ nodes, links, profile, selectedIds, world });
    }

    requestRender() {
        this.subRenderer.requestRender();
    }

    setTransform(transform) {
        this.subRenderer.setTransform(transform);
    }

    setDetailLevel(detailLevel) {
        this.detailLevel = { ...this.detailLevel, ...detailLevel };
        this.subRenderer.setDetailLevel(this.detailLevel);
    }

    setLinkVisibility(showMain, showRegular) {
        this.subRenderer.setLinkVisibility(showMain, showRegular);
    }

    setHighlightedChain(chainId) {
        this.subRenderer.setHighlightedChain(chainId);
    }

    updateSelection(selectedIds) {
        this.subRenderer.updateSelection(selectedIds);
    }

    resize() {
        this.subRenderer.resize();
    }

    hitTest(event) {
        return this.subRenderer.hitTest(event);
    }

    hitTestLink(event) {
        return this.subRenderer.hitTestLink(event);
    }

    worldToScreen(worldX, worldY) {
        return this.subRenderer.worldToScreen(worldX, worldY);
    }

    isCanvasMode() {
        return false;
    }

    destroy() {
        this.subRenderer.destroy();
        this.subRenderer = null;
    }
}
