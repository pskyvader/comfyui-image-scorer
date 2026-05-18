/**
 * Low-level Canvas Graph Renderer
 */

class CanvasGraphRenderer {
    constructor({ container, svg }) {
        this.container = container;
        this.svg = svg;
        this.canvas = document.createElement("canvas");
        this.canvas.className = "chain-map-canvas";
        this.canvas.setAttribute("aria-hidden", "true");
        this.canvas.style.pointerEvents = "auto";
        this.container.insertBefore(this.canvas, this.container.firstChild);

        this.ctx = this.canvas.getContext("2d", { alpha: true });
        this.nodes = [];
        this.links = [];
        this.profile = {};
        this.transform = d3.zoomIdentity;
        this.selectedIds = new Set();
        this.pendingFrame = 0;
        this.visibleNodeCount = 0;
        this.visibleLinkCount = 0;
        this.nodeBatchSize = 0;
        this.linkBatchSize = 0;
        this.detailLevel = {
            showNodeBorders: false,
            showLabels: false,
            showArrows: false,
            showLinks: true,
            labelCap: 0,
            arrowCap: 0,
            zoom: 1,
        };
        this.dpr = window.devicePixelRatio;
        this.worldBounds = null;
        this._tickCount = 0;
    }

    render({ nodes, links, profile, selectedIds, world, initialNodeCount }) {
        this.nodes = nodes;
        this.links = links;
        this.profile = profile;
        this.selectedIds = new Set(selectedIds);
        this.world = world;
        this.worldBounds = world ? { x: world.x, y: world.y, width: world.width, height: world.height } : null;
        this.visibleNodeCount = initialNodeCount || 0;
        this.visibleLinkCount = 0;
        this.worldBoundsFrozen = false;
        this.requestRender();
    }

    freezeWorldBounds() {
        if (this.nodes.length > 0) {
            let minX = Infinity; let maxX = -Infinity; let minY = Infinity; let maxY = -Infinity;
            for (const n of this.nodes) {
                if (n.x < minX) {
                    minX = n.x;
                }
                if (n.x > maxX) {
                    maxX = n.x;
                }
                if (n.y < minY) {
                    minY = n.y;
                }
                if (n.y > maxY) {
                    maxY = n.y;
                }
            }
            const wb = RENDER.bounds;
            const padding = wb.padding;
            this.worldBounds = { x: minX - padding, y: minY - padding, width: (maxX - minX) + padding * 2, height: (maxY - minY) + padding * 2 };
            this.worldBoundsFrozen = true;
        }
    }

    update(nodes, links) {
        this.nodes = nodes;
        this.links = links;
        this.requestRender();
    }

    setTransform(transform) {
        this.transform = transform;
        this.requestRender();
    }

    setDetailLevel(detailLevel) {
        this.detailLevel = detailLevel;
        this.requestRender();
    }

    updateSelection(selectedIds) {
        this.selectedIds = new Set(selectedIds);
        this.requestRender();
    }

    hitTest(event) {
        if (!CONTROLS.enablePointerHits) {
            return null;
        }

        let px; let py;
        if (event.sourceEvent) {
            [px, py] = d3.pointer(event.sourceEvent, this.svg.node());
        } else if (event.x !== undefined && event.y !== undefined && event.sourceEvent === undefined) {
            px = event.x;
            py = event.y;
        } else {
            [px, py] = d3.pointer(event, this.svg.node());
        }

        const scale = Math.max(this.transform.k, CAMERA.minScale);
        const worldX = (px - this.transform.x) / scale;
        const worldY = (py - this.transform.y) / scale;

        const padding = RENDER.viewport.hitTestPadding / scale;
        let bestNode = null;
        let bestDistanceSq = Infinity;

        for (let i = Math.min(this.visibleNodeCount, this.nodes.length) - 1; i >= 0; i--) {
            const node = this.nodes[i];
            const dx = worldX - node.x;
            const dy = worldY - node.y;
            const distSq = dx * dx + dy * dy;
            const radius = node._radius + padding;

            if (distSq <= radius * radius) {
                if (distSq < bestDistanceSq) {
                    bestDistanceSq = distSq;
                    bestNode = node;
                }
            }
        }
        return bestNode;
    }

    resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.canvas.width = Math.round(width * this.dpr);
        this.canvas.height = Math.round(height * this.dpr);
        this.canvas.style.width = `${width}px`;
        this.canvas.style.height = `${height}px`;
        this.requestRender();
    }

    requestRender() {
        if (this.pendingFrame) {
            return;
        }
        this.pendingFrame = window.requestAnimationFrame(() => this.draw());
    }

    draw() {
        this.pendingFrame = 0;
        if (!this.ctx || !this.profile) {
            return;
        }

        const width = this.canvas.width;
        const height = this.canvas.height;
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.clearRect(0, 0, width, height);
        this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
        this.ctx.translate(this.transform.x, this.transform.y);
        this.ctx.scale(this.transform.k, this.transform.k);

        const viewport = this.getViewportBounds(CAMERA.viewportPadding / Math.max(this.transform.k, CAMERA.minScale));
        this.drawLinks(viewport);
        this.drawNodes();
        this.drawSelectedNodes();
        this.drawNodeDetails(viewport);

        if (this.worldBounds) {
            const wbc = RENDER.bounds;
            const wb = this.worldBounds;
            this.ctx.strokeStyle = wbc.strokeColor;
            this.ctx.lineWidth = wbc.lineWidth / this.transform.k;
            const dash = wbc.dashArray;
            this.ctx.setLineDash([dash[0] / this.transform.k, dash[1] / this.transform.k]);
            this.ctx.strokeRect(wb.x, wb.y, wb.width, wb.height);
            this.ctx.setLineDash([]);
        }

    }

    drawLinks(viewport) {
        if (!this.profile.drawLinks || !this.detailLevel.showLinks || this.transform.k < (this.profile.linkVisibilityZoomThreshold)) {
            return;
        }

        const visibleLinks = Math.min(this.visibleLinkCount, this.links.length);
        const visibleNodeIds = new Set();
        for (let i = 0; i < this.visibleNodeCount; i++) {
            visibleNodeIds.add(this.nodes[i].id);
        }

        this.ctx.lineWidth = RENDER.link.linkLineWidth;
        this.ctx.strokeStyle = RENDER.link.linkColor;

        const maxVisualDist = RENDER.link.linkOpacityMaxDist;

        if (!this.profile.usePerLinkOpacity) {
            this.ctx.globalAlpha = RENDER.link.linkOpacityMax;
            this.ctx.beginPath();
            for (let i = 0; i < visibleLinks; i++) {
                const link = this.links[i];
                if (!visibleNodeIds.has(link.source.id) || !visibleNodeIds.has(link.target.id)) {
                    continue;
                }
                const dx = link.target.x - link.source.x;
                const dy = link.target.y - link.source.y;
                if ((dx * dx + dy * dy) > maxVisualDist * maxVisualDist) {
                    continue;
                }
                const midX = (link.source.x + link.target.x) / 2;
                const midY = (link.source.y + link.target.y) / 2;
                this.ctx.moveTo(link.source.x, link.source.y);
                this.ctx.lineTo(midX, midY);
                this.ctx.lineTo(link.target.x, link.target.y);
            }
            this.ctx.stroke();
            this.ctx.globalAlpha = 1;
            if (this.detailLevel.showArrows && this.detailLevel.arrowCap > 0) {
                this.drawLinkArrows(viewport);
            }
            return;
        }

        for (let i = 0; i < visibleLinks; i++) {
            const link = this.links[i];
            const source = link.source;
            const target = link.target;

            if (!visibleNodeIds.has(source.id) || !visibleNodeIds.has(target.id)) {
                continue;
            }

            if (!this.isNodeInViewport(source, viewport, RENDER.link.linkViewportPadding) && !this.isNodeInViewport(target, viewport, RENDER.link.linkViewportPadding)) {
                continue;
            }

            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const distSq = dx * dx + dy * dy;
            if (distSq > maxVisualDist * maxVisualDist) {
                continue;
            }
            const dist = Math.sqrt(distSq);

            const minDist = RENDER.link.linkOpacityMinDist;
            const maxDist = RENDER.link.linkOpacityMaxDist;
            const opacity = RENDER.link.linkOpacityMax - t * (RENDER.link.linkOpacityMax - RENDER.link.linkOpacityMin);

            this.ctx.globalAlpha = opacity;
            this.ctx.beginPath();
            const midX = (source.x + target.x) / 2;
            const midY = (source.y + target.y) / 2;
            this.ctx.moveTo(source.x, source.y);
            this.ctx.lineTo(midX, midY);
            this.ctx.lineTo(target.x, target.y);
            this.ctx.stroke();
        }
        this.ctx.globalAlpha = 1;

        if (this.detailLevel.showArrows && this.detailLevel.arrowCap > 0) {
            this.drawLinkArrows(viewport);
        }
    }

    drawNodes() {
        const visibleNodes = Math.min(this.visibleNodeCount, this.nodes.length);
        let fill = "";
        this.ctx.globalAlpha = 1;
        const k = this.transform.k;
        const minR = RENDER.node.minScreenRadius / k;

        for (let i = 0; i < visibleNodes; i++) {
            const node = this.nodes[i];
            if (this.selectedIds.has(node.id)) {
                continue;
            }
            if (node._fill !== fill) {
                fill = node._fill;
                this.ctx.fillStyle = fill;
            }
            const r = Math.max(node._radius, minR);
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    drawSelectedNodes() {
        if (!this.selectedIds.size) {
            return;
        }

        const visibleNodes = Math.min(this.visibleNodeCount, this.nodes.length);
        this.ctx.lineWidth = RENDER.node.selectedLineWidth;
        this.ctx.strokeStyle = RENDER.node.highlightColor;
        const k = this.transform.k;
        const minR = RENDER.node.minScreenRadius / k;

        for (let i = 0; i < visibleNodes; i++) {
            const node = this.nodes[i];
            if (!this.selectedIds.has(node.id)) {
                continue;
            }
            this.ctx.fillStyle = node._fill;
            const r = Math.max(node._radius, minR);
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.stroke();
        }
    }

    drawNodeDetails(viewport) {
        if (!this.detailLevel.showNodeBorders && !this.detailLevel.showLabels) {
            return;
        }

        const visibleNodes = Math.min(this.visibleNodeCount, this.nodes.length);

        if (this.detailLevel.showNodeBorders) {
            this.ctx.strokeStyle = RENDER.node.borderStroke;
            this.ctx.lineWidth = RENDER.node.borderLineWidth;
            const k = this.transform.k;
            const minR = RENDER.node.minScreenRadius / k;
            for (let i = 0; i < visibleNodes; i++) {
                const node = this.nodes[i];
                const r = Math.max(node._radius, minR);
                if (!this.isNodeInViewport(node, viewport, r + RENDER.border.nodeBorderViewportPadding)) {
                    continue;
                }
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
                this.ctx.stroke();
            }
        }

        if (!this.detailLevel.showLabels || this.detailLevel.labelCap <= 0 || this.transform.k < CAMERA.labelThreshold) {
            return;
        }

        const labelLimit = Math.min(this.detailLevel.labelCap, visibleNodes);
        let drawn = 0;
        this.ctx.fillStyle = RENDER.label.labelColor;
        this.ctx.textBaseline = "middle";

        let lastFontSize = -1;

        for (let i = 0; i < visibleNodes && drawn < labelLimit; i++) {
            const node = this.nodes[i];
            const r = Math.max(node._radius, RENDER.node.minScreenRadius / this.transform.k);
            if (!this.isNodeInViewport(node, viewport, r + RENDER.label.labelViewportPadding)) {
                continue;
            }
            const fontSize = Math.max(RENDER.label.labelFontSizeMin, Math.min(RENDER.label.labelFontSizeMax, Math.round(r * RENDER.label.labelFontSizeMultiplier)));
            if (fontSize !== lastFontSize) {
                this.ctx.font = `${fontSize}px sans-serif`;
                lastFontSize = fontSize;
            }
            this.ctx.fillText(node._shortLabel, node.x + r + RENDER.label.labelOffset, node.y);
            drawn++;
        }
    }

    drawLinkArrows(viewport) {
        const visibleLinks = Math.min(this.visibleLinkCount, this.links.length);
        const cap = Math.min(this.detailLevel.arrowCap, visibleLinks);
        let drawn = 0;
            const minLengthSq = RENDER.arrow.arrowMinLengthSq;

        this.ctx.fillStyle = RENDER.arrow.arrowColor;
        this.ctx.globalAlpha = 1;

        const visibleNodeIds = new Set();
        for (let i = 0; i < this.visibleNodeCount; i++) {
            visibleNodeIds.add(this.nodes[i].id);
        }

        for (let i = 0; i < visibleLinks && drawn < cap; i++) {
            const link = this.links[i];
            const source = link.source;
            const target = link.target;
            if (!visibleNodeIds.has(source.id) || !visibleNodeIds.has(target.id)) {
                continue;
            }
            if (!this.isNodeInViewport(source, viewport, RENDER.arrow.arrowViewportPadding) && !this.isNodeInViewport(target, viewport, RENDER.arrow.arrowViewportPadding)) {
                continue;
            }

            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const lenSq = (dx * dx) + (dy * dy);
            if (lenSq < minLengthSq || lenSq > maxVisualDist * maxVisualDist) {
                continue;
            }

            const midX = (source.x + target.x) * 0.5;
            const midY = (source.y + target.y) * 0.5;
            this.drawArrowhead(midX - (dx * RENDER.arrow.arrowMidOffsetRatio), midY - (dy * RENDER.arrow.arrowMidOffsetRatio), dx, dy, RENDER.arrow.arrowheadSize);
            drawn++;
        }
    }

    drawArrowhead(x, y, dx, dy, size) {
        const length = Math.sqrt((dx * dx) + (dy * dy));
        const ux = dx / length;
        const uy = dy / length;
        const px = -uy;
        const py = ux;

        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
        this.ctx.lineTo(x - (ux * size) + (px * size * RENDER.arrow.arrowheadWingRatio), y - (uy * size) + (py * size * RENDER.arrow.arrowheadWingRatio));
        this.ctx.lineTo(x - (ux * size) - (px * size * RENDER.arrow.arrowheadWingRatio), y - (uy * size) - (py * size * RENDER.arrow.arrowheadWingRatio));
        this.ctx.closePath();
        this.ctx.fill();
    }

    getViewportBounds(pad = 0) {
        const scale = Math.max(this.transform.k, CAMERA.minScale);
        const width = this.canvas.width / this.dpr;
        const height = this.canvas.height / this.dpr;
        return {
            minX: ((-this.transform.x) / scale) - pad,
            maxX: ((width - this.transform.x) / scale) + pad,
            minY: ((-this.transform.y) / scale) - pad,
            maxY: ((height - this.transform.y) / scale) + pad,
        };
    }

    isNodeInViewport(node, viewport, padding = 0) {
        return node.x >= (viewport.minX - padding)
            && node.x <= (viewport.maxX + padding)
            && node.y >= (viewport.minY - padding)
            && node.y <= (viewport.maxY + padding);
    }

    destroy() {
        if (this.pendingFrame) {
            window.cancelAnimationFrame(this.pendingFrame);
        }
        this.pendingFrame = 0;
        this.canvas.remove();
    }
}
