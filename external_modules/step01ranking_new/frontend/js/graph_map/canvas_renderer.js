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
        this.profile = null;
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
        this.dpr = Math.max(1, window.devicePixelRatio || 1);
        this.resize();
    }

    render({ nodes, links, profile, selectedIds, world }) {
        this.nodes = nodes;
        this.links = links;
        this.profile = profile;
        this.selectedIds = new Set(selectedIds);
        this.world = world;
        this.visibleNodeCount = profile.progressiveReveal ? 0 : nodes.length;
        this.visibleLinkCount = profile.progressiveReveal ? 0 : links.length;
        this.nodeBatchSize = Math.max(800, Math.ceil(nodes.length / 12));
        this.linkBatchSize = Math.max(1500, Math.ceil(links.length / 12));
        this.requestRender();
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
        this.detailLevel = detailLevel || this.detailLevel;
        this.requestRender();
    }

    updateSelection(selectedIds) {
        this.selectedIds = new Set(selectedIds);
        this.requestRender();
    }

    hitTest(event) {
        if (!this.profile?.enablePointerHits) {
            return null;
        }

        let px, py;
        if (event.sourceEvent) {
            [px, py] = d3.pointer(event.sourceEvent, this.svg.node());
        } else if (event.x !== undefined && event.y !== undefined && event.sourceEvent === undefined) {
            px = event.x;
            py = event.y;
        } else {
            [px, py] = d3.pointer(event, this.svg.node());
        }

        const scale = Math.max(this.transform.k, 0.001);
        const worldX = (px - this.transform.x) / scale;
        const worldY = (py - this.transform.y) / scale;
        
        const padding = 10 / scale; 
        let bestNode = null;
        let bestDistanceSq = Infinity;

        for (let i = Math.min(this.visibleNodeCount, this.nodes.length) - 1; i >= 0; i--) {
            const node = this.nodes[i];
            const dx = worldX - node.x;
            const dy = worldY - node.y;
            const distSq = dx * dx + dy * dy;
            const radius = (node._radius || 5) + padding;

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
        const width = this.container.clientWidth || 800;
        const height = this.container.clientHeight || 600;
        this.canvas.width = Math.round(width * this.dpr);
        this.canvas.height = Math.round(height * this.dpr);
        this.canvas.style.width = `${width}px`;
        this.canvas.style.height = `${height}px`;
        this.requestRender();
    }

    requestRender() {
        if (this.pendingFrame) return;
        this.pendingFrame = window.requestAnimationFrame(() => this.draw());
    }

    draw() {
        this.pendingFrame = 0;
        if (!this.ctx || !this.profile) return;

        const width = this.canvas.width;
        const height = this.canvas.height;
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.clearRect(0, 0, width, height);
        this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
        this.ctx.translate(this.transform.x, this.transform.y);
        this.ctx.scale(this.transform.k, this.transform.k);

        if (this.profile.progressiveReveal) {
            this.visibleNodeCount = Math.min(this.nodes.length, this.visibleNodeCount + this.nodeBatchSize);
            this.visibleLinkCount = Math.min(this.links.length, this.visibleLinkCount + this.linkBatchSize);
        }

        const viewport = this.getViewportBounds(24 / Math.max(this.transform.k, 0.001));
        this.drawLinks(viewport);
        this.drawNodes();
        this.drawSelectedNodes();
        this.drawNodeDetails(viewport);

        // Draw world bounds
        if (this.nodes.length > 0) {
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            const limit = Math.min(this.visibleNodeCount, this.nodes.length);
            for (let i = 0; i < limit; i++) {
                const n = this.nodes[i];
                if (n.x < minX) minX = n.x;
                if (n.x > maxX) maxX = n.x;
                if (n.y < minY) minY = n.y;
                if (n.y > maxY) maxY = n.y;
            }
            
            const padding = 150;
            const wX = minX - padding;
            const wY = minY - padding;
            const wWidth = (maxX - minX) + padding * 2;
            const wHeight = (maxY - minY) + padding * 2;

            this.ctx.strokeStyle = "rgba(139, 92, 246, 0.8)";
            this.ctx.lineWidth = 2 / this.transform.k; 
            this.ctx.setLineDash([12 / this.transform.k, 6 / this.transform.k]);
            this.ctx.strokeRect(wX, wY, wWidth, wHeight);
            this.ctx.setLineDash([]);
        }

        if (this.visibleNodeCount < this.nodes.length || this.visibleLinkCount < this.links.length) {
            this.requestRender();
        }
    }

    drawLinks(viewport) {
        if (!this.profile.drawLinks || !this.detailLevel.showLinks || this.transform.k < (this.profile.linkVisibilityZoomThreshold || 0.05)) {
            return;
        }

        const visibleLinks = Math.min(this.visibleLinkCount, this.links.length);
        this.ctx.lineWidth = 1;
        this.ctx.strokeStyle = "#ffffff";

        if (!this.profile.usePerLinkOpacity) {
            this.ctx.globalAlpha = this.profile.linkGlobalOpacity;
            this.ctx.beginPath();
            for (let i = 0; i < visibleLinks; i++) {
                const link = this.links[i];
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
            
            if (!this.isNodeInViewport(source, viewport, 10) && !this.isNodeInViewport(target, viewport, 10)) {
                continue;
            }

            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            
            const maxDist = 600;
            const opacity = Math.max(0.1, Math.min(1.0, 1 - (dist / maxDist)));
            
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

        for (let i = 0; i < visibleNodes; i++) {
            const node = this.nodes[i];
            if (this.selectedIds.has(node.id)) continue;
            if (node._fill !== fill) {
                fill = node._fill;
                this.ctx.fillStyle = fill;
            }
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, node._radius, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    drawSelectedNodes() {
        if (!this.selectedIds.size) return;

        const visibleNodes = Math.min(this.visibleNodeCount, this.nodes.length);
        this.ctx.lineWidth = 2;
        // Access config via global variable
        this.ctx.strokeStyle = (typeof MAP_VISUALS !== 'undefined') ? MAP_VISUALS.highlightColor : "#f472b6";

        for (let i = 0; i < visibleNodes; i++) {
            const node = this.nodes[i];
            if (!this.selectedIds.has(node.id)) continue;
            this.ctx.fillStyle = node._fill;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, node._radius, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.stroke();
        }
    }

    drawNodeDetails(viewport) {
        if (!this.detailLevel.showNodeBorders && !this.detailLevel.showLabels) return;

        const visibleNodes = Math.min(this.visibleNodeCount, this.nodes.length);

        if (this.detailLevel.showNodeBorders) {
            this.ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
            this.ctx.lineWidth = 0.9;
            for (let i = 0; i < visibleNodes; i++) {
                const node = this.nodes[i];
                if (!this.isNodeInViewport(node, viewport, node._radius + 1)) continue;
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, node._radius, 0, Math.PI * 2);
                this.ctx.stroke();
            }
        }

        if (!this.detailLevel.showLabels || this.detailLevel.labelCap <= 0 || this.transform.k < 0.25) return;

        const labelLimit = Math.min(this.detailLevel.labelCap, visibleNodes);
        let drawn = 0;
        this.ctx.fillStyle = "rgba(226, 232, 240, 0.92)";
        this.ctx.textBaseline = "middle";

        let lastFontSize = -1;

        for (let i = 0; i < visibleNodes && drawn < labelLimit; i++) {
            const node = this.nodes[i];
            if (!this.isNodeInViewport(node, viewport, node._radius + 20)) continue;
            const fontSize = Math.max(6, Math.min(16, Math.round(node._radius * 1.6)));
            if (fontSize !== lastFontSize) {
                this.ctx.font = `${fontSize}px sans-serif`;
                lastFontSize = fontSize;
            }
            this.ctx.fillText(node._shortLabel || node._label || String(node.id), node.x + node._radius + 3, node.y);
            drawn++;
        }
    }

    drawLinkArrows(viewport) {
        const visibleLinks = Math.min(this.visibleLinkCount, this.links.length);
        const cap = Math.min(this.detailLevel.arrowCap, visibleLinks);
        let drawn = 0;
        const minLengthSq = 36;

        this.ctx.fillStyle = "rgba(248, 250, 252, 0.62)";
        this.ctx.globalAlpha = 1;

        for (let i = 0; i < visibleLinks && drawn < cap; i++) {
            const link = this.links[i];
            const source = link.source;
            const target = link.target;
            if (!this.isNodeInViewport(source, viewport, 12) && !this.isNodeInViewport(target, viewport, 12)) continue;

            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const lenSq = (dx * dx) + (dy * dy);
            if (lenSq < minLengthSq) continue;

            const midX = (source.x + target.x) * 0.5;
            const midY = (source.y + target.y) * 0.5;
            this.drawArrowhead(midX - (dx * 0.07), midY - (dy * 0.07), dx, dy, 2.8);
            drawn++;
        }
    }

    drawArrowhead(x, y, dx, dy, size) {
        const length = Math.sqrt((dx * dx) + (dy * dy)) || 1;
        const ux = dx / length;
        const uy = dy / length;
        const px = -uy;
        const py = ux;

        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
        this.ctx.lineTo(x - (ux * size) + (px * size * 0.7), y - (uy * size) + (py * size * 0.7));
        this.ctx.lineTo(x - (ux * size) - (px * size * 0.7), y - (uy * size) - (py * size * 0.7));
        this.ctx.closePath();
        this.ctx.fill();
    }

    getViewportBounds(pad = 0) {
        const scale = Math.max(this.transform.k, 0.001);
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
        if (this.pendingFrame) window.cancelAnimationFrame(this.pendingFrame);
        this.pendingFrame = 0;
        this.canvas.remove();
    }
}
