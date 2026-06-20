ChainMapUI.prototype.setupThreeRenderer = function() {
    this.renderer = new ChainMapRenderer({
        container: this.container,
    });

    const canvas = this.renderer.canvas;
    if (!canvas) return;

    const self = this;
    let lastZoomTime = 0;
    const sub = this.renderer.subRenderer;

    if (sub.controls) {
        sub.controls.addEventListener('change', () => {
            const cam = sub.camera;
            const zoom = cam.zoom;
            lastZoomTime = Date.now();

            self.updateHUD({ k: zoom });

            self.renderer.setDetailLevel({
                showLabels: self.labelsOverridden ? self.renderer.detailLevel.showLabels : zoom > CAMERA.labelThreshold,
                showArrows: zoom > CAMERA.arrowThreshold,
                showNodeBorders: zoom > CAMERA.borderThreshold,
                showLinks: true,
                labelCap: RENDER.label.labelCap,
                arrowCap: RENDER.arrow.arrowCap,
                zoom: zoom,
            });
        });
    }

    canvas.addEventListener("mousemove", (event) => {
        const node = self.renderer.hitTest({ x: event.clientX, y: event.clientY });
        if (node) {
            self.showTooltip(event, node);
            canvas.style.cursor = "pointer";
        } else {
            const link = self.renderer.hitTestLink({ x: event.clientX, y: event.clientY });
            if (link) {
                self.hideTooltip();
                canvas.style.cursor = "pointer";
            } else {
                self.hideTooltip();
                canvas.style.cursor = "default";
            }
        }
    });

    canvas.addEventListener("click", (event) => {
        if (self.renderer.subRenderer.didDrag) return;
        const timeSinceZoom = Date.now() - lastZoomTime;
        if (timeSinceZoom < 300) return;
        if (self.nodeDetails && (self.nodeDetails === event.target || self.nodeDetails.contains(event.target))) return;

        const node = self.renderer.hitTest({ x: event.clientX, y: event.clientY });
        if (node) {
            console.log("[click] hit node:", node.id);
            self.showNodeDetails(node);
            return;
        }
        const link = self.renderer.hitTestLink({ x: event.clientX, y: event.clientY });
        if (link) {
            console.log("[click] hit link:", link.source.id, "->", link.target.id);
            self.showLinkDetails(link);
            return;
        }
        self.hideNodeDetails();
    });

    this._zoomEnabled = true;
};

ChainMapUI.prototype.focusNode = function(id) {
    const node = this._simNodes.find(n => n.id === id);
    if (!node) return;

    const controls = this.renderer.subRenderer.controls;
    if (!controls) return;

    const targetX = node.x;
    const targetY = node.y;

    const cx = controls.target.x;
    const cy = controls.target.y;

    const steps = 30;
    let step = 0;
    const animate = () => {
        step++;
        const t = step / steps;
        const ease = 1 - Math.pow(1 - t, 3);
        controls.target.set(
            cx + (targetX - cx) * ease,
            cy + (targetY - cy) * ease,
            0
        );
        controls.update();
        if (step < steps) requestAnimationFrame(animate);
    };
    animate();
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
