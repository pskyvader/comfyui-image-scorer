globalThis.ChainMapUI.prototype.setupRenderer = function () {
    this.renderer = new globalThis.ThreeGraphRenderer({ container: this.container });
    const canvas = this.renderer.canvas;
    if (!canvas) {
        return;
    }
    const self = this;
    const sub = this.renderer;
    let lastZoom = 0;

    if (sub.controls) {
        sub.controls.addEventListener("change", () => {
            lastZoom = Date.now();
            self.updateHUD({ k: sub.camera.zoom });
        });
    }

    canvas.addEventListener("pointerdown", (e) => {
        sub._pointerDown = { x: e.clientX, y: e.clientY, t: Date.now() };
    });
    canvas.addEventListener("pointermove", (event) => {
        const hitNode = self.renderer.hitTest(event);
        if (hitNode) {
            self.showTooltip(event, hitNode);
            canvas.style.cursor = "pointer";
        } else {
            const hitLink = self.renderer.hitTestLink(event);
            if (hitLink) {
                self.hideTooltip();
                canvas.style.cursor = "pointer";
            } else {
                self.hideTooltip();
                canvas.style.cursor = "default";
            }
        }
    });
    canvas.addEventListener("pointerup", (event) => {
        const pd = sub._pointerDown;
        if (!pd) {
            return;
        }
        const dt = Date.now() - pd.t;
        const dx = event.clientX - pd.x;
        const dy = event.clientY - pd.y;
        if (dt > 500 || Math.sqrt(dx * dx + dy * dy) > 10) {
            return;
        }
        if (Date.now() - lastZoom < 300) {
            return;
        }
        if (self.nodeDetails && (self.nodeDetails === event.target || self.nodeDetails.contains(event.target))) {
            return;
        }
        const hitNode = self.renderer.hitTest(event);
        if (hitNode) {
            self.showNodeDetails(hitNode);
            return;
        }
        const hitLink = self.renderer.hitTestLink(event);
        if (hitLink) {
            self.showLinkDetails(hitLink);
            return;
        }
        self.hideNodeDetails();
    });
};
