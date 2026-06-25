globalThis.ChainMapUI.prototype.updateHUD = function ({ k }) {
    if (this.zoomScaleEl) {
        this.zoomScaleEl.textContent = (k || 1).toFixed(2) + "x";
    }
    if (this.viewCoordsEl && this.renderer && this.renderer.controls) {
        const t = this.renderer.controls.target;
        this.viewCoordsEl.textContent = "X: " + Math.round(t.x) + ", Y: " + Math.round(t.y);
    }
};

globalThis.ChainMapUI.prototype.updateSelectionUI = function () {
    const c = this.selectedNodes.length;
    if (this.selectedCountEl) {
        this.selectedCountEl.textContent = c;
    }
    if (this.compareSelectedBtn) {
        this.compareSelectedBtn.classList.toggle("hidden", c < 2);
    }
};
