globalThis.ChainMapUI.prototype.focusNode = function (id) {
    const node = this._simNodes.find(n => n.id === id);
    if (!node) {
        return;
    }
    const controls = this.renderer.controls;
    if (!controls) {
        return;
    }
    const targetX = node.x;
    const targetY = node.y;
    const originX = controls.target.x;
    const originY = controls.target.y;
    let step = 0;
    const anim = () => {
        step = step + 1;
        const t = step / 30;
        const eased = 1 - Math.pow(1 - t, 3);
        controls.target.set(originX + (targetX - originX) * eased, originY + (targetY - originY) * eased, 0);
        controls.update();
        if (step < 30) {
            requestAnimationFrame(anim);
        }
    };
    anim();
};

globalThis.ChainMapUI.prototype.toggleSelection = function (id) {
    const index = this.selectedNodes.indexOf(id);
    if (index > -1) {
        this.selectedNodes.splice(index, 1);
    } else {
        if (this.selectedNodes.length >= 2) {
            this.selectedNodes.shift();
        }
        this.selectedNodes.push(id);
    }
    this.updateSelectionUI();
    this.renderer.updateSelection(this.selectedNodes);
};

globalThis.ChainMapUI.prototype.compareSelected = function () {
    window.location.hash = "#compare?left=" + encodeURIComponent(this.selectedNodes[0]) + "&right=" + encodeURIComponent(this.selectedNodes[1]);
};
