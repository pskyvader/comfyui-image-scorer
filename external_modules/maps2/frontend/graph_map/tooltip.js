globalThis.ChainMapUI.prototype.showTooltip = function (e, node) {
    this.tooltip.classList.remove("hidden");
    const r = this.container.getBoundingClientRect();
    this.tooltip.style.left = (e.clientX - r.left + 10) + "px";
    this.tooltip.style.top = (e.clientY - r.top + 10) + "px";
    const sid = node.id.replace(/'/g, "\\'");
    let ch = "";
    if (node._chainPrev || node._chainNext) {
        ch = "<div class=\"text-[8px] text-purple-400\">" + (node._chainPrev
            ? "Chain \u2191 " + node._chainPrev.split("/")
                .pop()
            : "Chain \u2191 (start)") + " | " + (node._chainNext
            ? "Chain \u2193 " + node._chainNext.split("/")
                .pop()
            : "Chain \u2193 (end)") + "</div>";
    }
    this.tooltip.innerHTML = "<div class=\"font-semibold truncate\" style=\"cursor:pointer\" onclick=\"window.chainMapUI.showNodeDetailsById('" + sid + "')\">" + node.id.split("/")
        .pop() + "</div><div class=\"text-[9px] text-gray-400\">" + node.score.toFixed(3) + " \u00B7 cmp:" + node.comparison_count + " \u00B7 comp:" + (node._compSize || "?") + "</div>" + ch;
};

globalThis.ChainMapUI.prototype.hideTooltip = function () {
    if (this._ttTimer) {
        clearTimeout(this._ttTimer);
    }
    this._ttTimer = setTimeout(() => this.tooltip.classList.add("hidden"), 200);
};
