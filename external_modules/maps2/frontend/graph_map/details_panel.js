globalThis.ChainMapUI.prototype.hideNodeDetails = function () {
    if (this.nodeDetails) {
        this.nodeDetails.classList.add("hidden");
    }
    this._activeNode = null;
    if (this.renderer) {
        this.renderer.setHighlightedChain(null);
    }
};

globalThis.ChainMapUI.prototype.showNodeDetails = function (d) {
    if (!this.nodeDetails) {
        return;
    }
    this._activeNode = d;
    if (this.renderer) {
        this.renderer.setHighlightedChain(d._chainId);
    }
    this.nodeDetails.classList.remove("hidden");
    this.updateDetailsPosition();
    const fn = d.id.split("/")
        .pop();
    this.nodeDetails.innerHTML = "<div class=\"flex justify-between items-start mb-2\"><h3 class=\"text-white font-bold text-[10px] truncate max-w-[80%]\">" + fn + "</h3><button onclick=\"window.chainMapUI.hideNodeDetails()\" class=\"text-gray-500 hover:text-white p-1\">&times;</button></div><div class=\"flex flex-col gap-2\"><img src=\"/images/" + encodeURIComponent(d.id) + "\" class=\"w-full h-32 md:h-48 object-contain rounded bg-black/40 border border-white/5\" onerror=\"this.src='/output/ranked/" + encodeURIComponent(d.id) + "'\"><div class=\"flex flex-wrap gap-x-3 gap-y-1 text-[9px] text-gray-300 justify-center\"><span class=\"flex items-center gap-1\">Score: <b class=\"text-purple-400\">" + d.score.toFixed(3) + "</b></span><span class=\"flex items-center gap-1\">Comparisons: <b class=\"text-purple-400\">" + d.comparison_count + "</b></span></div><div class=\"flex gap-2\"><button onclick=\"window.chainMapUI.toggleSelection('" + d.id + "')\" class=\"flex-1 py-1.5 bg-pink-600 hover:bg-pink-500 rounded text-[9px] font-bold transition\">Select</button><button onclick=\"window.chainMapUI.focusNode('" + d.id + "')\" class=\"px-3 py-1.5 bg-white/10 hover:bg-white/20 rounded text-[9px] transition\">Focus</button></div></div>";
};

globalThis.ChainMapUI.prototype.showLinkDetails = function (link) {
    if (!this.nodeDetails) {
        return;
    }
    this._activeNode = null;
    this.nodeDetails.classList.remove("hidden");
    this.nodeDetails.classList.add("top-4", "left-4");
    this.nodeDetails.style.left = this.nodeDetails.style.top = "";
    let ci = "";
    if (link.isMainChain && link.source._allChains && link.target._allChains) {
        const shared = link.source._allChains.filter(c => link.target._allChains.includes(c));
        if (shared.length) {
            if (this.renderer) {
                this.renderer.setHighlightedChain(shared[0]);
            }
            ci = "<div class=\"text-[9px] text-purple-400 mb-2 font-bold flex items-center justify-between\"><span>Main Chain Segment</span><span class=\"bg-purple-900/50 px-2 py-0.5 rounded\">Chain " + shared[0] + "</span></div>";
        } else if (this.renderer) {
            this.renderer.setHighlightedChain(null);
        }
    } else if (this.renderer) {
        this.renderer.setHighlightedChain(null);
    }
    this.nodeDetails.innerHTML = "<div class=\"flex justify-between items-start mb-2\"><h3 class=\"text-white font-bold text-[10px]\">Link Details</h3><button onclick=\"window.chainMapUI.hideNodeDetails()\" class=\"text-gray-500 hover:text-white p-1\">&times;</button></div>" + ci + "<div class=\"flex flex-col gap-2 text-[9px] text-gray-300\"><div class=\"flex justify-between items-center p-2 bg-white/5 rounded\"><div><b class=\"text-purple-400\">" + link.source.id.split("/")
        .pop() + "</b><span class=\"text-gray-500 ml-1\">score " + link.source.score.toFixed(3) + "</span></div><span class=\"text-gray-600\">&harr;</span><div class=\"text-right\"><b class=\"text-purple-400\">" + link.target.id.split("/")
        .pop() + "</b><span class=\"text-gray-500 ml-1\">score " + link.target.score.toFixed(3) + "</span></div></div><div class=\"flex gap-2\"><button onclick=\"window.chainMapUI.focusNode('" + link.source.id + "')\" class=\"flex-1 py-1.5 bg-white/10 hover:bg-white/20 rounded font-bold transition\">Focus Source</button><button onclick=\"window.chainMapUI.focusNode('" + link.target.id + "')\" class=\"flex-1 py-1.5 bg-white/10 hover:bg-white/20 rounded font-bold transition\">Focus Target</button></div></div>";
};

globalThis.ChainMapUI.prototype.showNodeDetailsById = function (id) {
    const n = this._simNodes.find(n => n.id === id);
    if (n) {
        this.showNodeDetails(n);
    }
};

globalThis.ChainMapUI.prototype.updateDetailsPosition = function () {
    if (!this._activeNode || !this.nodeDetails || this.nodeDetails.classList.contains("hidden")) {
        return;
    }
    if (!this.renderer) {
        return;
    }
    const { x, y } = this.renderer.worldToScreen(this._activeNode.x, this._activeNode.y);
    this.nodeDetails.style.left = Math.round(x + 20) + "px";
    this.nodeDetails.style.top = Math.round(y + 20) + "px";
    this.nodeDetails.classList.remove("top-4", "left-4");
};
