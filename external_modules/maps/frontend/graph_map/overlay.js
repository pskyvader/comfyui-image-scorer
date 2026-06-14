/**
 * UI Overlays (Tooltips, Details Panel, HUD) for ChainMapUI
 */

ChainMapUI.prototype.updateHUD = function(transform) {
    if (this.zoomScaleEl) this.zoomScaleEl.textContent = `${transform.k.toFixed(2)}x`;
    if (this.viewCoordsEl) {
        const centerX = Math.round((-transform.x + (this.width / 2)) / transform.k);
        const centerY = Math.round((-transform.y + (this.height / 2)) / transform.k);
        this.viewCoordsEl.textContent = `X: ${centerX}, Y: ${centerY}`;
    }
};

ChainMapUI.prototype.showTooltip = function(event, node) {
    this.tooltip.classList.remove("hidden");
    this.tooltip.style.left = `${event.pageX + 10}px`;
    this.tooltip.style.top = `${event.pageY + 10}px`;
    this.tooltip.innerHTML = `
        <div class="font-bold mb-1" style="cursor:pointer" onclick="window.chainMapUI.showNodeDetailsById('${node.id.replace(/'/g, "\\'")}')">${node.id.split('/').pop()}</div>
        <div>Score: ${node.score.toFixed(3)}</div>
        <div>Component: ${node._component_size}</div>
        <div>Comparisons: ${node.comparison_count}</div>
    `;
};

ChainMapUI.prototype.hideTooltip = function() {
    if (this._tooltipTimer) clearTimeout(this._tooltipTimer);
    this._tooltipTimer = setTimeout(() => {
        this.tooltip.classList.add("hidden");
    }, 200);
};

ChainMapUI.prototype.showNodeDetails = function(d) {
    try {
        if (!this.nodeDetails) return;
        this.nodeDetails.classList.remove("hidden");
        const filename = d.id.split('/').pop();
        this.nodeDetails.innerHTML = `
            <div class="flex justify-between items-start mb-2">
                <h3 class="text-white font-bold text-[10px] truncate max-w-[80%]">${filename}</h3>
                <button onclick="document.getElementById('node-details').classList.add('hidden')" class="text-gray-500 hover:text-white p-1">&times;</button>
            </div>
            <div class="flex flex-col gap-2">
                <img src="/images/${encodeURIComponent(d.id)}" class="w-full h-32 md:h-48 object-contain rounded bg-black/40 border border-white/5" onerror="this.src='/output/ranked/${encodeURIComponent(d.id)}'">
                
                <div class="flex flex-wrap gap-x-3 gap-y-1 text-[9px] text-gray-300 justify-center">
                    <span class="flex items-center gap-1">Score: <b class="text-purple-400">${d.score.toFixed(3)}</b></span>
                    <span class="flex items-center gap-1">Component: <b class="text-purple-400">${d._component_size}</b></span>
                    <span class="flex items-center gap-1">Comparisons: <b class="text-purple-400">${d.comparison_count}</b></span>
                </div>

                <div class="flex gap-2">
                    <button onclick="window.chainMapUI.toggleSelection('${d.id}')" class="flex-1 py-1.5 bg-pink-600 hover:bg-pink-500 rounded text-[9px] font-bold transition">
                        Select
                    </button>
                    <button onclick="window.chainMapUI.focusNode('${d.id}')" class="px-3 py-1.5 bg-white/10 hover:bg-white/20 rounded text-[9px] transition">
                        Focus
                    </button>
                </div>
            </div>
        `;
    } catch (e) {
        console.error("showNodeDetails error:", e);
    }
};

ChainMapUI.prototype.showLinkDetails = function(link) {
    try {
        if (!this.nodeDetails) return;
        this.nodeDetails.classList.remove("hidden");
        const srcName = link.source.id.split('/').pop();
        const tgtName = link.target.id.split('/').pop();
        this.nodeDetails.innerHTML = `
            <div class="flex justify-between items-start mb-2">
                <h3 class="text-white font-bold text-[10px]">Link</h3>
                <button onclick="document.getElementById('node-details').classList.add('hidden')" class="text-gray-500 hover:text-white p-1">&times;</button>
            </div>
            <div class="flex flex-col gap-2 text-[9px] text-gray-300">
                <div class="flex justify-between items-center p-2 bg-white/5 rounded">
                    <div>
                        <b class="text-purple-400">${srcName}</b>
                        <span class="text-gray-500 ml-1">score ${link.source.score.toFixed(3)}</span>
                    </div>
                    <span class="text-gray-600">&harr;</span>
                    <div class="text-right">
                        <b class="text-purple-400">${tgtName}</b>
                        <span class="text-gray-500 ml-1">score ${link.target.score.toFixed(3)}</span>
                    </div>
                </div>
                <div class="flex gap-2">
                    <button onclick="window.chainMapUI.focusNode('${link.source.id}')" class="flex-1 py-1.5 bg-white/10 hover:bg-white/20 rounded font-bold transition">
                        Focus Source
                    </button>
                    <button onclick="window.chainMapUI.focusNode('${link.target.id}')" class="flex-1 py-1.5 bg-white/10 hover:bg-white/20 rounded font-bold transition">
                        Focus Target
                    </button>
                </div>
            </div>
        `;
    } catch (e) {
        console.error("showLinkDetails error:", e);
    }
};

ChainMapUI.prototype.showNodeDetailsById = function(id) {
    const node = this.chainSim.nodes.find(n => n.id === id);
    if (node) this.showNodeDetails(node);
};

ChainMapUI.prototype.updateSelectionUI = function() {
    const count = this.selectedNodes.length;
    if (this.selectedCountEl) this.selectedCountEl.textContent = count;
    if (this.compareSelectedBtn) this.compareSelectedBtn.classList.toggle("hidden", count < 2);
};
